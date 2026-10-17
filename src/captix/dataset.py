"""Dataset, transforms and label encoding for captix.

The model uses CTC loss, so labels are encoded as integer sequences with a
reserved blank index 0. The default vocabulary is the full alphanumeric set
``0-9a-zA-Z`` (62 classes), giving 63 logits including blank.
"""
from __future__ import annotations

import re
import string
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

PathLike = Union[str, Path]

# Default vocabulary: digits + lowercase + uppercase
DEFAULT_VOCAB: List[str] = list(string.digits + string.ascii_lowercase + string.ascii_uppercase)

# Index 0 is reserved as the CTC blank
BLANK_INDEX = 0

_LABEL_RE = re.compile(r"(?<=_)[0-9a-zA-Z]+")


def _build_char_to_idx(vocab: Sequence[str]) -> dict:
    return {ch: i + 1 for i, ch in enumerate(vocab)}  # +1 to skip blank


def encode_label(label: str, vocab: Sequence[str] = DEFAULT_VOCAB) -> torch.Tensor:
    """Encode a string label into a 1D LongTensor of vocab indices (no blanks)."""
    c2i = _build_char_to_idx(vocab)
    try:
        return torch.tensor([c2i[c] for c in label], dtype=torch.long)
    except KeyError as e:
        raise ValueError(f"Character {e!s} not in vocabulary") from None


def decode_indices(
    indices: Sequence[int],
    vocab: Sequence[str] = DEFAULT_VOCAB,
    collapse: bool = True,
) -> str:
    """Decode a CTC index sequence to a string.

    If `collapse` is True, applies CTC collapse (remove repeats then blanks).
    """
    out_chars: List[str] = []
    prev = -1
    for i in indices:
        i = int(i)
        if collapse:
            if i == prev:
                continue
            prev = i
        if i == BLANK_INDEX:
            continue
        out_chars.append(vocab[i - 1])
    return "".join(out_chars)


def _flatten_transparency(img: Image.Image) -> Image.Image:
    """If ``img`` has an alpha channel, composite it onto a solid background
    whose luminance contrasts with the average color of the visible pixels,
    then return an RGB image. Otherwise return ``img.convert('RGB')``.

    PIL's default ``convert('RGB')`` drops alpha by pasting onto black — so a
    captcha drawn as black glyphs on a transparent PNG (TJPE, some PJe
    installs) collapses to an all-black image and the model sees nothing.
    Picking the background as the tonal inverse of the mean visible color
    recovers the contrast regardless of whether the glyphs are dark or light.
    """
    has_alpha = img.mode in ("RGBA", "LA") or (
        img.mode == "P" and "transparency" in img.info
    )
    if not has_alpha:
        return img.convert("RGB")
    rgba = img.convert("RGBA")
    arr = np.asarray(rgba)
    visible = arr[..., 3] > 0
    if visible.any():
        mean_lum = float(arr[..., :3][visible].mean())
    else:
        mean_lum = 0.0
    bg_value = 255 if mean_lum < 128 else 0
    bg = Image.new("RGB", rgba.size, (bg_value, bg_value, bg_value))
    bg.paste(rgba, mask=rgba.split()[-1])
    return bg


def transform_image(
    files: Union[PathLike, Sequence[PathLike], Image.Image, Sequence[Image.Image]],
    height: int = 32,
    max_width: Optional[int] = None,
) -> torch.Tensor:
    """Load and resize image(s) to a fixed height, preserving aspect ratio.

    Returns a 4D tensor (B, 3, H, W). When passing a batch with different widths,
    images are padded to the maximum width in the batch (or `max_width` if given).
    """
    if isinstance(files, (str, Path, Image.Image)):
        files_list = [files]
    else:
        files_list = list(files)

    to_tensor = T.ToTensor()
    tensors: List[torch.Tensor] = []
    for f in files_list:
        if isinstance(f, Image.Image):
            img = _flatten_transparency(f)
        else:
            img = _flatten_transparency(Image.open(f))
        w, h = img.size
        new_w = max(1, int(round(w * (height / h))))
        img = img.resize((new_w, height), Image.BILINEAR)
        tensors.append(to_tensor(img))

    target_w = max_width if max_width is not None else max(t.shape[-1] for t in tensors)
    padded = []
    for t in tensors:
        if t.shape[-1] < target_w:
            pad = torch.zeros(3, height, target_w - t.shape[-1])
            t = torch.cat([t, pad], dim=-1)
        elif t.shape[-1] > target_w:
            t = t[..., :target_w]
        padded.append(t)
    return torch.stack(padded, dim=0)


def _list_files(root: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"}
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def _label_from_path(p: Path) -> Optional[str]:
    m = _LABEL_RE.search(p.stem)
    return m.group(0) if m else None


class CaptchaDataset(Dataset):
    """Dataset of labeled captcha images.

    Files must follow the pattern ``<id>_<label>.<ext>``. Labels can have
    different lengths (CTC handles variable-length sequences).
    """

    def __init__(
        self,
        root: PathLike,
        vocab: Sequence[str] = DEFAULT_VOCAB,
        height: int = 32,
        case_sensitive: bool = True,
    ):
        self.root = Path(root)
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)
        self.height = height
        self.vocab = list(vocab)
        self.case_sensitive = case_sensitive
        self.files: List[Path] = []
        self.labels: List[str] = []
        for f in _list_files(self.root):
            lab = _label_from_path(f)
            if lab is None:
                continue
            if not case_sensitive:
                lab = lab.lower()
            self.files.append(f)
            self.labels.append(lab)
        if not self.files:
            raise ValueError(f"No labeled captcha files found in {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_t = transform_image(self.files[idx], height=self.height)[0]  # (3, H, W)
        target = encode_label(self.labels[idx], self.vocab)
        return img_t, target


def pad_collate(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function: pads images to the max width and stacks targets.

    Returns
    -------
    images: (B, 3, H, W_max)
    targets: 1D tensor concatenating all label sequences
    target_lengths: (B,) lengths for CTC loss
    """
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    H = images[0].shape[1]
    W_max = max(im.shape[-1] for im in images)
    out = torch.zeros(len(images), 3, H, W_max)
    for i, im in enumerate(images):
        out[i, :, :, : im.shape[-1]] = im
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets_cat = torch.cat(targets) if targets else torch.zeros(0, dtype=torch.long)
    return out, targets_cat, target_lengths
