"""decrypt: predict labels for captcha images using a trained CRNN model."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch

from .captcha import Captcha
from .dataset import BLANK_INDEX, decode_indices, transform_image
from .model import CRNN

PathLike = Union[str, Path]
MaskType = Optional[Union[Sequence[str], str, re.Pattern]]


def _resolve_mask(
    mask: MaskType,
    vocab: Sequence[str],
    case_sensitive: bool,
) -> Optional[List[bool]]:
    """Returns a boolean list of length len(vocab): True for allowed chars.

    None means "no masking".
    """
    if mask is None:
        return None

    if isinstance(mask, (list, tuple, set)):
        allowed = set(mask)
        if not case_sensitive:
            allowed = {c.lower() for c in allowed} | {c.upper() for c in allowed}
        return [c in allowed for c in vocab]

    if isinstance(mask, re.Pattern):
        pat = mask
    elif isinstance(mask, str):
        # Treat as a regex; if not a character class, wrap it.
        pat_src = mask if mask.startswith("[") else f"[{mask}]"
        flags = 0 if case_sensitive else re.IGNORECASE
        pat = re.compile(pat_src, flags)
    else:
        raise TypeError(f"Unsupported mask type: {type(mask)!r}")

    return [bool(pat.fullmatch(c)) for c in vocab]


def _apply_mask(logits: torch.Tensor, allowed: Optional[List[bool]]) -> torch.Tensor:
    """Set logits of forbidden classes to -inf. Blank stays untouched."""
    if allowed is None:
        return logits
    mask_t = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
    mask_t[BLANK_INDEX] = True  # always allow blank
    for i, ok in enumerate(allowed):
        if ok:
            mask_t[i + 1] = True
    forbidden = ~mask_t
    return logits.masked_fill(forbidden, float("-inf"))


@torch.no_grad()
def decrypt(
    files: Union[PathLike, Sequence[PathLike], Captcha],
    model: CRNN,
    mask: MaskType = None,
    case_sensitive: bool = True,
    device: Optional[Union[str, torch.device]] = None,
) -> List[str]:
    """Predict labels for one or more captcha images.

    Parameters
    ----------
    files: file path, list of paths, or a Captcha object.
    model: a trained `CRNN` instance.
    mask: optional restriction on output characters. Either a list of allowed
        characters (e.g. ``['a','b','c']``), a regex character class
        (e.g. ``'[0-9a-z]'`` or ``'0-9'``), or a compiled `re.Pattern`.
    case_sensitive: when False and a list/string mask is given, both cases
        are accepted; output is also lowercased.
    device: torch device for inference. Defaults to model's current device.
    """
    if isinstance(files, Captcha):
        path_list: List[Union[str, Path]] = list(files.paths)
    elif isinstance(files, (str, Path)):
        path_list = [files]
    else:
        path_list = list(files)

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    model.eval()
    images = transform_image(path_list).to(device)
    logits = model(images)  # (T, B, C)
    allowed = _resolve_mask(mask, model.vocab, case_sensitive)
    logits = _apply_mask(logits, allowed)
    indices = logits.argmax(dim=-1).transpose(0, 1)  # (B, T)

    results: List[str] = []
    for row in indices:
        text = decode_indices(row.tolist(), model.vocab, collapse=True)
        if not case_sensitive:
            text = text.lower()
        results.append(text)
    return results
