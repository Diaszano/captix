import torch
from PIL import Image

from captix import (
    DEFAULT_VOCAB,
    CaptchaDataset,
    decode_indices,
    encode_label,
    pad_collate,
    transform_image,
)
from captix.dataset import BLANK_INDEX, _flatten_transparency


def test_default_vocab():
    assert len(DEFAULT_VOCAB) == 62
    assert "0" in DEFAULT_VOCAB and "a" in DEFAULT_VOCAB and "Z" in DEFAULT_VOCAB


def test_encode_decode_roundtrip():
    enc = encode_label("aB3")
    assert enc.dtype == torch.long
    assert (enc != BLANK_INDEX).all()
    dec = decode_indices(enc.tolist())
    assert dec == "aB3"


def test_encode_invalid_char():
    import pytest

    with pytest.raises(ValueError):
        encode_label("a$b")


def test_decode_collapses_repeats_and_blanks():
    # vocab idx 1='0' .. so simulate CTC: [1,1,0,1,2,2,0]
    # collapse: 1,1->1; 0; 1; 2,2->2; 0
    s = decode_indices([1, 1, 0, 1, 2, 2, 0])
    # collapse [1,1,0,1,2,2,0] -> [1,0,1,2,0] -> drop blanks -> [1,1,2] -> "001"
    assert s == "001"


def test_transform_image_single(tmp_captcha_file):
    t = transform_image(tmp_captcha_file)
    assert t.ndim == 4
    assert t.shape[0] == 1 and t.shape[1] == 3 and t.shape[2] == 32


def test_transform_image_batch_padding(tmp_labeled_dir):
    files = sorted(tmp_labeled_dir.glob("*.png"))
    t = transform_image(files[:3])
    assert t.shape[0] == 3 and t.shape[2] == 32


def test_dataset_basic(tmp_labeled_dir):
    ds = CaptchaDataset(tmp_labeled_dir)
    assert len(ds) >= 1
    img, target = ds[0]
    assert img.shape[0] == 3 and img.shape[1] == 32
    assert target.ndim == 1 and target.numel() > 0


def test_pad_collate(tmp_labeled_dir):
    ds = CaptchaDataset(tmp_labeled_dir)
    batch = [ds[i] for i in range(min(4, len(ds)))]
    images, targets, lengths = pad_collate(batch)
    assert images.shape[0] == len(batch)
    assert lengths.sum().item() == targets.numel()


def _make_rgba(fg_color, size=(40, 20)):
    """Build an RGBA image with opaque glyphs in `fg_color` on a transparent
    background. A few columns are set to the foreground color."""
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    px = img.load()
    for x in range(5, 15):
        for y in range(5, 15):
            px[x, y] = (*fg_color, 255)
    return img


def test_flatten_transparency_dark_fg_gets_white_bg():
    img = _make_rgba(fg_color=(0, 0, 0))
    flat = _flatten_transparency(img)
    assert flat.mode == "RGB"
    # corners were transparent — must now be white, not black
    assert flat.getpixel((0, 0)) == (255, 255, 255)
    # foreground pixels preserved
    assert flat.getpixel((10, 10)) == (0, 0, 0)


def test_flatten_transparency_light_fg_gets_black_bg():
    img = _make_rgba(fg_color=(240, 240, 240))
    flat = _flatten_transparency(img)
    assert flat.getpixel((0, 0)) == (0, 0, 0)
    assert flat.getpixel((10, 10)) == (240, 240, 240)


def test_flatten_transparency_opaque_image_is_noop():
    img = Image.new("RGB", (10, 10), (128, 64, 32))
    flat = _flatten_transparency(img)
    assert flat.mode == "RGB"
    assert flat.getpixel((0, 0)) == (128, 64, 32)


def test_transform_image_handles_transparent_png(tmp_path):
    img = _make_rgba(fg_color=(0, 0, 0))
    path = tmp_path / "cap.png"
    img.save(path)
    tensor = transform_image(path)
    # After transform, the top-left area came from transparent pixels and
    # should be near-white rather than black.
    assert tensor.shape[0] == 1 and tensor.shape[1] == 3
    corner = tensor[0, :, 0, 0].mean().item()
    assert corner > 0.9, f"corner should be near-white, got {corner}"
