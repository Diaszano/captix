import torch

from txtcaptcha import (
    CaptchaDataset,
    DEFAULT_VOCAB,
    decode_indices,
    encode_label,
    pad_collate,
    transform_image,
)
from txtcaptcha.dataset import BLANK_INDEX


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
