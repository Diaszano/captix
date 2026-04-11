from pathlib import Path

import pytest
from PIL import Image

from txtcaptcha import Captcha, read_captcha


def test_read_single_file(tmp_captcha_file):
    cap = read_captcha(tmp_captcha_file)
    assert isinstance(cap, Captcha)
    assert len(cap) == 1
    assert cap.paths == [Path(tmp_captcha_file)]
    assert cap.labels == [None]
    assert isinstance(cap.images[0], Image.Image)


def test_read_with_label_in_path(tmp_captcha_file):
    cap = read_captcha(tmp_captcha_file, lab_in_path=True)
    assert cap.labels == ["ab12"]


def test_read_unlabeled_with_lab_in_path(tmp_unlabeled_file):
    cap = read_captcha(tmp_unlabeled_file, lab_in_path=True)
    # No alphanumeric token after underscore matched fully -> 'label'
    # The regex extracts the substring after the first _
    assert cap.labels[0] is not None  # gets some label from filename


def test_read_multiple(tmp_labeled_dir):
    files = sorted(tmp_labeled_dir.glob("*.png"))
    cap = read_captcha(files, lab_in_path=True)
    assert len(cap) == len(files)
    assert all(lab is not None for lab in cap.labels)


def test_subset(tmp_labeled_dir):
    files = sorted(tmp_labeled_dir.glob("*.png"))
    cap = read_captcha(files, lab_in_path=True)
    sub = cap[:3]
    assert len(sub) == 3
    assert isinstance(sub, Captcha)
    one = cap[0]
    assert len(one) == 1


def test_read_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_captcha(tmp_path / "nope.png")
