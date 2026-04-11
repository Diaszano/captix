from pathlib import Path

import pytest
from PIL import Image

from txtcaptcha import annotate


def _make(p: Path):
    Image.new("RGB", (40, 20), "white").save(p)


def test_annotate_with_labels(tmp_path):
    f1 = tmp_path / "a.png"; _make(f1)
    f2 = tmp_path / "b.png"; _make(f2)
    out = annotate([f1, f2], labels=["xx11", "yy22"])
    assert all(p.exists() for p in out)
    assert out[0].name == "a_xx11.png"
    assert out[1].name == "b_yy22.png"


def test_annotate_to_path(tmp_path):
    f = tmp_path / "img.png"; _make(f)
    target = tmp_path / "out"
    out = annotate(f, labels=["abc"], path=target)
    assert out[0].parent == target


def test_annotate_rm_old(tmp_path):
    f = tmp_path / "img.png"; _make(f)
    annotate(f, labels=["abc"], rm_old=True)
    assert not f.exists()


def test_annotate_prompt(tmp_path):
    f = tmp_path / "img.png"; _make(f)
    out = annotate(f, prompt=lambda p: "ZZZZ")
    assert out[0].name == "img_ZZZZ.png"


def test_annotate_label_count_mismatch(tmp_path):
    f = tmp_path / "img.png"; _make(f)
    with pytest.raises(ValueError):
        annotate([f], labels=["a", "b"])
