"""Shared pytest fixtures for captix tests."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw


def _make_image(text: str, size=(80, 32), color="white") -> Image.Image:
    img = Image.new("RGB", size, color=color)
    draw = ImageDraw.Draw(img)
    draw.text((4, 8), text, fill="black")
    return img


@pytest.fixture
def tmp_captcha_file(tmp_path: Path) -> Path:
    p = tmp_path / "abc1_ab12.png"
    _make_image("ab12").save(p)
    return p


@pytest.fixture
def tmp_unlabeled_file(tmp_path: Path) -> Path:
    p = tmp_path / "no_label_here.png"
    _make_image("xx").save(p)
    return p


@pytest.fixture
def tmp_labeled_dir(tmp_path: Path) -> Path:
    """Directory with several labeled captchas (4-char labels)."""
    d = tmp_path / "labeled"
    d.mkdir()
    labels = ["ab12", "cd34", "ef56", "gh78", "ij90", "kl12", "mn34", "op56"]
    for i, lab in enumerate(labels):
        _make_image(lab).save(d / f"img{i:03d}_{lab}.png")
    return d
