import tomllib

import numpy as np
import pytest
import torch
from PIL import Image

from captix import CRNN
from captix.decrypt import _ctc_decode_fixed_length
from captix.dataset import transform_image


def test_distribution_identity_and_wheel_package():
    with open("pyproject.toml", "rb") as f:
        project = tomllib.load(f)
    assert project["project"]["name"] == "captix"
    assert project["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["src/captix", "src/captix_runtime"]
    assert "src/captix_runtime" in project["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]


@pytest.mark.parametrize(("width", "timesteps"), [(80, 20), (160, 40)])
def test_network_output_shapes(width, timesteps):
    model = CRNN(cnn_channels=(4, 8, 8, 8), rnn_hidden=4, rnn_layers=1).eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 32, width))
    assert tuple(output.shape) == (timesteps, 1, 63)


def test_deterministic_png_and_rgba_transform_fixtures():
    rgb = Image.new("RGB", (8, 4), "white")
    rgba = Image.new("RGBA", (8, 4), (0, 0, 0, 0))
    rgba.putpixel((2, 2), (0, 0, 0, 255))
    assert tuple(transform_image([rgb, rgba], height=32).shape) == (2, 3, 32, 64)


def test_ctc_paths_are_frozen():
    cases = [
        (np.array([[0.1, 0.8, 0.1], [0.1, 0.8, 0.1], [0.8, 0.1, 0.1]]), 1, [1, 1, 0]),
        (np.array([[0.1, 0.8, 0.1], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]), 2, [1, 0, 1]),
    ]
    for log_probs, length, expected in cases:
        assert _ctc_decode_fixed_length(log_probs, length) == expected


def test_baseline_fixture_bytes_are_stable(tmp_path):
    image = Image.new("RGBA", (2, 2), (1, 2, 3, 4))
    path = tmp_path / "synthetic.png"
    image.save(path, format="PNG", optimize=False)
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
