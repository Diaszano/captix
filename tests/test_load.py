import pytest
import torch

from captix import CRNN, available_models, load_model, save_model
from captix.data import available_datasets


def test_save_and_load_roundtrip(tmp_path):
    model = CRNN()
    p = tmp_path / "m.pt"
    save_model(model, p)
    assert p.exists()
    loaded = load_model(p)
    assert loaded.num_classes == model.num_classes
    # weights match
    for a, b in zip(model.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(a, b)


def test_load_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nope.pt")


def test_load_invalid_format(tmp_path):
    p = tmp_path / "bad.pt"
    torch.save({"foo": "bar"}, p)
    with pytest.raises(ValueError):
        load_model(p)


def test_available_models_lists_legacy():
    names = available_models()
    assert "tjmg" in names and "rfb" in names


def test_available_datasets():
    names = available_datasets()
    assert "tjmg" in names and len(names) == 10
