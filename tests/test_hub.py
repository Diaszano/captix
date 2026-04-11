"""Tests for txtcaptcha.hub and the HF-backed decrypt fallback.

All HF network calls are mocked — tests must not touch the real Hub.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from txtcaptcha import CRNN, decrypt
from txtcaptcha.hub import (
    CONFIG_FILENAME,
    WEIGHTS_FILENAME,
    _model_config,
    _model_from_config,
    _resolve_token,
    from_pretrained,
    load_local,
    save_pretrained,
)
import importlib

decrypt_mod = importlib.import_module("txtcaptcha.decrypt")  # the module


# ---------------------------------------------------------------- save/load


def test_save_pretrained_writes_expected_files(tmp_path):
    model = CRNN()
    out = save_pretrained(model, tmp_path, model_card="# hi")
    assert out == tmp_path
    assert (tmp_path / WEIGHTS_FILENAME).is_file()
    assert (tmp_path / CONFIG_FILENAME).is_file()
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "# hi"


def test_save_pretrained_config_has_expected_fields(tmp_path):
    model = CRNN()
    save_pretrained(model, tmp_path)
    import json
    cfg = json.loads((tmp_path / CONFIG_FILENAME).read_text(encoding="utf-8"))
    assert cfg["architecture"] == "CRNN"
    assert cfg["library_name"] == "txtcaptcha"
    assert cfg["rnn_hidden"] == 256
    assert cfg["rnn_layers"] == 2
    assert cfg["blank_index"] == 0
    assert len(cfg["vocab"]) == 62


def test_save_load_roundtrip_preserves_weights(tmp_path):
    model = CRNN()
    save_pretrained(model, tmp_path)
    loaded = load_local(tmp_path)
    # compare every parameter
    for (n1, p1), (n2, p2) in zip(
        model.state_dict().items(), loaded.state_dict().items()
    ):
        assert n1 == n2
        assert torch.allclose(p1, p2)


def test_save_load_preserves_forward(tmp_path):
    model = CRNN()
    model.eval()
    save_pretrained(model, tmp_path)
    loaded = load_local(tmp_path).eval()
    x = torch.randn(1, 3, 32, 96)
    with torch.no_grad():
        a = model(x)
        b = loaded(x)
    assert torch.allclose(a, b, atol=1e-5)


def test_model_config_roundtrip():
    model = CRNN(rnn_hidden=128, rnn_layers=1, dropout=0.1)
    cfg = _model_config(model)
    rebuilt = _model_from_config(cfg)
    assert rebuilt.rnn.hidden_size == 128
    assert rebuilt.rnn.num_layers == 1


# ------------------------------------------------------------ token resolution


def test_resolve_token_explicit_wins(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "env-token")
    assert _resolve_token("explicit") == "explicit"


def test_resolve_token_hf_token(monkeypatch):
    monkeypatch.delenv("HF_API_KEY", raising=False)
    monkeypatch.setenv("HF_TOKEN", "t1")
    assert _resolve_token() == "t1"


def test_resolve_token_hf_api_key_fallback(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_API_KEY", "api-k")
    assert _resolve_token() == "api-k"


def test_resolve_token_none(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_API_KEY", raising=False)
    assert _resolve_token() is None


# ----------------------------------------------------------- from_pretrained


def test_from_pretrained_downloads_and_loads(tmp_path):
    # Stage a valid saved model in a fake cache folder
    model = CRNN()
    save_pretrained(model, tmp_path)

    def fake_download(repo_id, filename, **kw):
        return str(tmp_path / filename)

    with patch("huggingface_hub.hf_hub_download", side_effect=fake_download):
        loaded = from_pretrained("fake/repo")
    assert isinstance(loaded, CRNN)
    assert list(loaded.vocab) == list(model.vocab)


def test_from_pretrained_passes_revision(tmp_path):
    model = CRNN()
    save_pretrained(model, tmp_path)
    seen = {}

    def fake_download(repo_id, filename, **kw):
        seen.setdefault(filename, kw)
        return str(tmp_path / filename)

    with patch("huggingface_hub.hf_hub_download", side_effect=fake_download):
        from_pretrained("fake/repo", revision="v1.2.3", token="t")
    assert seen[WEIGHTS_FILENAME]["revision"] == "v1.2.3"
    assert seen[WEIGHTS_FILENAME]["token"] == "t"


# --------------------------------------------------------- push_to_hub


def test_push_to_hub_without_token_raises(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_API_KEY", raising=False)
    from txtcaptcha.hub import push_to_hub

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        push_to_hub(CRNN(), repo_id="fake/repo")


def test_push_to_hub_calls_create_and_upload(monkeypatch):
    calls = {"create": None, "upload": None, "tag": None}

    class FakeApi:
        def __init__(self, token=None):
            calls["init_token"] = token

        def create_repo(self, repo_id, repo_type, exist_ok, private):
            calls["create"] = (repo_id, repo_type, exist_ok, private)

        def upload_folder(self, folder_path, repo_id, repo_type, commit_message):
            # Verify save_pretrained actually produced the expected files
            p = Path(folder_path)
            assert (p / WEIGHTS_FILENAME).exists()
            assert (p / CONFIG_FILENAME).exists()
            assert (p / "README.md").exists()
            calls["upload"] = repo_id

        def create_tag(self, repo_id, tag, repo_type):
            calls["tag"] = tag

    with patch("huggingface_hub.HfApi", FakeApi):
        from txtcaptcha.hub import push_to_hub
        url = push_to_hub(
            CRNN(),
            repo_id="user/repo",
            token="t",
            model_card="# card",
            tag="v1.0",
        )
    assert url == "https://huggingface.co/user/repo"
    assert calls["create"] == ("user/repo", "model", True, False)
    assert calls["upload"] == "user/repo"
    assert calls["tag"] == "v1.0"


# --------------------------------------------------- decrypt auto-download


def test_decrypt_auto_downloads_when_model_none(tmp_path, tmp_captcha_file):
    model = CRNN()
    # Clear any prior cache entries so the patch is exercised
    decrypt_mod._DEFAULT_MODEL_CACHE.clear()

    calls = {"n": 0}

    def fake_from_pretrained(repo_id, **kw):
        calls["n"] += 1
        calls["repo"] = repo_id
        return model

    with patch("txtcaptcha.hub.from_pretrained", side_effect=fake_from_pretrained):
        out1 = decrypt(tmp_captcha_file)
        out2 = decrypt(tmp_captcha_file)  # should hit the cache

    assert len(out1) == 1
    assert len(out2) == 1
    assert calls["n"] == 1  # only downloaded once
    from txtcaptcha.hub import DEFAULT_REPO_ID
    assert calls["repo"] == DEFAULT_REPO_ID


def test_decrypt_with_string_model_loads_that_repo(tmp_captcha_file):
    model = CRNN()
    decrypt_mod._DEFAULT_MODEL_CACHE.clear()

    calls = {}

    def fake_from_pretrained(repo_id, **kw):
        calls["repo"] = repo_id
        return model

    with patch("txtcaptcha.hub.from_pretrained", side_effect=fake_from_pretrained):
        decrypt(tmp_captcha_file, model="custom/repo")
    assert calls["repo"] == "custom/repo"


def test_decrypt_with_crnn_instance_skips_download(tmp_captcha_file):
    model = CRNN()
    decrypt_mod._DEFAULT_MODEL_CACHE.clear()

    def boom(*a, **k):
        raise AssertionError("should not be called")

    with patch("txtcaptcha.hub.from_pretrained", side_effect=boom):
        out = decrypt(tmp_captcha_file, model=model)
    assert len(out) == 1
