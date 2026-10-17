"""Pinned downloads and local synthetic checkpoint export; no Hub access."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from captix.hub import load_local
from captix.model import CRNN
from scripts.export_onnx import export_checkpoint
from scripts.fetch_checkpoint import fetch_checkpoint

REVISION = "a" * 40
SOURCE_COMMIT = "b" * 40


@pytest.mark.parametrize(
    "revision", ["main", "v1.0", "a" * 39, "A" * 40, "a" * 40 + "\n"]
)
def test_fetch_rejects_mutable_or_malformed_revision(tmp_path, monkeypatch, revision):
    def forbidden_download(**kwargs):
        pytest.fail("invalid revision reached the network")

    monkeypatch.setattr("huggingface_hub.snapshot_download", forbidden_download)
    with pytest.raises(ValueError, match="SHA"):
        fetch_checkpoint("jtrecenti/txtcaptcha-crnn", revision, tmp_path / "checkpoint")
    assert not (tmp_path / "checkpoint").exists()


def test_fetch_downloads_only_pinned_checkpoint_files(tmp_path, monkeypatch):
    destination = tmp_path / "checkpoint"

    def download(*, repo_id, revision, local_dir, allow_patterns):
        assert repo_id == "jtrecenti/txtcaptcha-crnn"
        assert revision == REVISION
        assert Path(local_dir) == destination
        assert allow_patterns == ["model.safetensors", "config.json"]
        destination.mkdir()
        for name in allow_patterns:
            (destination / name).write_bytes(b"checkpoint")
        return str(destination)

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    assert (
        fetch_checkpoint("jtrecenti/txtcaptcha-crnn", REVISION, destination)
        == destination
    )
    assert {p.name for p in destination.iterdir()} == {
        "model.safetensors",
        "config.json",
    }


def test_fetch_refuses_to_mix_revisions_in_existing_destination(tmp_path, monkeypatch):
    def forbidden_download(**kwargs):
        pytest.fail("existing checkpoint must be protected")

    monkeypatch.setattr("huggingface_hub.snapshot_download", forbidden_download)
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"existing weights")
    with pytest.raises(FileExistsError):
        fetch_checkpoint("jtrecenti/txtcaptcha-crnn", REVISION, tmp_path)
    assert weights.read_bytes() == b"existing weights"


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    def forbidden_download(*args, **kwargs):
        pytest.fail("export must remain offline")

    monkeypatch.setattr("huggingface_hub.snapshot_download", forbidden_download)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", forbidden_download)
    source = tmp_path / "checkpoint"
    source.mkdir()
    torch.manual_seed(7)
    model = CRNN(
        vocab=list("ab"), cnn_channels=(2, 2, 2, 2), rnn_hidden=2, rnn_layers=2
    )
    save_file(model.state_dict(), str(source / "model.safetensors"))
    config = {
        "architecture": "CRNN",
        "vocab": list("ab"),
        "cnn_channels": [2, 2, 2, 2],
        "rnn_hidden": 2,
        "rnn_layers": 2,
        "dropout": 0.25,
        "img_height": 32,
        "blank_index": 0,
    }
    (source / "config.json").write_text(json.dumps(config))
    return source


@pytest.mark.parametrize("argument", ["revision_sha", "source_commit"])
def test_export_rejects_unpinned_provenance(checkpoint, tmp_path, argument):
    provenance = {"revision_sha": REVISION, "source_commit": SOURCE_COMMIT}
    provenance[argument] = "main"
    with pytest.raises(ValueError, match="SHA"):
        export_checkpoint(checkpoint, tmp_path / "out", **provenance)
    assert not (tmp_path / "out").exists()


def test_failed_export_does_not_publish_partial_artifacts(
    checkpoint, tmp_path, monkeypatch
):
    def failed_export(*args, **kwargs):
        Path(args[2]).write_bytes(b"partial ONNX")
        raise RuntimeError("dynamic LSTM export failed")

    monkeypatch.setattr(torch.onnx, "export", failed_export)
    output = tmp_path / "out"
    with pytest.raises(RuntimeError, match="dynamic LSTM"):
        export_checkpoint(
            checkpoint, output, revision_sha=REVISION, source_commit=SOURCE_COMMIT
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".captix-export-*"))


def test_export_does_not_overwrite_existing_artifacts(checkpoint, tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    sentinel = output / "model.onnx"
    sentinel.write_bytes(b"existing artifact")
    with pytest.raises(FileExistsError):
        export_checkpoint(
            checkpoint, output, revision_sha=REVISION, source_commit=SOURCE_COMMIT
        )
    assert sentinel.read_bytes() == b"existing artifact"


@pytest.mark.parametrize("width", [1, 7, 40, 256])
def test_height_mean_pooling_matches_adaptive_pool(width):
    from scripts.export_onnx import HeightMean

    features = torch.randn(1, 2, 4, width)
    expected = torch.nn.AdaptiveAvgPool2d((1, None))(features)
    torch.testing.assert_close(HeightMean()(features), expected, atol=1e-6, rtol=1e-6)


def test_export_local_crnn_preserves_dynamic_width_and_manifest(checkpoint, tmp_path):
    output = tmp_path / "out"
    result = export_checkpoint(
        checkpoint, output, revision_sha=REVISION, source_commit=SOURCE_COMMIT
    )
    assert result == output / "model.onnx"
    graph = onnx.load(result)
    onnx.checker.check_model(graph)
    assert len(graph.graph.input) == len(graph.graph.output) == 1
    inp, out = graph.graph.input[0], graph.graph.output[0]
    assert (inp.name, out.name) == ("images", "logits")
    dims = inp.type.tensor_type.shape.dim
    assert [d.dim_value for d in dims[:3]] == [1, 3, 32]
    assert dims[3].dim_param
    assert out.type.tensor_type.shape.dim[1].dim_value == 1
    assert (
        inp.type.tensor_type.elem_type
        == out.type.tensor_type.elem_type
        == onnx.TensorProto.FLOAT
    )

    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(result), options, providers=["CPUExecutionProvider"]
    )
    reference = load_local(checkpoint)
    for width in (4, 31, 160, 1024):
        images = torch.rand(1, 3, 32, width)
        with torch.inference_mode():
            expected = reference(images).numpy()
        actual = session.run(["logits"], {"images": images.numpy()})[0]
        assert actual.shape == (width // 4, 1, 3)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["model_version"] == "captix-v1-onnx-fp32"
    assert manifest["model_sha256"] == hashlib.sha256(result.read_bytes()).hexdigest()
    assert manifest["source_commit"] == SOURCE_COMMIT
    assert manifest["revision_sha"] == REVISION
    for name in ("model.safetensors", "config.json"):
        assert (
            manifest["source_sha256"][name]
            == hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
        )
    assert all(
        manifest["versions"][name]
        for name in ("torch", "onnx", "onnxscript", "onnxruntime")
    )
    assert manifest["height"] == 32
    assert manifest["input_name"] == "images"
    assert manifest["output_name"] == "logits"
    assert manifest["dtype"] == "float32"
    assert manifest["vocab"] == ["a", "b"]
    assert manifest["blank_index"] == 0
    assert manifest["width_range"] == [4, 1024]
    assert manifest["input_scale"] == [0, 1]
    assert manifest["input_layout"] == "NCHW"
    assert manifest["output_layout"] == "TBC"
