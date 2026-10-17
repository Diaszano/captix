"""Artifact verification rejects corrupt files and unsupported contracts offline."""

import hashlib
import json

import pytest

from captix_runtime.artifacts import ArtifactError, load_verified_manifest


@pytest.fixture
def artifact(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"synthetic graph")
    manifest = {
        "schema_version": 1,
        "model_version": "captix-v1-onnx-fp32",
        "exporter": "torchscript",
        "opset": 18,
        "model_sha256": hashlib.sha256(b"synthetic graph").hexdigest(),
        "source_commit": "a" * 40,
        "revision_sha": "b" * 40,
        "source_sha256": {"model.safetensors": "c" * 64, "config.json": "d" * 64},
        "versions": {
            name: "1.0" for name in ("torch", "onnx", "onnxscript", "onnxruntime")
        },
        "height": 32,
        "input_name": "images",
        "output_name": "logits",
        "dtype": "float32",
        "vocab": list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        "blank_index": 0,
        "width_range": [4, 1024],
        "input_scale": [0, 1],
        "input_layout": "NCHW",
        "output_layout": "TBC",
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path, manifest


def test_verified_manifest_exposes_contract_without_loading_graph(
    artifact, monkeypatch
):
    directory, _ = artifact
    monkeypatch.setattr(
        type(directory), "read_bytes", lambda self: pytest.fail("hash must stream")
    )
    manifest = load_verified_manifest(directory)
    assert manifest.input_name == "images"
    assert manifest.output_name == "logits"
    assert manifest.width_range == (4, 1024)
    assert len(manifest.vocab) == 62


def test_corrupted_graph_is_rejected(artifact):
    directory, _ = artifact
    (directory / "model.onnx").write_bytes(b"tampered")
    with pytest.raises(ArtifactError, match="model.onnx.*SHA256"):
        load_verified_manifest(directory)


@pytest.mark.parametrize("filename", ["manifest.json", "model.onnx"])
def test_missing_artifact_is_rejected(artifact, filename):
    directory, _ = artifact
    (directory / filename).unlink()
    with pytest.raises(ArtifactError):
        load_verified_manifest(directory)


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "model_version",
        "exporter",
        "opset",
        "model_sha256",
        "source_commit",
        "revision_sha",
        "source_sha256",
        "versions",
        "height",
        "input_name",
        "output_name",
        "dtype",
        "vocab",
        "blank_index",
        "width_range",
        "input_scale",
        "input_layout",
        "output_layout",
    ],
)
def test_missing_contract_field_is_rejected(artifact, field):
    directory, manifest = artifact
    manifest.pop(field)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ArtifactError, match=field):
        load_verified_manifest(directory)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("schema_version", 1.0),
        ("model_version", "other"),
        ("exporter", "dynamo"),
        ("opset", 17),
        ("model_sha256", "A" * 64),
        ("model_sha256", "a" * 64 + "\n"),
        ("source_commit", "main"),
        ("revision_sha", "B" * 40),
        ("source_sha256", {}),
        ("source_sha256", {"config.json": "x" * 64, "model.safetensors": "c" * 64}),
        ("source_sha256", {"config.json": "d" * 64}),
        ("versions", {}),
        ("versions", {"torch": "", "onnx": "1", "onnxscript": "1", "onnxruntime": "1"}),
        ("height", 31),
        ("height", 32.0),
        ("input_name", "input"),
        ("output_name", "output"),
        ("dtype", "float16"),
        ("vocab", ["0", "0"]),
        ("vocab", []),
        (
            "vocab",
            list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"),
        ),
        ("blank_index", 1),
        ("blank_index", False),
        ("width_range", [3, 1024]),
        ("width_range", [4, 1025]),
        ("width_range", [4.0, 1024]),
        ("input_scale", [-1, 1]),
        ("input_scale", [False, True]),
        ("input_layout", "NHWC"),
        ("output_layout", "BTC"),
        ("unexpected", 1),
    ],
)
def test_unsupported_contract_is_rejected(artifact, field, value):
    directory, manifest = artifact
    manifest[field] = value
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ArtifactError, match=field):
        load_verified_manifest(directory)


@pytest.mark.parametrize(
    "content", ["{", "[]", '{"schema_version":1,"schema_version":1}']
)
def test_invalid_or_ambiguous_json_is_rejected(artifact, content):
    directory, _ = artifact
    (directory / "manifest.json").write_text(content)
    with pytest.raises(ArtifactError):
        load_verified_manifest(directory)
