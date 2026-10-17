"""Pinned downloads and local synthetic checkpoint export; no Hub access."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from PIL import Image

from captix.dataset import DEFAULT_VOCAB
from captix.hub import load_local
from captix.model import CRNN
from captix_runtime.artifacts import ArtifactError
from scripts.export_onnx import export_checkpoint
from scripts.fetch_checkpoint import fetch_checkpoint
from scripts.validate_onnx import main as validate_main
from scripts.validate_onnx import validate_equivalence

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
        vocab=DEFAULT_VOCAB, cnn_channels=(2, 2, 2, 2), rnn_hidden=2, rnn_layers=2
    )
    save_file(model.state_dict(), str(source / "model.safetensors"))
    config = {
        "architecture": "CRNN",
        "vocab": list(DEFAULT_VOCAB),
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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("architecture", None, id="missing-architecture"),
        pytest.param("architecture", "Transformer", id="invalid-architecture"),
        pytest.param("architecture", ["CRNN"], id="non-string-architecture"),
        pytest.param("vocab", None, id="missing-vocab"),
        pytest.param("vocab", [], id="empty-vocab"),
        pytest.param("vocab", DEFAULT_VOCAB[::-1], id="reordered-vocab"),
        pytest.param("vocab", DEFAULT_VOCAB[:-1], id="truncated-vocab"),
        pytest.param("vocab", DEFAULT_VOCAB + ["0"], id="duplicate-vocab"),
        pytest.param("vocab", DEFAULT_VOCAB[:-1] + ["?"], id="invalid-vocab"),
        pytest.param("vocab", "".join(DEFAULT_VOCAB), id="non-list-vocab"),
    ],
)
def test_export_rejects_unsupported_config_before_loading(
    checkpoint, tmp_path, monkeypatch, field, value
):
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    if value is None:
        config.pop(field)
    else:
        config[field] = value
    config_path.write_text(json.dumps(config))

    def forbidden_load(*args, **kwargs):
        pytest.fail("unsupported checkpoint config reached model loading")

    monkeypatch.setattr("captix.hub.load_local", forbidden_load)
    output = tmp_path / "out"
    with pytest.raises(ValueError, match=field):
        export_checkpoint(
            checkpoint, output, revision_sha=REVISION, source_commit=SOURCE_COMMIT
        )
    assert not output.exists()


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
        assert actual.shape == (width // 4, 1, 63)
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
    assert manifest["vocab"] == list(
        "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )
    assert manifest["blank_index"] == 0
    assert manifest["width_range"] == [4, 1024]
    assert manifest["input_scale"] == [0, 1]
    assert manifest["input_layout"] == "NCHW"
    assert manifest["output_layout"] == "TBC"


@pytest.fixture
def exported(checkpoint, tmp_path):
    directory = tmp_path / "out"
    export_checkpoint(
        checkpoint, directory, revision_sha=REVISION, source_commit=SOURCE_COMMIT
    )
    return directory


def validation_inputs():
    images = []
    for width in (80, 160, 240):
        buffer = BytesIO()
        Image.new("RGB", (width, 32), color=(90, 120, 150)).save(buffer, format="PNG")
        images.append(buffer.getvalue())
    return images


def rewrite_graph(directory, graph):
    onnx.save(graph, directory / "model.onnx")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model_sha256"] = hashlib.sha256(
        (directory / "model.onnx").read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


def test_validator_compares_dynamic_logits_and_reference_decoders(checkpoint, exported):
    inputs = validation_inputs()
    report = validate_equivalence(checkpoint, exported, inputs, atol=1e-4, rtol=1e-3)
    assert report.status == "PASS"
    assert report.max_abs_diff < 1e-4
    assert report.max_rel_diff >= 0
    assert report.versions["validation"]["onnxruntime"] == ort.__version__
    assert report.source_commit == SOURCE_COMMIT
    assert report.revision_sha == REVISION
    assert [item.width for item in report.examples] == [80, 160, 240]
    for index, (item, data) in enumerate(zip(report.examples, inputs)):
        assert item.example_id == str(index)
        assert item.sha256 == hashlib.sha256(data).hexdigest()
        assert item.shape == (item.width // 4, 1, 63)
        assert item.predictions
        assert {prediction.mask for prediction in item.predictions} >= {
            None,
            "[0-9]",
            "[a-z]",
            "[A-Z]",
        }
        assert {prediction.length for prediction in item.predictions} == {None, 4}
        assert {prediction.case_sensitive for prediction in item.predictions} == {
            True,
            False,
        }
        assert all(
            prediction.reference == prediction.onnx for prediction in item.predictions
        )
    serialized = json.dumps(report.summary())
    assert "reference" not in serialized and "predictions" not in serialized
    assert "PNG" not in serialized


@pytest.mark.parametrize("delta", [0.1, float("nan")])
def test_numerical_divergence_produces_hold(checkpoint, exported, delta):
    graph = onnx.load(exported / "model.onnx")
    for node in graph.graph.node:
        for index, name in enumerate(node.output):
            if name == "logits":
                node.output[index] = "original_logits"
    graph.graph.initializer.append(
        onnx.numpy_helper.from_array(np.array(delta, dtype=np.float32), "offset")
    )
    graph.graph.node.append(
        onnx.helper.make_node("Add", ["original_logits", "offset"], ["logits"])
    )
    rewrite_graph(exported, graph)
    report = validate_equivalence(checkpoint, exported, validation_inputs(), 1e-4, 1e-3)
    assert report.status == "HOLD"
    assert all(item.status == "HOLD" for item in report.examples)
    assert "predictions" not in json.dumps(report.summary())
    assert all(len(item.sha256) == 64 for item in report.examples)


def test_prediction_mismatch_holds_even_with_loose_logit_tolerance(
    checkpoint, exported
):
    graph = onnx.load(exported / "model.onnx")
    for node in graph.graph.node:
        for index, name in enumerate(node.output):
            if name == "logits":
                node.output[index] = "original_logits"
    offset = np.zeros(63, dtype=np.float32)
    offset[1] = 10
    graph.graph.initializer.append(onnx.numpy_helper.from_array(offset, "offset"))
    graph.graph.node.append(
        onnx.helper.make_node("Add", ["original_logits", "offset"], ["logits"])
    )
    rewrite_graph(exported, graph)
    report = validate_equivalence(checkpoint, exported, validation_inputs(), 100, 100)
    assert report.status == "HOLD"
    assert any(
        p.reference != p.onnx for item in report.examples for p in item.predictions
    )


@pytest.mark.parametrize("filename", ["config.json", "model.safetensors"])
def test_validator_checks_source_hashes_before_loading(checkpoint, exported, filename):
    path = checkpoint / filename
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ArtifactError, match="source.*SHA256"):
        validate_equivalence(checkpoint, exported, validation_inputs(), 1e-4, 1e-3)


@pytest.mark.parametrize(
    "defect",
    [
        "input_name",
        "output_name",
        "input_shape",
        "output_shape",
        "input_dtype",
        "output_dtype",
    ],
)
def test_validator_checks_real_session_contract(checkpoint, exported, defect):
    graph = onnx.load(exported / "model.onnx")
    inp, output = graph.graph.input[0], graph.graph.output[0]
    if defect == "input_name":
        inp.name = "wrong"
        for node in graph.graph.node:
            for index, name in enumerate(node.input):
                if name == "images":
                    node.input[index] = "wrong"
    elif defect == "output_name":
        output.name = "wrong"
        for node in graph.graph.node:
            for index, name in enumerate(node.output):
                if name == "logits":
                    node.output[index] = "wrong"
    elif defect == "input_shape":
        inp.type.tensor_type.shape.dim[2].dim_value = 31
    elif defect == "input_dtype":
        inp.type.tensor_type.elem_type = onnx.TensorProto.INT32
        for node in graph.graph.node:
            for index, name in enumerate(node.input):
                if name == "images":
                    node.input[index] = "float_images"
        graph.graph.node.insert(
            0,
            onnx.helper.make_node(
                "Cast", ["images"], ["float_images"], to=onnx.TensorProto.FLOAT
            ),
        )
    else:
        for node in graph.graph.node:
            for index, name in enumerate(node.output):
                if name == "logits":
                    node.output[index] = "original_logits"
        if defect == "output_dtype":
            output.type.tensor_type.elem_type = onnx.TensorProto.INT32
            graph.graph.node.append(
                onnx.helper.make_node(
                    "Cast", ["original_logits"], ["logits"], to=onnx.TensorProto.INT32
                )
            )
        else:
            output.type.tensor_type.shape.dim[2].dim_value = 62
            for name, values in (("starts", [0]), ("ends", [62]), ("axes", [2])):
                graph.graph.initializer.append(
                    onnx.numpy_helper.from_array(np.array(values, dtype=np.int64), name)
                )
            graph.graph.node.append(
                onnx.helper.make_node(
                    "Slice", ["original_logits", "starts", "ends", "axes"], ["logits"]
                )
            )
    rewrite_graph(exported, graph)
    expected_boundary = "input" if defect.startswith("input") else "output"
    with pytest.raises(ArtifactError, match=f"session {expected_boundary} contract"):
        validate_equivalence(checkpoint, exported, validation_inputs(), 1e-4, 1e-3)


@pytest.mark.parametrize(
    ("inputs", "atol", "rtol"),
    [([], 1e-4, 1e-3), ([b"invalid"], -1, 1e-3), ([b"invalid"], 1e-4, float("inf"))],
)
def test_validator_cannot_pass_empty_evaluation_or_invalid_tolerances(
    checkpoint, exported, inputs, atol, rtol
):
    with pytest.raises(ValueError):
        validate_equivalence(checkpoint, exported, inputs, atol, rtol)


def test_validator_cli_redacts_stdout_and_restricts_local_report(
    checkpoint, exported, tmp_path, monkeypatch, capsys
):
    image = tmp_path / "input.png"
    image.write_bytes(validation_inputs()[0])
    report_path = tmp_path / "evaluation.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_onnx.py",
            "--source-dir",
            str(checkpoint),
            "--model-dir",
            str(exported),
            "--report",
            str(report_path),
            str(image),
        ],
    )
    assert validate_main() == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "PASS"
    assert "predictions" not in json.dumps(summary)
    assert report_path.stat().st_mode & 0o777 == 0o600
    assert json.loads(report_path.read_text())["examples"][0]["predictions"]
    with pytest.raises(FileExistsError):
        validate_main()
