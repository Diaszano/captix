"""Fail-closed CPU inference using verified local synthetic artifacts only."""

import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import pytest

from captix_runtime.engine import EngineError, OnnxEngine


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
    return tmp_path


@pytest.fixture(autouse=True)
def no_hub(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("runtime must not download a model")

    monkeypatch.setattr("huggingface_hub.snapshot_download", forbidden)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", forbidden)


class FakeSession:
    def __init__(self):
        self.inputs = [
            SimpleNamespace(name="images", type="tensor(float)", shape=[1, 3, 32, "W"])
        ]
        self.outputs = [
            SimpleNamespace(name="logits", type="tensor(float)", shape=["T", 1, 63])
        ]
        self.providers = ["CPUExecutionProvider"]
        self.calls = []
        self.result = None
        self.error = None

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def get_providers(self):
        return self.providers

    def run(self, outputs, inputs):
        assert outputs == ["logits"]
        assert list(inputs) == ["images"]
        images = inputs["images"]
        self.calls.append(images)
        if self.error:
            raise self.error
        if self.result is not None:
            return self.result
        return [np.zeros((images.shape[3] // 4, 1, 63), dtype=np.float32)]


@pytest.fixture
def session(monkeypatch):
    fake = FakeSession()
    creations = []

    def create(path, sess_options, providers):
        assert path.endswith("/model.onnx")
        assert providers == ["CPUExecutionProvider"]
        assert sess_options.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL
        assert sess_options.inter_op_num_threads == 1
        creations.append(sess_options.intra_op_num_threads)
        return fake

    monkeypatch.setattr(
        ort, "get_available_providers", lambda: ["CPUExecutionProvider"]
    )
    monkeypatch.setattr(ort, "InferenceSession", create)
    return fake, creations


def test_warmup_checks_output_and_marks_ready_with_single_cpu_session(
    artifact, session
):
    fake, creations = session
    engine = OnnxEngine(artifact)
    assert engine.ready is False
    engine.warmup()
    assert engine.ready is True
    assert creations == [2]
    assert fake.calls[0].shape == (1, 3, 32, 32)
    assert fake.calls[0].dtype == np.float32
    assert np.isfinite(fake.calls[0]).all()
    assert engine.version == "captix-v1-onnx-fp32"
    assert engine.vocab == tuple(
        "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )
    for width, steps in ((80, 20), (160, 40), (240, 60), (4, 1), (31, 7), (1024, 256)):
        logits = engine.infer(np.zeros((1, 3, 32, width), dtype=np.float32))
        assert logits.shape == (steps, 1, 63)
    assert creations == [2]


def test_thread_count_is_configurable(artifact, session):
    _, creations = session
    OnnxEngine(artifact, intra_op_threads=3)
    assert creations == [3]


@pytest.mark.parametrize("threads", [0, -1, True, 1.5])
def test_invalid_thread_count_is_rejected_before_session(artifact, session, threads):
    with pytest.raises(EngineError, match="threads"):
        OnnxEngine(artifact, intra_op_threads=threads)
    assert session[1] == []


@pytest.mark.parametrize("defect", ["manifest.json", "model.onnx", "hash"])
def test_invalid_artifact_fails_before_constructing_session(artifact, session, defect):
    if defect == "hash":
        (artifact / "model.onnx").write_bytes(b"tampered")
    else:
        (artifact / defect).unlink()
    with pytest.raises(EngineError):
        OnnxEngine(artifact)
    assert session[1] == []


def test_unavailable_cpu_provider_fails_before_session(artifact, session, monkeypatch):
    monkeypatch.setattr(
        ort, "get_available_providers", lambda: ["CUDAExecutionProvider"]
    )
    with pytest.raises(EngineError, match="CPUExecutionProvider"):
        OnnxEngine(artifact)
    assert session[1] == []


@pytest.mark.parametrize(
    "providers",
    [[], ["CUDAExecutionProvider"], ["CPUExecutionProvider", "CUDAExecutionProvider"]],
)
def test_active_provider_must_be_cpu_only(artifact, session, providers):
    session[0].providers = providers
    with pytest.raises(EngineError, match="CPUExecutionProvider"):
        OnnxEngine(artifact)


@pytest.mark.parametrize(
    ("side", "field", "value"),
    [
        ("inputs", "name", "wrong"),
        ("outputs", "name", "wrong"),
        ("inputs", "type", "tensor(double)"),
        ("outputs", "type", "tensor(double)"),
        ("inputs", "shape", [1, 3, 32, 160]),
        ("inputs", "shape", [2, 3, 32, "W"]),
        ("inputs", "shape", [1, 1, 32, "W"]),
        ("inputs", "shape", [1, 3, 31, "W"]),
        ("inputs", "shape", [1, 3, 32, None]),
        ("outputs", "shape", [40, 1, 63]),
        ("outputs", "shape", [None, 1, 63]),
        ("outputs", "shape", ["T", 2, 63]),
        ("outputs", "shape", ["T", 1, 62]),
    ],
)
def test_session_metadata_must_match_dynamic_contract(
    artifact, session, side, field, value
):
    setattr(getattr(session[0], side)[0], field, value)
    with pytest.raises(EngineError, match="contract"):
        OnnxEngine(artifact)


@pytest.mark.parametrize("side", ["inputs", "outputs"])
def test_extra_tensors_are_rejected(artifact, session, side):
    getattr(session[0], side).append(getattr(session[0], side)[0])
    with pytest.raises(EngineError, match="contract"):
        OnnxEngine(artifact)


@pytest.mark.parametrize(
    "images",
    [
        None,
        np.zeros((3, 32, 80), dtype=np.float32),
        np.zeros((2, 3, 32, 80), dtype=np.float32),
        np.zeros((1, 1, 32, 80), dtype=np.float32),
        np.zeros((1, 3, 31, 80), dtype=np.float32),
        np.zeros((1, 3, 32, 80), dtype=np.float64),
        np.zeros((1, 3, 32, 3), dtype=np.float32),
        np.zeros((1, 3, 32, 1025), dtype=np.float32),
        np.full((1, 3, 32, 80), np.nan, dtype=np.float32),
        np.full((1, 3, 32, 80), np.inf, dtype=np.float32),
        np.full((1, 3, 32, 80), -0.01, dtype=np.float32),
        np.full((1, 3, 32, 80), 1.01, dtype=np.float32),
    ],
)
def test_invalid_input_never_reaches_session(artifact, session, images):
    engine = OnnxEngine(artifact)
    with pytest.raises(EngineError, match="input"):
        engine.infer(images)
    assert session[0].calls == []


@pytest.mark.parametrize(
    "result",
    [
        [],
        [None],
        [np.zeros((8, 1, 62), dtype=np.float32)],
        [np.zeros((7, 1, 63), dtype=np.float32)],
        [np.zeros((8, 2, 63), dtype=np.float32)],
        [np.zeros((8, 1, 63), dtype=np.float64)],
        [np.full((8, 1, 63), np.nan, dtype=np.float32)],
        [np.full((8, 1, 63), np.inf, dtype=np.float32)],
    ],
)
def test_invalid_warmup_output_leaves_engine_unready(artifact, session, result):
    session[0].result = result
    engine = OnnxEngine(artifact)
    with pytest.raises(EngineError, match="output"):
        engine.warmup()
    assert engine.ready is False


def test_inference_rejects_wrong_dynamic_output_length(artifact, session):
    engine = OnnxEngine(artifact)
    engine.warmup()
    session[0].result = [np.zeros((8, 1, 63), dtype=np.float32)]
    with pytest.raises(EngineError, match="output"):
        engine.infer(np.zeros((1, 3, 32, 80), dtype=np.float32))


def test_failed_repeated_warmup_resets_readiness(artifact, session):
    engine = OnnxEngine(artifact)
    engine.warmup()
    session[0].error = RuntimeError("session failed")
    with pytest.raises(EngineError, match="inference"):
        engine.warmup()
    assert engine.ready is False


def test_session_creation_failure_is_typed(artifact, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("invalid ONNX")

    monkeypatch.setattr(ort, "InferenceSession", fail)
    with pytest.raises(EngineError, match="session"):
        OnnxEngine(artifact)


def test_synthetic_dynamic_graph_runs_without_torch_or_hub_imports(artifact):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv", ["images", "weights"], ["features"], strides=[32, 4]
            ),
            helper.make_node("Squeeze", ["features", "squeeze_axes"], ["squeezed"]),
            helper.make_node("Transpose", ["squeezed"], ["transposed"], perm=[1, 0]),
            helper.make_node("Unsqueeze", ["transposed", "batch_axis"], ["logits"]),
        ],
        "synthetic-dynamic-width",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 32, "W"])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["T", 1, 63])],
        [
            numpy_helper.from_array(
                np.zeros((63, 3, 32, 4), dtype=np.float32), "weights"
            ),
            numpy_helper.from_array(np.array([0, 2], dtype=np.int64), "squeeze_axes"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), "batch_axis"),
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
    )
    onnx.checker.check_model(model)
    onnx.save(model, artifact / "model.onnx")
    manifest = json.loads((artifact / "manifest.json").read_text())
    manifest["model_sha256"] = hashlib.sha256(
        (artifact / "model.onnx").read_bytes()
    ).hexdigest()
    (artifact / "manifest.json").write_text(json.dumps(manifest))
    script = """
import importlib.abc
import sys
from pathlib import Path

class BlockTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'huggingface_hub'}:
            raise AssertionError(f'runtime imported {fullname}')

sys.meta_path.insert(0, BlockTrainingImports())
import numpy as np
from captix_runtime.engine import OnnxEngine
engine = OnnxEngine(Path(sys.argv[1]))
assert not engine.ready
engine.warmup()
assert engine.ready
for width, steps in ((4, 1), (31, 7), (80, 20), (160, 40), (240, 60), (1024, 256)):
    logits = engine.infer(np.zeros((1, 3, 32, width), dtype=np.float32))
    assert logits.shape == (steps, 1, 63)
    assert np.isfinite(logits).all()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(artifact)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
