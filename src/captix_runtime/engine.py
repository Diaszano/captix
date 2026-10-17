"""CPU-only ONNX inference over verified local Captix artifacts."""

from pathlib import Path

import numpy as np
import onnxruntime as ort

from .artifacts import ArtifactError, load_verified_manifest


class EngineError(RuntimeError):
    """An artifact, session, input, or inference result failed validation."""


class OnnxEngine:
    def __init__(self, model_dir: Path, *, intra_op_threads: int = 2):
        self.ready = False
        if type(intra_op_threads) is not int or intra_op_threads < 1:
            raise EngineError("intra_op_threads must be a positive integer")
        try:
            self._manifest = load_verified_manifest(model_dir)
        except ArtifactError as exc:
            raise EngineError(str(exc)) from exc
        self.vocab = self._manifest.vocab
        self.version = self._manifest.model_version
        try:
            if "CPUExecutionProvider" not in ort.get_available_providers():
                raise EngineError("CPUExecutionProvider is unavailable")
            options = ort.SessionOptions()
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.inter_op_num_threads = 1
            options.intra_op_num_threads = intra_op_threads
            self._session = ort.InferenceSession(
                str(Path(model_dir) / "model.onnx"),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            if self._session.get_providers() != ["CPUExecutionProvider"]:
                raise EngineError("Session must use only CPUExecutionProvider")
            self._check_session()
        except EngineError:
            raise
        except Exception as exc:
            raise EngineError("Cannot initialize ONNX CPU session") from exc

    def _check_session(self) -> None:
        inputs, outputs = self._session.get_inputs(), self._session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise EngineError("ONNX session tensor contract mismatch")
        inp, out = inputs[0], outputs[0]
        if (
            inp.name != self._manifest.input_name
            or inp.type != "tensor(float)"
            or not isinstance(inp.shape, list)
            or len(inp.shape) != 4
            or inp.shape[:3] != [1, 3, self._manifest.height]
            or not isinstance(inp.shape[3], str)
            or not inp.shape[3]
        ):
            raise EngineError("ONNX session input contract mismatch")
        if (
            out.name != self._manifest.output_name
            or out.type != "tensor(float)"
            or not isinstance(out.shape, list)
            or len(out.shape) != 3
            or not isinstance(out.shape[0], str)
            or not out.shape[0]
            or out.shape[1:] != [1, len(self.vocab) + 1]
        ):
            raise EngineError("ONNX session output contract mismatch")

    def warmup(self) -> None:
        self.ready = False
        self.infer(np.zeros((1, 3, 32, 32), dtype=np.float32))
        self.ready = True

    def infer(self, images: np.ndarray) -> np.ndarray:
        if (
            not isinstance(images, np.ndarray)
            or images.dtype != np.float32
            or images.ndim != 4
            or images.shape[:3] != (1, 3, self._manifest.height)
            or not self._manifest.width_range[0]
            <= images.shape[3]
            <= self._manifest.width_range[1]
            or not np.isfinite(images).all()
            or (images < self._manifest.input_scale[0]).any()
            or (images > self._manifest.input_scale[1]).any()
        ):
            raise EngineError("Invalid ONNX input tensor")
        try:
            outputs = self._session.run(
                [self._manifest.output_name], {self._manifest.input_name: images}
            )
        except Exception as exc:
            raise EngineError("ONNX inference failed") from exc
        if not isinstance(outputs, list) or len(outputs) != 1:
            raise EngineError("Invalid ONNX output tensors")
        logits = outputs[0]
        if (
            not isinstance(logits, np.ndarray)
            or logits.dtype != np.float32
            or logits.shape != (images.shape[3] // 4, 1, len(self.vocab) + 1)
            or not np.isfinite(logits).all()
        ):
            raise EngineError("Invalid ONNX output tensor")
        return logits
