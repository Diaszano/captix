"""Compare local PyTorch and ONNX artifacts; print a text-free evaluation summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from importlib.metadata import version
from pathlib import Path

import numpy as np

from captix_runtime.artifacts import ArtifactError, load_verified_manifest, sha256_file
from captix_runtime.ctc import decode_fixed_length, decode_greedy
from captix_runtime.preprocessing import preprocess_image


@dataclass(frozen=True)
class PredictionComparison:
    mask: str | None
    length: int | None
    case_sensitive: bool
    reference: str = field(repr=False)
    onnx: str = field(repr=False)


@dataclass(frozen=True)
class ExampleComparison:
    example_id: str
    sha256: str
    width: int
    shape: tuple[int, ...]
    max_abs_diff: float | None
    max_rel_diff: float | None
    status: str
    reason: str | None
    predictions: tuple[PredictionComparison, ...] = field(repr=False)


@dataclass(frozen=True)
class ValidationReport:
    status: str
    atol: float
    rtol: float
    max_abs_diff: float | None
    max_rel_diff: float | None
    model_sha256: str
    source_commit: str
    revision_sha: str
    source_sha256: dict[str, str]
    versions: dict[str, dict[str, str]]
    examples: tuple[ExampleComparison, ...]

    def summary(self) -> dict:
        """Operational output deliberately excludes every predicted string."""
        result = asdict(self)
        for example in result["examples"]:
            example.pop("predictions")
        return result


def _check_session(session, manifest) -> None:
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ArtifactError("ONNX session must have one input and one output")
    inp, out = inputs[0], outputs[0]
    if (
        inp.name != manifest.input_name
        or inp.type != "tensor(float)"
        or len(inp.shape) != 4
        or inp.shape[:3] != [1, 3, manifest.height]
        or not isinstance(inp.shape[3], str)
        or not inp.shape[3]
    ):
        raise ArtifactError("ONNX session input contract mismatch")
    if (
        out.name != manifest.output_name
        or out.type != "tensor(float)"
        or len(out.shape) != 3
        or not isinstance(out.shape[0], str)
        or not out.shape[0]
        or out.shape[1:] != [1, len(manifest.vocab) + 1]
    ):
        raise ArtifactError("ONNX session output contract mismatch")


def _compare_predictions(reference, actual, vocab) -> tuple[PredictionComparison, ...]:
    import torch
    from torch.nn import functional

    from captix.dataset import decode_indices
    from captix.decrypt import _apply_mask, _ctc_decode_fixed_length, _resolve_mask

    predictions = []
    for mask in (None, "[0-9]", "[a-z]", "[A-Z]"):
        for case_sensitive in (True, False):
            masked = _apply_mask(
                torch.from_numpy(reference), _resolve_mask(mask, vocab, case_sensitive)
            )
            for length in (None, min(4, reference.shape[0])):
                if length is None:
                    path = masked.argmax(dim=-1).tolist()
                    result = decode_greedy(
                        actual, vocab, mask, case_sensitive=case_sensitive
                    )
                else:
                    log_probs = functional.log_softmax(masked, dim=-1).numpy()
                    path = _ctc_decode_fixed_length(log_probs, length)
                    result = decode_fixed_length(
                        actual, vocab, length, mask, case_sensitive=case_sensitive
                    )
                expected = decode_indices(path, vocab, collapse=True)
                if not case_sensitive:
                    expected = expected.lower()
                predictions.append(
                    PredictionComparison(mask, length, case_sensitive, expected, result)
                )
    return tuple(predictions)


def validate_equivalence(
    source_dir: Path,
    model_dir: Path,
    inputs: Sequence[bytes],
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> ValidationReport:
    """Evaluate supplied images only; tolerances are initial comparison limits.

    Predictions stay in the returned local evaluation object. A PASS says nothing
    about recognition accuracy, corpus coverage, or untested model revisions.
    """
    if not inputs:
        raise ValueError("Evaluation requires at least one image")
    if any(not math.isfinite(value) or value < 0 for value in (atol, rtol)):
        raise ValueError("Tolerances must be finite and nonnegative")
    source_dir, model_dir = Path(source_dir), Path(model_dir)
    manifest = load_verified_manifest(model_dir)
    for name, expected in manifest.source_sha256.items():
        if sha256_file(source_dir / name) != expected:
            raise ArtifactError(f"source {name} SHA256 mismatch")

    import onnxruntime as ort
    import torch

    from captix.hub import load_local

    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    try:
        session = ort.InferenceSession(
            str(model_dir / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )
    except Exception as exc:
        raise ArtifactError("Cannot load ONNX session") from exc
    _check_session(session, manifest)
    reference = load_local(source_dir).cpu().float().eval()
    examples = []
    for index, data in enumerate(inputs):
        images = preprocess_image(data).array
        width = images.shape[3]
        with torch.inference_mode():
            expected = reference(torch.from_numpy(images)).numpy()
        actual = session.run([manifest.output_name], {manifest.input_name: images})[0]
        shape = (width // 4, 1, len(manifest.vocab) + 1)
        if (
            expected.shape != shape
            or actual.shape != shape
            or expected.dtype != np.float32
            or actual.dtype != np.float32
        ):
            raise ArtifactError("Inference output shape or dtype contract mismatch")
        predictions = ()
        absolute = relative = None
        reason = "non_finite_logits"
        passed = False
        if np.isfinite(expected).all() and np.isfinite(actual).all():
            difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
            absolute = float(difference.max())
            relative = float(
                (
                    difference / np.maximum(np.abs(expected), np.finfo(np.float32).tiny)
                ).max()
            )
            predictions = _compare_predictions(
                expected[:, 0], actual[:, 0], manifest.vocab
            )
            passed = bool(np.allclose(actual, expected, atol=atol, rtol=rtol)) and all(
                p.reference == p.onnx for p in predictions
            )
            reason = None if passed else "logit_or_prediction_mismatch"
        examples.append(
            ExampleComparison(
                str(index),
                hashlib.sha256(data).hexdigest(),
                width,
                shape,
                absolute,
                relative,
                "PASS" if passed else "HOLD",
                reason,
                predictions,
            )
        )
    finite = all(example.max_abs_diff is not None for example in examples)
    return ValidationReport(
        status="PASS"
        if all(example.status == "PASS" for example in examples)
        else "HOLD",
        atol=atol,
        rtol=rtol,
        max_abs_diff=max(example.max_abs_diff for example in examples)
        if finite
        else None,
        max_rel_diff=max(example.max_rel_diff for example in examples)
        if finite
        else None,
        model_sha256=manifest.model_sha256,
        source_commit=manifest.source_commit,
        revision_sha=manifest.revision_sha,
        source_sha256=manifest.source_sha256,
        versions={
            "export": manifest.versions,
            "validation": {name: version(name) for name in manifest.versions},
        },
        examples=tuple(examples),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", required=True, type=Path, help="Local safetensors checkpoint"
    )
    parser.add_argument(
        "--model-dir", required=True, type=Path, help="Verified ONNX artifact directory"
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Initial absolute tolerance, to validate on your corpus",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Initial relative tolerance, to validate on your corpus",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="New private JSON file WITH predictions; controlled local evaluation only",
    )
    parser.add_argument("images", nargs="+", type=Path)
    args = parser.parse_args()
    report = validate_equivalence(
        args.source_dir,
        args.model_dir,
        [path.read_bytes() for path in args.images],
        args.atol,
        args.rtol,
    )
    if args.report is not None:
        descriptor = os.open(args.report, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(asdict(report), stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(report.summary(), allow_nan=False))
    return 0 if report.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
