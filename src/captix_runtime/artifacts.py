"""Verify the fixed captix-v1 artifact contract without importing Torch or ONNX."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, fields
from pathlib import Path


class ArtifactError(ValueError):
    """Missing, corrupt, or incompatible model artifacts."""


@dataclass(frozen=True)
class ModelManifest:
    schema_version: int
    model_version: str
    exporter: str
    opset: int
    model_sha256: str
    source_commit: str
    revision_sha: str
    source_sha256: dict[str, str]
    versions: dict[str, str]
    height: int
    input_name: str
    output_name: str
    dtype: str
    vocab: tuple[str, ...]
    blank_index: int
    width_range: tuple[int, int]
    input_scale: tuple[int, int]
    input_layout: str
    output_layout: str


def sha256_file(path: Path) -> str:
    """Hash artifacts in bounded chunks, including large weight files."""
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ArtifactError(f"Cannot read artifact {path.name}") from exc
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactError("Duplicate manifest field")
        result[key] = value
    return result


def load_verified_manifest(model_dir: Path) -> ModelManifest:
    """Reject unsupported metadata and verify model.onnx before session creation."""
    directory = Path(model_dir)
    try:
        data = json.loads(
            (directory / "manifest.json").read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError("Cannot read manifest.json") from exc
    if not isinstance(data, dict):
        raise ArtifactError("Manifest must be a JSON object")
    expected_fields = {field.name for field in fields(ModelManifest)}
    if set(data) != expected_fields:
        names = sorted(set(data) ^ expected_fields)
        raise ArtifactError(f"Unsupported manifest fields: {', '.join(names)}")

    fixed = {
        "schema_version": 1,
        "model_version": "captix-v1-onnx-fp32",
        "exporter": "torchscript",
        "opset": 18,
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
    for name, expected in fixed.items():
        value = data[name]
        if type(value) is not type(expected) or value != expected:
            raise ArtifactError(f"Unsupported {name}")
        if isinstance(expected, list) and any(
            type(item) is not type(wanted) for item, wanted in zip(value, expected)
        ):
            raise ArtifactError(f"Unsupported {name}")
    for name, size in (
        ("model_sha256", 64),
        ("source_commit", 40),
        ("revision_sha", 40),
    ):
        if (
            not isinstance(data[name], str)
            or re.fullmatch(rf"[0-9a-f]{{{size}}}", data[name]) is None
        ):
            raise ArtifactError(f"Invalid {name}")
    source = data["source_sha256"]
    if not isinstance(source, dict) or set(source) != {
        "model.safetensors",
        "config.json",
    }:
        raise ArtifactError("Invalid source_sha256")
    if any(
        not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{64}", sha) is None
        for sha in source.values()
    ):
        raise ArtifactError("Invalid source_sha256")
    versions = data["versions"]
    if (
        not isinstance(versions, dict)
        or set(versions) != {"torch", "onnx", "onnxscript", "onnxruntime"}
        or any(
            not isinstance(value, str) or not value.strip()
            for value in versions.values()
        )
    ):
        raise ArtifactError("Invalid versions")
    if sha256_file(directory / "model.onnx") != data["model_sha256"]:
        raise ArtifactError("model.onnx SHA256 mismatch")
    for name in ("vocab", "width_range", "input_scale"):
        data[name] = tuple(data[name])
    return ModelManifest(**data)
