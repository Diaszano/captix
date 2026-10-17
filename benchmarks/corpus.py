"""Deterministic manifests for explicitly authorized benchmark images."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class CorpusRecord:
    path: str
    label: str
    source: str
    license: str
    sha256: str
    mask: str | None = None
    length: int | None = None
    case_sensitive: bool = True


def _validated_record(record: CorpusRecord, base: Path | None = None) -> CorpusRecord:
    for field in ("label", "source", "license"):
        if not getattr(record, field).strip():
            raise ValueError(f"{field} must not be empty")
    if record.mask is not None and not isinstance(record.mask, str):
        raise ValueError("mask must be a string or null")
    if record.length is not None and (
        type(record.length) is not int or not 1 <= record.length <= 16
    ):
        raise ValueError("length must be an integer in 1..16 or null")
    if type(record.case_sensitive) is not bool:
        raise ValueError("case_sensitive must be boolean")
    path = Path(record.path)
    if base is not None and not path.is_absolute():
        path = base / path
    if not path.is_file():
        raise ValueError(f"Corpus file does not exist: {record.path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if record.sha256 and record.sha256 != digest:
        raise ValueError(f"sha256 mismatch for {record.path}")
    return CorpusRecord(**{**asdict(record), "sha256": digest})


def freeze_corpus(
    records: Sequence[CorpusRecord],
    out: Path,
    *,
    authorized: bool = False,
    split: str = "unconfirmed",
) -> dict:
    """Write a sorted manifest after checking each referenced file; copy none."""
    out = Path(out)
    if type(authorized) is not bool or split not in {
        "unconfirmed",
        "train",
        "calibration",
        "test",
    }:
        raise ValueError("Expected boolean authorized and a supported split")
    out.parent.mkdir(parents=True, exist_ok=True)
    base = out.parent.resolve()
    frozen = []
    for record in records:
        path = Path(record.path)
        if not path.is_absolute():
            path = Path(os.path.relpath(path.resolve(), base))
            record = CorpusRecord(**{**asdict(record), "path": str(path)})
        frozen.append(_validated_record(record, base))
    hashes = [record.sha256 for record in frozen]
    if len(set(hashes)) != len(hashes):
        raise ValueError("duplicate sha256 in corpus")
    frozen.sort(key=lambda record: (record.source, record.sha256))
    manifest = {
        "authorized": authorized,
        "split": split,
        "records": [asdict(record) for record in frozen],
    }
    out.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def verify_corpus(manifest: Path) -> tuple[CorpusRecord, ...]:
    """Load a manifest and verify ordering, unique hashes, and file contents."""
    manifest = Path(manifest)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    records = tuple(CorpusRecord(**row) for row in data["records"])
    if any(
        len(record.sha256) != 64
        or any(c not in "0123456789abcdef" for c in record.sha256)
        for record in records
    ):
        raise ValueError("Frozen corpus requires lowercase sha256 hashes")
    actual = tuple(_validated_record(record, manifest.parent) for record in records)
    if len({record.sha256 for record in actual}) != len(actual):
        raise ValueError("duplicate sha256 in corpus")
    if actual != tuple(
        sorted(actual, key=lambda record: (record.source, record.sha256))
    ):
        raise ValueError("corpus records are not sorted by source and sha256")
    return actual
