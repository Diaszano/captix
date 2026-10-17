import hashlib
import json

import pytest

from benchmarks.corpus import CorpusRecord, freeze_corpus, verify_corpus


def _record(path, label="A1", source="synthetic", license="CC0-1.0"):
    return CorpusRecord(str(path), label, source, license, hashlib.sha256(path.read_bytes()).hexdigest())


def test_freeze_and_verify_corpus_is_order_independent(tmp_path):
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    first.write_bytes(b"first fixture")
    second.write_bytes(b"second fixture")
    records = [_record(first), _record(second, label="B2")]
    a, b = tmp_path / "a.json", tmp_path / "b.json"

    freeze_corpus(records, a)
    freeze_corpus(list(reversed(records)), b)

    assert a.read_bytes() == b.read_bytes()
    assert tuple(row.sha256 for row in verify_corpus(a)) == tuple(
        row.sha256 for row in sorted(records, key=lambda row: (row.source, row.sha256))
    )
    assert json.loads(a.read_text())["records"]


def test_corpus_rejects_duplicate_hash_and_empty_metadata(tmp_path):
    path = tmp_path / "fixture.png"
    path.write_bytes(b"fixture")
    record = _record(path)
    out = tmp_path / "manifest.json"

    with pytest.raises(ValueError, match="duplicate"):
        freeze_corpus([record, record], out)
    for field in ("label", "source", "license"):
        broken = record.__class__(**{**record.__dict__, field: ""})
        with pytest.raises(ValueError, match=field):
            freeze_corpus([broken], out)


def test_verify_corpus_rejects_tampered_file(tmp_path):
    path = tmp_path / "fixture.png"
    path.write_bytes(b"original")
    manifest = tmp_path / "manifest.json"
    freeze_corpus([_record(path)], manifest)
    path.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="sha256"):
        verify_corpus(manifest)


def test_relative_paths_resolve_from_manifest(tmp_path, monkeypatch):
    path = tmp_path / "fixture.png"
    path.write_bytes(b"fixture")
    monkeypatch.chdir(tmp_path)
    manifest = tmp_path / "manifests" / "corpus.json"

    freeze_corpus([_record(path.relative_to(tmp_path))], manifest)

    assert verify_corpus(manifest)[0].path == "../fixture.png"
