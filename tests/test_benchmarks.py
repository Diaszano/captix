import hashlib
import json
from dataclasses import asdict, replace

import pytest

from benchmarks.bench_runtime import (
    BenchmarkReport,
    _measure,
    _pss,
    run_benchmark,
    summarize_runs,
)
from benchmarks.compare_results import (
    LabeledResult,
    evaluate_promotion,
    paired_accuracy,
    write_comparison,
)
from benchmarks.corpus import CorpusRecord, freeze_corpus, verify_corpus


def _record(path, label="A1", source="synthetic", license="CC0-1.0"):
    return CorpusRecord(
        str(path), label, source, license, hashlib.sha256(path.read_bytes()).hexdigest()
    )


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


def test_verify_rejects_missing_frozen_hash(tmp_path):
    path = tmp_path / "fixture.png"
    path.write_bytes(b"fixture")
    manifest = tmp_path / "manifest.json"
    freeze_corpus([_record(path)], manifest)
    data = json.loads(manifest.read_text())
    data["records"][0]["sha256"] = ""
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="sha256"):
        verify_corpus(manifest)


def _runs(p95=100.0, rss=1000):
    return [
        {
            "latency_ms": {
                stage: {"p50": p95 / 2, "p95": p95, "p99": p95 * 2}
                for stage in ("preprocess", "inference", "decode", "total")
            },
            "throughput": 10.0,
            "rss_peak_bytes": rss,
            "pss_peak_bytes": None,
            "cold_start_ms": 200.0,
            "docker_bytes": 3000,
        }
        for _ in range(3)
    ]


def _evidence(n=1000):
    return [
        LabeledResult(f"{i:064x}", "source-a" if i < 100 else "source-b", True, True)
        for i in range(n)
    ]


def _report(variant="pytorch-original", *, p95=100.0, rss=1000, labeled=None):
    labeled = _evidence() if labeled is None else labeled
    candidate = variant == "onnx-fp32"
    return BenchmarkReport(
        variant=variant,
        status="measured",
        settings={
            "warmup": 100,
            "requests": 1000,
            "repetitions": 3,
            "seed": 0,
            "batch_size": 1,
            "official": True,
        },
        environment={
            "host_id": "same-host",
            "cpu": "cpu",
            "kernel": "linux",
            "ram_bytes": 4096,
            "affinity": [0, 1],
            "cpu_quota_cores": 2.0,
            "memory_limit_bytes": 1073741824,
            "isolated": True,
            "commit": "a" * 40,
            "threads": 2,
            "dirty": False,
            "versions": {"python": "3.11"},
        },
        corpus={
            "sha256": "b" * 64,
            "authorized": True,
            "split": "test",
            "samples": len(labeled),
        },
        artifact={
            "source_sha256": {"model.safetensors": "c" * 64, "config.json": "d" * 64}
        },
        runs=_runs(p95, rss),
        summary=summarize_runs(_runs(p95, rss)),
        accuracy=[
            {
                "sha256": row.sha256,
                "source": row.source,
                "correct": row.candidate_correct if candidate else row.baseline_correct,
            }
            for row in labeled
        ],
        checks={
            name: True
            for name in (
                "golden",
                "api_smoke",
                "container_smoke",
                "offline",
                "payload",
                "auth",
                "saturation",
                "shutdown",
            )
        },
    )


def test_summary_preserves_missing_values_and_repetition_dispersion():
    runs = _runs()
    runs[0]["latency_ms"]["total"]["p95"] = 80
    runs[2]["latency_ms"]["total"]["p95"] = 120
    summary = summarize_runs(runs)
    assert summary["latency_ms"]["total"]["p95"] == {
        "median": 100.0,
        "min": 80.0,
        "max": 120.0,
        "stdev": 20.0,
    }
    assert summary["pss_peak_bytes"]["median"] is None
    assert summary["throughput"]["median"] == 10


def test_runner_returns_unavailable_without_local_model_and_serializes_null(
    tmp_path, monkeypatch
):
    path = tmp_path / "fixture.png"
    path.write_bytes(b"fixture")
    manifest = tmp_path / "manifest.json"
    freeze_corpus([_record(path)], manifest)
    monkeypatch.setenv("CAPTIX_MODEL_DIR", str(tmp_path / "missing"))
    report = run_benchmark("onnx-fp32", manifest)
    assert report.status == "unavailable"
    assert report.summary["rss_peak_bytes"]["median"] is None
    assert report.settings["official"] is True
    assert not report.corpus["authorized"]
    assert '"median": null' in json.dumps(asdict(report))


@pytest.mark.parametrize(
    "field,value", [("warmup", 0), ("requests", 0), ("repetitions", 0)]
)
def test_runner_rejects_invalid_counts_before_work(tmp_path, field, value):
    with pytest.raises(ValueError):
        run_benchmark("onnx-fp32", tmp_path / "absent.json", **{field: value})


def test_promotion_accepts_latency_gain_with_memory_within_five_percent():
    decision = evaluate_promotion(
        _report(), _report("onnx-fp32", p95=89, rss=1050), _evidence()
    )
    assert decision.status == "PROMOTE"
    assert decision.reasons == []


@pytest.mark.parametrize("n", [0, 999])
def test_promotion_holds_insufficient_labeled_evidence(n):
    labeled = _evidence(n)
    decision = evaluate_promotion(
        _report(labeled=labeled), _report("onnx-fp32", p95=89, labeled=labeled), labeled
    )
    assert decision.status == "HOLD"
    assert any("1000" in reason for reason in decision.reasons)


@pytest.mark.parametrize("losses,source_loss", [(3, False), (2, True)])
def test_accuracy_policy_holds_overall_or_per_source_regression(losses, source_loss):
    rows = _evidence()
    for i in range(losses):
        rows[i] = replace(rows[i], candidate_correct=False)
    if source_loss:  # Offset source-a's -2 p.p.; the overall difference is zero.
        for i in (100, 101):
            rows[i] = replace(rows[i], baseline_correct=False)
    decision = evaluate_promotion(
        _report(labeled=rows), _report("onnx-fp32", p95=89, labeled=rows), rows
    )
    assert decision.status == "HOLD"
    assert any(
        ("source-a" if source_loss else "overall") in r for r in decision.reasons
    )


@pytest.mark.parametrize(
    "change",
    [
        "golden",
        "operational",
        "host",
        "quotas",
        "seed",
        "unauthorized",
        "split",
        "unmeasured",
        "experimental",
        "artifact",
        "duplicate",
        "unpaired",
        "rss",
        "no_gain",
        "docker",
        "dirty",
    ],
)
def test_promotion_fails_closed(change):
    baseline, candidate, rows = _report(), _report("onnx-fp32", p95=89), _evidence()
    if change == "golden":
        candidate.checks["golden"] = False
    if change == "operational":
        candidate.checks["offline"] = None
    if change == "host":
        candidate.environment["host_id"] = "other-host"
    if change == "quotas":
        candidate.environment["cpu_quota_cores"] = 4
    if change == "seed":
        candidate.settings["seed"] = 1
    if change == "unauthorized":
        candidate.corpus["authorized"] = False
    if change == "split":
        candidate.corpus["split"] = "train"
    if change == "unmeasured":
        candidate.status = "unavailable"
    if change == "experimental":
        candidate.settings["official"] = False
    if change == "artifact":
        candidate.artifact["source_sha256"] = {}
    if change == "duplicate":
        rows[-1] = rows[0]
    if change == "unpaired":
        rows[0] = replace(rows[0], baseline_correct=False)
    if change == "rss":
        candidate.summary["rss_peak_bytes"]["median"] = 1051
    if change == "no_gain":
        candidate.summary["latency_ms"]["total"]["p95"]["median"] = 91
    if change == "docker":
        candidate.summary["docker_bytes"]["median"] = None
    if change == "dirty":
        candidate.environment["dirty"] = True
    assert evaluate_promotion(baseline, candidate, rows).status == "HOLD"


def test_paired_bootstrap_is_seeded_and_keeps_policy_separate():
    rows = _evidence()
    rows[0] = replace(rows[0], candidate_correct=False)
    result = paired_accuracy(rows, seed=17)
    assert result == paired_accuracy(rows, seed=17)
    assert result["delta_pp"] == pytest.approx(-0.1)
    assert result["ci95_pp"][0] <= result["delta_pp"] <= result["ci95_pp"][1]
    assert result["sources"]["source-a"]["n"] == 100
    assert paired_accuracy([], seed=17)["delta_pp"] is None


def test_comparison_outputs_only_aggregate_metrics(tmp_path):
    baseline, candidate = _report(), _report("onnx-fp32", p95=89)
    candidate.status = "unavailable"
    write_comparison(baseline, candidate, _evidence(), tmp_path / "comparison")
    data = json.loads((tmp_path / "comparison.json").read_text())
    assert data["decision"]["status"] == "HOLD"
    assert "accuracy" not in data["candidate"]
    assert (tmp_path / "comparison.csv").read_text().startswith("variant,metric,")
    assert "HOLD" in (tmp_path / "comparison.md").read_text()
    assert "source-a" in (tmp_path / "comparison.md").read_text()
    import csv

    with (tmp_path / "comparison.csv").open() as stream:
        accuracy_rows = [
            row for row in csv.DictReader(stream) if row["metric"] == "exact_match"
        ]
    assert accuracy_rows[0]["n"] == "1000"
    assert accuracy_rows[1]["source"] == "source-a"


def test_measure_percentiles_and_accuracy_cover_all_unique_samples(monkeypatch):
    from types import SimpleNamespace

    import benchmarks.bench_runtime as benchmark
    from captix_runtime.pipeline import Prediction

    records = [
        CorpusRecord("unused", label, "synthetic", "CC0", f"{i:064x}")
        for i, label in enumerate(("A", "B", "C"))
    ]
    calls = []

    def predict(blob, record):
        calls.append(record.sha256)
        # Three measured requests after one warmup, with hand-checkable percentiles.
        elapsed = float(len(calls))
        return Prediction(
            "A", "greedy", elapsed, {"preprocess": 1.0, "inference": 2.0, "decode": 3.0}
        )

    clock = iter([0, 1, 3, 4, 8, 9, 15, 20])
    monkeypatch.setattr(
        benchmark, "time", SimpleNamespace(perf_counter=lambda: next(clock))
    )

    run, accuracy = _measure(
        predict, records, [b"a", b"b", b"c"], warmup=1, requests=3, seed=0
    )
    assert run["latency_ms"]["total"] == {"p50": 4000.0, "p95": 5800.0, "p99": 5960.0}
    assert run["latency_ms"]["decode"] == {"p50": 3.0, "p95": 3.0, "p99": 3.0}
    assert run["throughput"] > 0
    assert [row["correct"] for row in accuracy] == [True, False, False]
    assert len(calls) == 7
    assert _pss(999999999) is None


def test_runner_real_subprocesses_with_synthetic_local_checkpoint(
    tmp_path, monkeypatch
):
    import os
    import subprocess
    import sys

    from PIL import Image

    from captix.hub import save_pretrained
    from captix.model import CRNN

    path = tmp_path / "fixture.png"
    Image.new("RGB", (32, 32), "white").save(path)
    manifest = tmp_path / "manifest.json"
    freeze_corpus([_record(path)], manifest)
    checkpoint = tmp_path / "checkpoint"
    save_pretrained(CRNN(vocab=["A", "1"], rnn_hidden=4, rnn_layers=1), checkpoint)
    monkeypatch.setenv("CAPTIX_BENCH_CHECKPOINT_DIR", str(checkpoint))
    monkeypatch.delenv("CAPTIX_BENCH_EVIDENCE", raising=False)
    monkeypatch.delenv("CAPTIX_BENCH_DOCKER_IMAGE", raising=False)
    report = run_benchmark(
        "pytorch-compatible", manifest, warmup=1, requests=2, repetitions=3
    )
    assert report.status == "measured", report.errors
    assert len(report.runs) == 3
    assert report.summary["cold_start_ms"]["median"] > 0
    assert report.summary["rss_peak_bytes"]["median"] > 0
    assert report.summary["docker_bytes"]["median"] is None
    assert not report.settings["official"]
    assert len(report.accuracy) == 1
    for variant in ("pytorch-original", "pytorch-compatible"):
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from benchmarks.bench_runtime import _predictor; "
                    f"_predictor({variant!r}); import sys; assert 'onnxruntime' not in sys.modules"
                ),
            ],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            check=False,
        )
        assert probe.returncode == 0, probe.stderr


@pytest.mark.parametrize(
    "field,value", [("length", 0), ("case_sensitive", "false"), ("mask", 42)]
)
def test_corpus_validates_decoder_constraints(tmp_path, field, value):
    path = tmp_path / "image.png"
    path.write_bytes(b"fixture")
    with pytest.raises(ValueError, match=field):
        freeze_corpus(
            [replace(_record(path), **{field: value})], tmp_path / "manifest.json"
        )


def test_corpus_preserves_decoder_constraints_and_authorization(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"fixture")
    manifest = tmp_path / "nested" / "manifest.json"
    freeze_corpus(
        [replace(_record(path), mask="[A-Z0-9]", length=2, case_sensitive=False)],
        manifest,
        authorized=True,
        split="test",
    )
    (record,) = verify_corpus(manifest)
    assert (record.mask, record.length, record.case_sensitive) == ("[A-Z0-9]", 2, False)
    assert json.loads(manifest.read_text())["authorized"] is True


def test_cgroup_limits_use_process_scope_and_tightest_ancestor(tmp_path):
    from benchmarks.bench_runtime import _cgroup_limits

    group = tmp_path / "user.slice" / "bench.scope"
    group.mkdir(parents=True)
    membership = tmp_path / "membership"
    membership.write_text("0::/user.slice/bench.scope\n")
    (tmp_path / "cpu.max").write_text("max 100000")
    (tmp_path / "memory.max").write_text("max")
    (group.parent / "cpu.max").write_text("200000 100000")
    (group.parent / "memory.max").write_text("1073741824")
    (group / "cpu.max").write_text("400000 100000")
    (group / "memory.max").write_text("2147483648")
    assert _cgroup_limits(tmp_path, membership) == (2.0, 1073741824)


def test_runner_rejects_negative_seed(tmp_path):
    with pytest.raises(ValueError, match="seed"):
        run_benchmark("onnx-fp32", tmp_path / "absent.json", seed=-1)


def test_memory_value_gate_and_accuracy_boundary_are_inclusive():
    rows = _evidence()
    # -1 p.p. per source, -0.2 p.p. overall, with no significance override.
    rows[0] = replace(rows[0], candidate_correct=False)
    rows[100] = replace(rows[100], candidate_correct=False)
    decision = evaluate_promotion(
        _report(labeled=rows),
        _report("onnx-fp32", p95=105, rss=800, labeled=rows),
        rows,
    )
    assert decision.status == "PROMOTE"
