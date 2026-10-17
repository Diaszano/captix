"""Offline, B=1 OCR measurements in a fresh subprocess for each repetition."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Support both `python benchmarks/bench_runtime.py` and module imports.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.corpus import verify_corpus

VARIANTS = ("pytorch-original", "pytorch-compatible", "onnx-fp32")
STAGES = ("preprocess", "inference", "decode", "total")
CHECKS = (
    "golden",
    "api_smoke",
    "container_smoke",
    "offline",
    "payload",
    "auth",
    "saturation",
    "shutdown",
)
SCALARS = (
    "throughput",
    "rss_peak_bytes",
    "pss_peak_bytes",
    "cold_start_ms",
    "docker_bytes",
)


@dataclass
class BenchmarkReport:
    variant: str
    status: str
    settings: dict
    environment: dict
    corpus: dict
    artifact: dict = field(default_factory=dict)
    runs: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    accuracy: list[dict] = field(default_factory=list)
    checks: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def _dispersion(values):
    if not values or any(v is None for v in values):
        return dict.fromkeys(("median", "min", "max", "stdev"))
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def summarize_runs(runs: list[dict]) -> dict:
    """Median, range and sample standard deviation across independent runs."""
    result = {name: _dispersion([run.get(name) for run in runs]) for name in SCALARS}
    result["latency_ms"] = {
        stage: {
            p: _dispersion([run["latency_ms"][stage][p] for run in runs])
            for p in ("p50", "p95", "p99")
        }
        for stage in STAGES
    }
    return result


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _cgroup_limits(root=Path("/sys/fs/cgroup"), membership=Path("/proc/self/cgroup")):
    """Resolve v2 process quotas, including tighter inherited limits."""
    content = _read(str(membership)) or ""
    relative = next(
        (line[3:] for line in content.splitlines() if line.startswith("0::")), None
    )
    if relative is None:
        return None, None  # v1/unknown limits cannot establish the reference profile.
    group = (root / relative.lstrip("/")).resolve()
    root = root.resolve()
    if not group.is_relative_to(root) or not group.is_dir():
        group = (
            root  # Container cgroup namespace can expose its scope at the mount root.
        )
    cpu_limits, memory_limits = [], []
    while True:
        cpu_max = _read(str(group / "cpu.max"))
        memory_max = _read(str(group / "memory.max"))
        if cpu_max and cpu_max.split()[0] != "max":
            quota, period = cpu_max.split()
            cpu_limits.append(int(quota) / int(period))
        if memory_max and memory_max != "max":
            memory_limits.append(int(memory_max))
        if group == root:
            break
        group = group.parent
    return min(cpu_limits, default=None), min(memory_limits, default=None)


def _environment() -> dict:
    import psutil

    cpu_limit, memory_limit = _cgroup_limits()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parents[1],
        ).strip()
        dirty = (
            subprocess.run(
                ["git", "diff", "--quiet", "HEAD"],
                stderr=subprocess.DEVNULL,
                check=False,
                cwd=Path(__file__).resolve().parents[1],
            ).returncode
            != 0
        )
    except (OSError, subprocess.CalledProcessError):
        commit = None
        dirty = None
    versions = {"python": platform.python_version()}
    for package in (
        "captix",
        "torch",
        "torchvision",
        "onnx",
        "onnxruntime",
        "Pillow",
        "numpy",
        "psutil",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    cpu = _read("/proc/cpuinfo") or ""
    model = next(
        (
            row.split(":", 1)[1].strip()
            for row in cpu.splitlines()
            if row.startswith("model name")
        ),
        platform.processor(),
    )
    machine = _read("/etc/machine-id")
    return {
        "host_id": hashlib.sha256(machine.encode()).hexdigest() if machine else None,
        "cpu": model,
        "kernel": platform.release(),
        "ram_bytes": psutil.virtual_memory().total,
        "affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "cpu_quota_cores": cpu_limit,
        "memory_limit_bytes": memory_limit,
        "isolated": os.environ.get("CAPTIX_BENCH_ISOLATED") == "true",
        "commit": commit,
        "dirty": dirty,
        "threads": 2,
        "versions": versions,
        "memory_sampling_ms": 10,
        "cold_start": "spawn through first prediction",
        "transport": "in-process pipeline, no HTTP or disk read in timed requests",
    }


def _hash(path: Path) -> str:
    from captix_runtime.artifacts import sha256_file

    return sha256_file(path)


def _artifact(variant: str) -> dict:
    if variant == "onnx-fp32":
        from captix_runtime.artifacts import load_verified_manifest

        manifest = load_verified_manifest(
            Path(os.environ.get("CAPTIX_MODEL_DIR", "models/captix-v1"))
        )
        return {
            "source_sha256": manifest.source_sha256,
            "model_sha256": manifest.model_sha256,
            "revision_sha": manifest.revision_sha,
            "source_commit": manifest.source_commit,
            "export_versions": manifest.versions,
        }
    directory = Path(os.environ.get("CAPTIX_BENCH_CHECKPOINT_DIR", "models/checkpoint"))
    return {
        "source_sha256": {
            name: _hash(directory / name)
            for name in ("model.safetensors", "config.json")
        }
    }


def _predictor(variant: str):
    """Load local weights only; the original adapter retains original transforms/CTC."""
    if variant == "onnx-fp32":
        from captix_runtime.engine import OnnxEngine
        from captix_runtime.pipeline import recognize

        engine = OnnxEngine(
            Path(os.environ.get("CAPTIX_MODEL_DIR", "models/captix-v1"))
        )
        return lambda data, record: recognize(
            data,
            engine,
            mask=record.mask,
            length=record.length,
            case_sensitive=record.case_sensitive,
        )

    import io
    from types import SimpleNamespace

    import torch
    from PIL import Image

    from captix.dataset import decode_indices, transform_image
    from captix.decrypt import _apply_mask, _ctc_decode_fixed_length, _resolve_mask
    from captix.hub import load_local

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    model = (
        load_local(
            Path(os.environ.get("CAPTIX_BENCH_CHECKPOINT_DIR", "models/checkpoint"))
        )
        .cpu()
        .float()
        .eval()
    )
    if variant == "pytorch-compatible":
        from scripts.export_onnx import HeightMean

        model.collapse = HeightMean()

    def predict(data, record):
        phase = time.perf_counter()
        with Image.open(io.BytesIO(data)) as image:
            images = transform_image(image)
        timings = {"preprocess": (time.perf_counter() - phase) * 1000}
        phase = time.perf_counter()
        with (
            torch.no_grad() if variant == "pytorch-original" else torch.inference_mode()
        ):
            logits = model(images)
        timings["inference"] = (time.perf_counter() - phase) * 1000
        phase = time.perf_counter()
        logits = _apply_mask(
            logits, _resolve_mask(record.mask, model.vocab, record.case_sensitive)
        )
        if record.length is None:
            path = logits[:, 0, :].argmax(dim=-1).tolist()
        else:
            log_probs = torch.nn.functional.log_softmax(logits[:, 0, :], dim=-1).numpy()
            path = _ctc_decode_fixed_length(log_probs, record.length)
        text = decode_indices(path, model.vocab, collapse=True)
        if not record.case_sensitive:
            text = text.lower()
        timings["decode"] = (time.perf_counter() - phase) * 1000
        return SimpleNamespace(text=text, timings_ms=timings)

    return predict


def _percentiles(values):
    import numpy as np

    return {f"p{p}": float(np.percentile(values, p)) for p in (50, 95, 99)}


def _measure(predict, records, blobs, *, warmup, requests, seed):
    rng = random.Random(seed)
    order = list(range(len(records)))
    rng.shuffle(order)
    for i in range(warmup):
        index = order[i % len(order)]
        predict(blobs[index], records[index])
    timings = {stage: [] for stage in STAGES}
    started = time.perf_counter()
    for i in range(requests):
        index = order[i % len(order)]
        request_started = time.perf_counter()
        result = predict(blobs[index], records[index])
        total_ms = (time.perf_counter() - request_started) * 1000
        for stage in STAGES:
            timings[stage].append(
                total_ms if stage == "total" else result.timings_ms[stage]
            )
    elapsed = time.perf_counter() - started
    # Accuracy covers every unique sample, independently of the timed request count.
    accuracy = []
    for record, blob in zip(records, blobs):
        result = predict(blob, record)
        expected = record.label if record.case_sensitive else record.label.lower()
        accuracy.append(
            {
                "sha256": record.sha256,
                "source": record.source,
                "correct": result.text == expected,
            }
        )
    return {
        "latency_ms": {
            stage: _percentiles(values) for stage, values in timings.items()
        },
        "throughput": requests / elapsed,
    }, accuracy


def _worker(args):
    import resource

    records = verify_corpus(args.corpus)
    blobs = [(args.corpus.parent / record.path).read_bytes() for record in records]
    # Verify bytes actually used, even if the corpus changes after verification.
    if any(
        hashlib.sha256(blob).hexdigest() != record.sha256
        for record, blob in zip(records, blobs)
    ):
        raise ValueError("Corpus changed while loading")
    predict = _predictor(args.variant)
    predict(blobs[0], records[0])
    cold = (time.perf_counter() - args.started) * 1000
    run, accuracy = _measure(
        predict,
        records,
        blobs,
        warmup=args.warmup,
        requests=args.requests,
        seed=args.seed,
    )
    run.update(
        cold_start_ms=cold,
        rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    )
    args.result.write_text(
        json.dumps({"run": run, "accuracy": accuracy}), encoding="utf-8"
    )


def _pss(pid: int) -> int | None:
    content = _read(f"/proc/{pid}/smaps_rollup")
    if content:
        for line in content.splitlines():
            if line.startswith("Pss:"):
                return int(line.split()[1]) * 1024
    return None


def _docker_bytes() -> int | None:
    image = os.environ.get("CAPTIX_BENCH_DOCKER_IMAGE")
    if not image:
        return None
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Size}}", image],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        return int(result.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def run_benchmark(
    variant: str,
    corpus_manifest: Path,
    *,
    warmup: int = 100,
    requests: int = 1000,
    repetitions: int = 3,
    seed: int = 0,
) -> BenchmarkReport:
    if variant not in VARIANTS:
        raise ValueError("Unsupported benchmark variant")
    if (
        any(type(n) is not int or n < 1 for n in (warmup, requests, repetitions))
        or type(seed) is not int
        or seed < 0
    ):
        raise ValueError(
            "Counts must be positive integers and seed a nonnegative integer"
        )
    corpus_manifest = Path(corpus_manifest).resolve()
    records = verify_corpus(corpus_manifest)
    if not records:
        raise ValueError("Corpus must not be empty")
    metadata = json.loads(corpus_manifest.read_text())
    settings = {
        "warmup": warmup,
        "requests": requests,
        "repetitions": repetitions,
        "seed": seed,
        "batch_size": 1,
        "official": (warmup, requests, repetitions) == (100, 1000, 3),
    }
    report = BenchmarkReport(
        variant,
        "unavailable",
        settings,
        _environment(),
        {
            "sha256": _hash(corpus_manifest),
            "authorized": metadata.get("authorized") is True,
            "split": metadata.get("split", "unconfirmed"),
            "samples": len(records),
        },
        summary=summarize_runs([]),
        checks=dict.fromkeys(CHECKS),
    )
    try:
        report.artifact = _artifact(variant)
    except (ValueError, OSError):
        report.errors.append("local_model_unavailable_or_invalid")
        return report
    docker_size = _docker_bytes()
    for _ in range(repetitions):
        with tempfile.TemporaryDirectory(prefix="captix-bench-") as temporary:
            result = Path(temporary) / "result.json"
            command = [
                sys.executable,
                "-m",
                "benchmarks.bench_runtime",
                "--worker",
                "--variant",
                variant,
                "--corpus",
                str(corpus_manifest),
                "--warmup",
                str(warmup),
                "--requests",
                str(requests),
                "--seed",
                str(seed),
                "--result",
                str(result),
                "--started",
                str(time.perf_counter()),
            ]
            pss_peak = None
            # Never forward worker diagnostics: decoder exceptions can contain private text.
            with subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=Path(__file__).resolve().parents[1],
            ) as process:
                while process.poll() is None:
                    pss = _pss(process.pid)
                    if pss is not None:
                        pss_peak = max(pss_peak or 0, pss)
                    time.sleep(0.01)
            if process.returncode != 0 or not result.is_file():
                report.errors.append("worker_failed")
                return report
            data = json.loads(result.read_text())
            if report.accuracy and report.accuracy != data["accuracy"]:
                report.errors.append("accuracy_not_repeatable")
                return report
            report.accuracy = data["accuracy"]
            data["run"].update(pss_peak_bytes=pss_peak, docker_bytes=docker_size)
            report.runs.append(data["run"])
    if report.artifact != _artifact(variant) or report.corpus["sha256"] != _hash(
        corpus_manifest
    ):
        report.errors.append("inputs_changed_during_benchmark")
        return report
    report.status = "measured"
    report.summary = summarize_runs(report.runs)
    evidence_path = os.environ.get("CAPTIX_BENCH_EVIDENCE")
    if evidence_path:
        evidence = json.loads(Path(evidence_path).read_text())
        if (
            evidence.get("commit") == report.environment["commit"]
            and evidence.get("source_sha256") == report.artifact["source_sha256"]
            and evidence.get("corpus_sha256") == report.corpus["sha256"]
        ):
            report.checks = {
                name: evidence.get("checks", {}).get(name) is True for name in CHECKS
            }
        else:
            report.errors.append("evidence_does_not_match_run")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument(
        "--out", type=Path, default=Path("benchmarks/output/runtime.json")
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--started", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        _worker(args)
    else:
        report = run_benchmark(
            args.variant,
            args.corpus,
            warmup=args.warmup,
            requests=args.requests,
            repetitions=args.repetitions,
            seed=args.seed,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(asdict(report), indent=2, allow_nan=False) + "\n"
        )
        print(f"{report.variant}: {report.status}; report written")


if __name__ == "__main__":
    main()
