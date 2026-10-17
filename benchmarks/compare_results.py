"""Aggregate paired accuracy and enforce the reviewed ONNX promotion policy."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_runtime import (
    CHECKS,
    SCALARS,
    STAGES,
    BenchmarkReport,
    summarize_runs,
)


@dataclass(frozen=True)
class LabeledResult:
    sha256: str
    source: str
    baseline_correct: bool
    candidate_correct: bool


@dataclass(frozen=True)
class PromotionDecision:
    status: Literal["PROMOTE", "HOLD"]
    reasons: list[str]


def paired_accuracy(labeled: Sequence[LabeledResult], *, seed: int = 0) -> dict:
    """Seeded 10,000-draw paired percentile bootstrap; deltas are percentage points."""
    import numpy as np

    rng = np.random.default_rng(seed)

    def aggregate(rows):
        n = len(rows)
        if not n:
            return {
                "n": 0,
                "baseline_exact_match": None,
                "candidate_exact_match": None,
                "delta_pp": None,
                "ci95_pp": None,
            }
        deltas = np.array(
            [int(row.candidate_correct) - int(row.baseline_correct) for row in rows]
        )
        # Resampling counts of paired differences is equivalent to resampling paired rows.
        counts = np.array([(deltas == value).sum() for value in (-1, 0, 1)])
        draws = rng.multinomial(n, counts / n, size=10000)
        samples = (draws[:, 2] - draws[:, 0]) * (100 / n)
        return {
            "n": n,
            "baseline_exact_match": sum(row.baseline_correct for row in rows) / n,
            "candidate_exact_match": sum(row.candidate_correct for row in rows) / n,
            "delta_pp": int(deltas.sum()) * 100 / n,
            "ci95_pp": np.percentile(samples, [2.5, 97.5]).tolist(),
        }

    result = aggregate(labeled)
    result["sources"] = {
        source: aggregate([row for row in labeled if row.source == source])
        for source in sorted({row.source for row in labeled})
    }
    result.update(bootstrap_seed=seed, bootstrap_draws=10000, confidence=0.95)
    return result


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def evaluate_promotion(
    baseline: BenchmarkReport,
    candidate: BenchmarkReport,
    labeled: Sequence[LabeledResult],
) -> PromotionDecision:
    reasons = []
    if baseline.variant != "pytorch-original" or candidate.variant != "onnx-fp32":
        reasons.append("Require pytorch-original baseline and onnx-fp32 candidate")
    for name, report in (("baseline", baseline), ("candidate", candidate)):
        if report.status != "measured" or report.errors:
            reasons.append(f"{name}: actual measurements unavailable or invalid")
        official = {
            "warmup": 100,
            "requests": 1000,
            "repetitions": 3,
            "batch_size": 1,
            "official": True,
        }
        if (
            any(report.settings.get(k) != v for k, v in official.items())
            or len(report.runs) != 3
        ):
            reasons.append(f"{name}: official 100/1000/3, B=1 measurements required")
        if (
            report.corpus.get("authorized") is not True
            or report.corpus.get("split") != "test"
        ):
            reasons.append(f"{name}: authorized held-out test corpus required")
        if report.environment.get("isolated") is not True:
            reasons.append(f"{name}: isolated host not confirmed")
        if report.environment.get("dirty") is not False:
            reasons.append(f"{name}: clean committed source required")
        if (
            report.environment.get("cpu_quota_cores") != 2
            or report.environment.get("memory_limit_bytes") != 1073741824
        ):
            reasons.append(f"{name}: reference profile must be 2 vCPU / 1 GiB")
        try:
            if report.summary != summarize_runs(report.runs):
                reasons.append(f"{name}: summary does not match repetitions")
        except (KeyError, TypeError, ValueError):
            reasons.append(f"{name}: invalid repetition metrics")
        for metric in SCALARS:
            # PSS is explicitly optional when Linux smaps_rollup is unavailable.
            if metric != "pss_peak_bytes" and not _positive(
                report.summary.get(metric, {}).get("median")
            ):
                reasons.append(f"{name}: {metric} unavailable or invalid")
        for stage in STAGES:
            for percentile in ("p50", "p95", "p99"):
                value = (
                    report.summary.get("latency_ms", {})
                    .get(stage, {})
                    .get(percentile, {})
                    .get("median")
                )
                if not _positive(value):
                    reasons.append(
                        f"{name}: {stage}/{percentile} unavailable or invalid"
                    )
    comparable = (
        "host_id",
        "cpu",
        "kernel",
        "ram_bytes",
        "affinity",
        "cpu_quota_cores",
        "memory_limit_bytes",
        "threads",
        "commit",
    )
    if any(
        baseline.environment.get(key) is None
        or baseline.environment.get(key) != candidate.environment.get(key)
        for key in comparable
    ):
        reasons.append(
            "Measurements must share host, CPU affinity, quotas, threads, kernel and commit"
        )
    if baseline.environment.get("versions") != candidate.environment.get("versions"):
        reasons.append("Measurement software versions differ")
    if baseline.settings.get("seed") != candidate.settings.get("seed"):
        reasons.append("Measurement seeds differ")
    if baseline.corpus != candidate.corpus:
        reasons.append("Measurements must share the same frozen corpus")
    source = baseline.artifact.get("source_sha256")
    if (
        not isinstance(source, dict)
        or set(source) != {"model.safetensors", "config.json"}
        or any(
            not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in source.values()
        )
        or source != candidate.artifact.get("source_sha256")
    ):
        reasons.append("Measurements must use identical verified checkpoint hashes")
    for check in CHECKS:
        if candidate.checks.get(check) is not True:
            reasons.append(
                f"{check}: passed evidence unavailable (golden requires 100% textual equality)"
            )
    valid_rows = all(
        isinstance(row.sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", row.sha256)
        and isinstance(row.source, str)
        and row.source.strip()
        and type(row.baseline_correct) is bool
        and type(row.candidate_correct) is bool
        for row in labeled
    )
    if not valid_rows or len({row.sha256 for row in labeled}) != len(labeled):
        reasons.append(
            "Labeled evidence requires valid unique sample hashes and paired booleans"
        )
    if len(labeled) < 1000:
        reasons.append(
            "Insufficient evidence: require at least 1000 authorized unique labeled samples"
        )
    for is_candidate, report in ((False, baseline), (True, candidate)):
        expected = [
            {
                "sha256": row.sha256,
                "source": row.source,
                "correct": row.candidate_correct
                if is_candidate
                else row.baseline_correct,
            }
            for row in labeled
        ]
        if sorted(expected, key=lambda row: row["sha256"]) != sorted(
            report.accuracy, key=lambda row: row["sha256"]
        ) or report.corpus.get("samples") != len(labeled):
            reasons.append(
                "Paired evidence does not match actual benchmark accuracy samples"
            )
    if valid_rows and labeled:
        accuracy = paired_accuracy(labeled, seed=baseline.settings.get("seed", 0))
        if abs(accuracy["delta_pp"]) > 0.2 + 1e-12:
            reasons.append("overall exact-match delta outside [-0.2, +0.2] p.p.")
        for source, result in accuracy["sources"].items():
            if result["n"] >= 100 and result["delta_pp"] < -1.0 - 1e-12:
                # Source names belong to the private manifest, keep report messages aggregate.
                reasons.append(
                    f"source {source}: exact-match drop exceeds 1 p.p. (n >= 100)"
                )
    try:
        latency_ratio = (
            candidate.summary["latency_ms"]["total"]["p95"]["median"]
            / baseline.summary["latency_ms"]["total"]["p95"]["median"]
        )
        rss_ratio = (
            candidate.summary["rss_peak_bytes"]["median"]
            / baseline.summary["rss_peak_bytes"]["median"]
        )
        if not (
            math.isfinite(latency_ratio)
            and math.isfinite(rss_ratio)
            and (
                (latency_ratio <= 0.90 and rss_ratio <= 1.05)
                or (rss_ratio <= 0.80 and latency_ratio <= 1.05)
            )
        ):
            reasons.append(
                "Value gate: require p95 <= 0.90x or RSS <= 0.80x, with other metric <= 1.05x"
            )
    except (KeyError, TypeError, ZeroDivisionError):
        reasons.append("Value gate: p95/RSS not measured")
    return PromotionDecision(
        "HOLD" if reasons else "PROMOTE", list(dict.fromkeys(reasons))
    )


def pair_reports(
    baseline: BenchmarkReport, candidate: BenchmarkReport
) -> list[LabeledResult]:
    """Join only real per-sample correctness records; mismatches stay a HOLD gate."""
    candidates = {row["sha256"]: row for row in candidate.accuracy}
    return [
        LabeledResult(
            row["sha256"],
            row["source"],
            row["correct"],
            candidates[row["sha256"]]["correct"],
        )
        for row in baseline.accuracy
        if row["sha256"] in candidates
        and row["source"] == candidates[row["sha256"]]["source"]
    ]


def write_comparison(
    baseline: BenchmarkReport,
    candidate: BenchmarkReport,
    labeled: Sequence[LabeledResult],
    out: Path,
) -> PromotionDecision:
    decision = evaluate_promotion(baseline, candidate, labeled)
    accuracy = paired_accuracy(labeled, seed=baseline.settings.get("seed", 0))
    reports = []
    for report in (baseline, candidate):
        data = asdict(report)
        data.pop("accuracy")
        reports.append(data)
    payload = {
        "decision": asdict(decision),
        "accuracy": accuracy,
        "baseline": reports[0],
        "candidate": reports[1],
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    table = []
    for report in (baseline, candidate):
        for stage in STAGES:
            for p in ("p50", "p95", "p99"):
                table.append(
                    (
                        report.variant,
                        f"{stage}_{p}_ms",
                        report.summary.get("latency_ms", {}).get(stage, {}).get(p, {}),
                    )
                )
        table.extend(
            (report.variant, metric, report.summary.get(metric, {}))
            for metric in SCALARS
        )
    accuracy_groups = [("overall", accuracy), *accuracy["sources"].items()]
    with out.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "variant",
                "metric",
                "median",
                "min",
                "max",
                "stdev",
                "source",
                "n",
                "baseline_exact_match",
                "candidate_exact_match",
                "delta_pp",
                "ci95_low_pp",
                "ci95_high_pp",
            )
        )
        for variant, metric, values in table:
            writer.writerow(
                (
                    variant,
                    metric,
                    *(
                        values.get(k) if values.get(k) is not None else "unavailable"
                        for k in ("median", "min", "max", "stdev")
                    ),
                    *("" for _ in range(7)),
                )
            )
        for source, result in accuracy_groups:
            writer.writerow(
                (
                    "paired",
                    "exact_match",
                    "",
                    "",
                    "",
                    "",
                    source,
                    result["n"],
                    result["baseline_exact_match"],
                    result["candidate_exact_match"],
                    result["delta_pp"],
                    *(result["ci95_pp"] or ("unavailable", "unavailable")),
                )
            )
    lines = [
        f"# Promotion: {decision.status}",
        "",
        "NOT MEASURED / unavailable values are never zero.",
        "",
        "| Variant | Metric | Median | Min | Max | Stddev |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for variant, metric, values in table:
        cells = [
            str(values.get(k)) if values.get(k) is not None else "unavailable"
            for k in ("median", "min", "max", "stdev")
        ]
        lines.append(f"| {variant} | {metric} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            (
                f"Paired samples: {accuracy['n']}; delta p.p.: {accuracy['delta_pp']}; "
                f"95% paired bootstrap CI: {accuracy['ci95_pp']}."
            ),
            "",
            "The confidence interval does not replace the product policy thresholds.",
            "",
        ]
    )
    lines.extend(
        [
            "| Source | n | Baseline exact-match | ONNX exact-match | Delta p.p. | 95% CI p.p. |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for source, result in accuracy_groups:
        cells = [
            str(result[key]) if result[key] is not None else "unavailable"
            for key in (
                "n",
                "baseline_exact_match",
                "candidate_exact_match",
                "delta_pp",
                "ci95_pp",
            )
        ]
        safe_source = source.replace("|", "\\|").replace("\n", " ").replace("\r", " ")
        lines.append(f"| {safe_source} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.extend(f"- {reason}" for reason in decision.reasons)
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument(
        "--out", type=Path, default=Path("benchmarks/output/comparison")
    )
    args = parser.parse_args()
    baseline = BenchmarkReport(**json.loads(args.baseline.read_text()))
    candidate = BenchmarkReport(**json.loads(args.candidate.read_text()))
    decision = write_comparison(
        baseline, candidate, pair_reports(baseline, candidate), args.out
    )
    print(decision.status)
    for reason in decision.reasons:
        print(f"- {reason}")


if __name__ == "__main__":
    main()
