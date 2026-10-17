"""Small fixed-shape benchmark for the runtime CTC decoder."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from captix_runtime.ctc import decode_fixed_length


@dataclass(frozen=True)
class DecoderCase:
    logits: np.ndarray
    vocab: Sequence[str]
    length: int
    reference: str
    mask: str | None = None


def bench_decoder(cases: Sequence[DecoderCase], iterations: int) -> dict:
    if not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")

    results = []
    for case in cases:
        started = time.perf_counter()
        output = ""
        for _ in range(iterations):
            output = decode_fixed_length(
                case.logits, case.vocab, case.length, case.mask
            )
        elapsed_ms = (time.perf_counter() - started) * 1000 / iterations
        results.append(
            {
                "T": int(case.logits.shape[0]),
                "L": case.length,
                "C": int(case.logits.shape[1]),
                "latency_ms": elapsed_ms,
                "matches_reference": output == case.reference,
            }
        )
    return {"iterations": iterations, "cases": results}


def _benchmark_cases() -> list[DecoderCase]:
    rng = np.random.default_rng(1042)
    vocab = [chr(33 + i) for i in range(62)]
    references = {
        (20, 4): 'W/"^',
        (20, 5): 'WH/"^',
        (40, 4): "T;,]",
        (40, 5): "T;,2]",
        (80, 4): "WB.B",
        (80, 5): "WSB.B",
    }
    cases = []
    for timesteps in (20, 40, 80):
        logits = rng.normal(size=(timesteps, 63)).astype(np.float32)
        for length in (4, 5):
            cases.append(
                DecoderCase(
                    logits,
                    vocab,
                    length,
                    references[(timesteps, length)],
                )
            )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(bench_decoder(_benchmark_cases(), args.iterations), indent=2))


if __name__ == "__main__":
    main()
