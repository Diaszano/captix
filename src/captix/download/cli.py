"""``download_captchas`` CLI entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from ._base import available_sources, download_captchas


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="download_captchas",
        description=(
            "Download fresh captchas from one or more supported sources. "
            "Failed fetches (site down, parse error) are skipped — the "
            "requested count is a target, not a guarantee."
        ),
    )
    p.add_argument(
        "-s",
        "--source",
        action="append",
        default=None,
        help=(
            "Source name. Pass multiple times, or use 'all' to hit every "
            f"source. Available: {', '.join(available_sources())}."
        ),
    )
    p.add_argument(
        "-n",
        "--n",
        type=int,
        default=16,
        help="Target number of captchas per source (default: 16).",
    )
    p.add_argument(
        "-d",
        "--dest",
        type=Path,
        required=True,
        help=(
            "Destination directory. One subdirectory per source will be "
            "created (e.g. <dest>/esaj/)."
        ),
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to wait between successful fetches (default: 0).",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-request timeout in seconds (default: 30).",
    )
    return p


def _resolve_sources(requested: Sequence[str] | None) -> list[str]:
    pool = available_sources()
    if not requested or "all" in requested:
        return pool
    unknown = [s for s in requested if s not in pool]
    if unknown:
        raise SystemExit(f"Unknown source(s): {unknown}. Available: {pool}")
    # Preserve user order, drop duplicates.
    seen: list[str] = []
    for s in requested:
        if s not in seen:
            seen.append(s)
    return seen


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sources = _resolve_sources(args.source)
    total = 0
    for src in sources:
        dest = args.dest / src
        print(f"[{src}] downloading up to {args.n} captchas into {dest}")
        saved = download_captchas(
            src,
            args.n,
            dest,
            sleep=args.sleep,
            timeout=args.timeout,
            on_error="skip",
        )
        print(f"[{src}] saved {len(saved)}/{args.n}")
        total += len(saved)
    print(f"total saved: {total}")
    return 0 if total > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
