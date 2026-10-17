"""Explicitly download safetensors and config from an immutable Hub revision."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def fetch_checkpoint(repo_id: str, revision_sha: str, destination: Path) -> Path:
    if re.fullmatch(r"[0-9a-f]{40}", revision_sha) is None:
        raise ValueError("revision_sha must be a lowercase 40-character commit SHA")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        revision=revision_sha,
        local_dir=destination,
        allow_patterns=["model.safetensors", "config.json"],
    )
    for name in ("model.safetensors", "config.json"):
        if not (destination / name).is_file():
            raise FileNotFoundError(destination / name)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="jtrecenti/txtcaptcha-crnn")
    parser.add_argument(
        "--revision-sha", required=True, help="Immutable 40-character model commit SHA"
    )
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    print(fetch_checkpoint(args.repo_id, args.revision_sha, args.destination))


if __name__ == "__main__":
    main()
