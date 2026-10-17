"""Validate repository branch names."""

import argparse
import re
import subprocess
import sys

BRANCH_NAME = re.compile(
    r"^(?:main|(?:feat|fix|docs|refactor|test|chore|ci|build|perf)/[a-z0-9]+(?:-[a-z0-9]+)*)$"
)


def is_valid_branch_name(branch: str) -> bool:
    return BRANCH_NAME.fullmatch(branch) is not None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--branch", help="branch name to validate (defaults to current branch)"
    )
    args = parser.parse_args(argv)

    if args.branch is not None:
        branch = args.branch
    else:
        try:
            branch = subprocess.run(
                ["git", "branch", "--show-current"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"Cannot determine current branch: {exc}", file=sys.stderr)
            return 1

    if not is_valid_branch_name(branch):
        print(f"Invalid branch name: {branch or '(detached HEAD)'}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
