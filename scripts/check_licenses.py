#!/usr/bin/env python3
"""Verify third-party dependency licenses against an allowlist."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = ROOT / "licenses" / "ALLOWED_LICENSES.txt"
OUTPUT = ROOT / "licenses" / "THIRD_PARTY_LICENSES.md"
PACKAGE_LINE = re.compile(r"^\s+([A-Za-z0-9][A-Za-z0-9._-]*)==")
PROJECT_NAME = "pioneer-ml"


def _load_allowlist() -> str:
    tokens = [
        line.strip()
        for line in ALLOWLIST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    return ";".join(tokens)


def _project_packages() -> list[str]:
    result = subprocess.run(
        [sys.executable, "-m", "pipdeptree", "-p", PROJECT_NAME, "-f", "--warn", "silence"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)

    packages: set[str] = set()
    for line in result.stdout.splitlines():
        match = PACKAGE_LINE.match(line)
        if match:
            packages.add(match.group(1))
    return sorted(packages)


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "piplicenses", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def generate_report() -> str:
    packages = _project_packages()
    result = _run(
        [
            "--format=markdown",
            "--with-urls",
            "--partial-match",
            "--packages",
            *packages,
        ]
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)

    header = (
        "# Third-Party Licenses\n\n"
        "Auto-generated inventory of Python package licenses for Pioneer ML dependencies.\n"
        "Regenerate with `make licenses`.\n\n"
        "> Pioneer ML is licensed under MIT. Third-party packages retain their own licenses.\n\n"
        f"> Scope: direct and transitive dependencies of `{PROJECT_NAME}` "
        f"({len(packages)} packages).\n\n"
    )
    return header + result.stdout


def check_allowlist(*, strict: bool) -> int:
    packages = _project_packages()
    result = _run(
        [
            "--allow-only",
            _load_allowlist(),
            "--partial-match",
            "--packages",
            *packages,
        ]
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        print(
            "\nLicense check failed: one or more packages use non-approved licenses.",
            file=sys.stderr,
        )
        if not strict:
            print(
                "Use --strict for CI. Run `make licenses` to inspect the full inventory.",
                file=sys.stderr,
            )
        return result.returncode

    print(
        f"License check passed: all {len(packages)} project dependencies "
        "use approved licenses."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["check", "generate"],
        help="check allowlist or generate THIRD_PARTY_LICENSES.md",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on license violations (for CI)",
    )
    args = parser.parse_args()

    for module in ("piplicenses", "pipdeptree"):
        try:
            __import__(module)
        except ImportError:
            print(
                f"{module} is required. Install with: pip install pip-licenses pipdeptree",
                file=sys.stderr,
            )
            return 1

    if args.command == "generate":
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(generate_report(), encoding="utf-8")
        print(f"Wrote {OUTPUT}")
        return 0

    return check_allowlist(strict=args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
