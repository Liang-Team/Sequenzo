#!/usr/bin/env python3
"""Ensure cibuildwheel's repaired-wheel directory contains a wheel on Windows."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "Usage: repair_windows_wheel.py <dest_dir> <wheel>",
            file=sys.stderr,
        )
        return 2

    dest = Path(sys.argv[1]).resolve()
    wheel = Path(sys.argv[2]).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    if not wheel.is_file():
        print(f"Source wheel not found: {wheel}", file=sys.stderr)
        return 1

    wheels = sorted(dest.glob("*.whl"))
    if not wheels:
        print("delvewheel did not emit a wheel; copying built wheel as-is")
        target = dest / wheel.name
        shutil.copy2(wheel, target)
        wheels = sorted(dest.glob("*.whl"))
        if not wheels and target.is_file():
            wheels = [target]

    if not wheels:
        print(
            f"Failed to place repaired wheel in {dest} "
            f"(source={wheel}, dest_exists={dest.is_dir()})",
            file=sys.stderr,
        )
        return 1

    print(f"Repaired wheel: {wheels[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
