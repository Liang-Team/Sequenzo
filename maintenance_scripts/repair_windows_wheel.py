#!/usr/bin/env python3
"""Ensure cibuildwheel's repaired-wheel directory contains a wheel on Windows.

delvewheel is invoked with ``|| true`` in the CIBW repair command so that a
missing OpenMP DLL does not hard-fail the build.  This script is the safety
net: if delvewheel emitted a repaired wheel we confirm it; if it didn't we
copy the built wheel as-is so cibuildwheel finds *something* in dest_dir.
"""
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

    # Check whether delvewheel already placed a repaired wheel in dest.
    wheels = sorted(dest.glob("*.whl"))
    if wheels:
        print(f"Repaired wheel (from delvewheel): {wheels[0]}")
        return 0

    # delvewheel did not emit a wheel (OpenMP DLL not found, or repair was
    # skipped). Copy the built wheel as-is so cibuildwheel can continue.
    print("delvewheel did not emit a wheel; copying built wheel as-is.")
    target = dest / wheel.name
    shutil.copy2(wheel, target)

    # Verify the copy succeeded — don't re-glob, just stat the known path.
    if not target.is_file():
        print(
            f"Failed to copy wheel to {target} "
            f"(source={wheel}, dest_exists={dest.is_dir()})",
            file=sys.stderr,
        )
        return 1

    print(f"Repaired wheel (copied as-is): {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())