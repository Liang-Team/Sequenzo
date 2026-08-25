#!/usr/bin/env python3
"""Inspect OpenMP linkage for extension modules in this source checkout."""

from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EXTENSION_MODULES = (
    "sequenzo.dissimilarity_measures.c_code",
    "sequenzo.clustering.clustering_c_code",
)


def _dependency_command(extension_path: str) -> list[str] | None:
    if sys.platform == "darwin":
        return ["otool", "-L", extension_path]
    if sys.platform.startswith("linux"):
        return ["ldd", extension_path]
    return None


def check_extension(module_name: str) -> bool:
    module = importlib.import_module(module_name)
    extension_path = str(Path(module.__file__).resolve())
    print(f"{module_name}: {extension_path}")

    command = _dependency_command(extension_path)
    if command is None:
        print("This diagnostic supports macOS and Linux only.")
        return False

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(result.stderr.strip())
        return False

    linked = any(name in result.stdout for name in ("libomp", "libgomp"))
    print(result.stdout.strip())
    print(f"OpenMP runtime found: {linked}")
    return linked


def check_in_fresh_process(module_name: str) -> bool:
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--module", module_name],
        cwd=PROJECT_ROOT,
        check=False,
    )
    return result.returncode == 0


def main(argv: list[str] | None = None):
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        if len(argv) != 2 or argv[0] != "--module":
            print("Usage: python developer/test_openmp.py [--module MODULE]")
            return False
        try:
            return check_extension(argv[1])
        except ImportError as exc:
            print(f"{argv[1]}: import failed: {exc}")
            return False

    return all(
        [check_in_fresh_process(module_name) for module_name in EXTENSION_MODULES]
    )


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
