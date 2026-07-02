#!/usr/bin/env python3
"""
macOS wheel repair helper for cibuildwheel.

1. Run delocate-wheel with explicit library search paths so the CI-built libomp
   (with __kmpc_dispatch_deinit stub) is bundled instead of any system copy.
2. Verify the bundled libomp exports __kmpc_dispatch_deinit.
3. If delocate picked a libomp without the stub, replace it with the CI libomp
   and rewrite the wheel RECORD.

Usage (called from CIBW_REPAIR_WHEEL_COMMAND_MACOS, cwd = project root):
    python maintenance_scripts/repair_macos_wheel.py <dest_dir> <wheel> [delocate_archs]
"""
from __future__ import annotations

import base64
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

DISPATCH_DEINIT_MARKER = "__kmpc_dispatch_deinit"


def libomp_exports_dispatch_deinit(libomp_path: Path) -> bool:
    """Return True when libomp.dylib defines (exports) the dispatch_deinit stub."""
    result = subprocess.run(
        ["nm", "-gU", str(libomp_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return False
    return DISPATCH_DEINIT_MARKER in result.stdout


def _find_bundled_libomp(root: Path) -> Path | None:
    matches = sorted(root.glob("**/.dylibs/libomp*.dylib"))
    return matches[0] if matches else None


def _sha256_record_line(path: Path) -> str:
    data = path.read_bytes()
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode("ascii").rstrip("=")
    return f"sha256={digest},{len(data)}"


def _rewrite_record(root: Path) -> None:
    dist_info_dirs = sorted(root.glob("*.dist-info"))
    if not dist_info_dirs:
        raise SystemExit("[repair-macos] ERROR: missing .dist-info directory in wheel")
    dist_info = dist_info_dirs[0]
    record_path = dist_info / "RECORD"
    entries: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel.endswith(".dist-info/RECORD"):
            continue
        entries.append(f"{rel},{_sha256_record_line(path)}")
    entries.append(f"{dist_info.name}/RECORD,,")
    record_path.write_text("\n".join(entries) + "\n", encoding="utf-8")


def _repack_wheel(root: Path, wheel_path: Path) -> None:
    _rewrite_record(root)
    if wheel_path.exists():
        wheel_path.unlink()
    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            zf.write(path, path.relative_to(root).as_posix())


def _ensure_bundled_libomp(wheel_path: Path, ci_libomp: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="sequenzo-wheel-") as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(wheel_path, "r") as zf:
            zf.extractall(root)

        bundled = _find_bundled_libomp(root)
        if bundled is None:
            print("[repair-macos] ERROR: repaired wheel has no bundled libomp in .dylibs/", file=sys.stderr)
            raise SystemExit(1)

        if libomp_exports_dispatch_deinit(bundled):
            print(f"[repair-macos] [OK] Bundled libomp exports {DISPATCH_DEINIT_MARKER}: {bundled}")
            return

        print(
            "[repair-macos] [WARN] Bundled libomp missing "
            f"{DISPATCH_DEINIT_MARKER}; replacing with CI libomp",
            file=sys.stderr,
        )
        shutil.copy2(ci_libomp, bundled)
        if not libomp_exports_dispatch_deinit(bundled):
            print(
                "[repair-macos] ERROR: CI libomp also lacks exported "
                f"{DISPATCH_DEINIT_MARKER}: {ci_libomp}",
                file=sys.stderr,
            )
            raise SystemExit(1)

        _repack_wheel(root, wheel_path)
        print(f"[repair-macos] [OK] Rewrote wheel with CI libomp stub: {wheel_path}")


def main(argv: list[str]) -> int:
    if len(argv) not in (3, 4):
        print(
            "Usage: repair_macos_wheel.py <dest_dir> <wheel> [delocate_archs]",
            file=sys.stderr,
        )
        return 2

    dest_dir = Path(argv[1])
    wheel = Path(argv[2])
    delocate_archs = argv[3] if len(argv) == 4 else ""

    repair_library_path = os.environ.get("REPAIR_LIBRARY_PATH", "").strip()
    if not repair_library_path:
        print("[repair-macos] ERROR: REPAIR_LIBRARY_PATH is not set", file=sys.stderr)
        return 1

    ci_libomp = Path(repair_library_path) / "libomp.dylib"
    if not ci_libomp.is_file():
        print(f"[repair-macos] ERROR: CI libomp not found: {ci_libomp}", file=sys.stderr)
        return 1

    if not libomp_exports_dispatch_deinit(ci_libomp):
        print(
            "[repair-macos] ERROR: CI libomp lacks exported "
            f"{DISPATCH_DEINIT_MARKER}. Rebuild via build_macos_ci_libomp.sh.",
            file=sys.stderr,
        )
        return 1

    print(f"[repair-macos] === Repairing macOS wheel ===")
    print(f"[repair-macos] Wheel: {wheel}")
    print(f"[repair-macos] REPAIR_LIBRARY_PATH: {repair_library_path}")
    print(f"[repair-macos] CI libomp: {ci_libomp}")

    dest_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = f"{repair_library_path}:{env.get('DYLD_LIBRARY_PATH', '')}".rstrip(":")

    delocate_cmd = [
        "delocate-wheel",
        "-w",
        str(dest_dir),
        "-v",
        str(wheel),
        "-L",
        repair_library_path,
    ]
    if delocate_archs:
        delocate_cmd[1:1] = ["--require-archs", delocate_archs]

    result = subprocess.run(delocate_cmd, env=env, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)
    if result.returncode != 0:
        print("[repair-macos] ERROR: delocate-wheel failed", file=sys.stderr)
        return result.returncode

    repaired_wheels = sorted(dest_dir.glob("*.whl"))
    if not repaired_wheels:
        print("[repair-macos] ERROR: delocate produced no wheel in dest_dir", file=sys.stderr)
        return 1

    repaired_wheel = repaired_wheels[0]
    print(f"[repair-macos] Repaired wheel: {repaired_wheel}")

    _ensure_bundled_libomp(repaired_wheel, ci_libomp)

    listdeps = subprocess.run(
        ["delocate-listdeps", "--all", str(repaired_wheel)],
        text=True,
        capture_output=True,
        check=False,
    )
    if listdeps.stdout:
        print(listdeps.stdout, end="" if listdeps.stdout.endswith("\n") else "\n")
    if "libomp" not in (listdeps.stdout or ""):
        print("[repair-macos] ERROR: libomp is not bundled into the macOS wheel.", file=sys.stderr)
        return 1

    print("[repair-macos] [OK] libomp successfully bundled into wheel")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
