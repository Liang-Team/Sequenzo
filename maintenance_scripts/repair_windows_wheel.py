#!/usr/bin/env python3
"""
Windows wheel repair helper for cibuildwheel.

Responsibilities:
1. Locate the MSVC LLVM OpenMP DLL (libomp140.x86_64.dll) via vswhere or
   well-known fallback paths on GitHub-hosted runners.
2. Run `delvewheel repair --add-path <dir>` to bundle it into the wheel.
3. If delvewheel does not produce a repaired wheel (DLL not found, or the
   wheel has no external dependencies), copy the built wheel as-is so that
   cibuildwheel always finds something in dest_dir.

Usage (called from CIBW_REPAIR_WHEEL_COMMAND_WINDOWS, cwd = project root):
    python maintenance_scripts/repair_windows_wheel.py <dest_dir> <wheel>
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# DLL discovery
# ---------------------------------------------------------------------------

def _find_vswhere() -> Path | None:
    """Return the path to vswhere.exe, or None if not found."""
    candidates = []

    # The canonical env var on Windows is "ProgramFiles(x86)"; os.environ gives
    # it under that exact key (parentheses and all).
    pf86 = os.environ.get("ProgramFiles(x86)") or os.environ.get("PROGRAMFILES(X86)")
    if pf86:
        candidates.append(Path(pf86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe")

    # Some shells expose a sanitised name instead.
    pf86b = os.environ.get("PROGRAMFILES_X86")
    if pf86b:
        candidates.append(Path(pf86b) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe")

    # Hard-coded last resort (GitHub-hosted Windows runners, VS2022 Enterprise).
    candidates.append(Path("C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe"))

    for p in candidates:
        if p.is_file():
            return p
    return None


def _find_omp_dll_via_vswhere(vswhere: Path) -> Path | None:
    """Ask vswhere for the LLVM OpenMP DLL bundled with MSVC tools."""
    try:
        result = subprocess.run(
            [
                str(vswhere),
                "-latest", "-products", "*",
                "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                # vswhere -find uses Windows path separators and glob patterns.
                "-find", r"VC\Tools\Llvm\x64\bin\libomp140.x86_64.dll",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        print(f"[repair] vswhere failed: {exc}", file=sys.stderr)
        return None

    for line in result.stdout.splitlines():
        p = Path(line.strip())
        if p.is_file():
            print(f"[repair] Found DLL via vswhere: {p}")
            return p
    return None


def _find_omp_dll_fallback() -> Path | None:
    """Walk well-known hard-coded paths on GitHub-hosted runners."""
    candidates = [
        "C:/Program Files/Microsoft Visual Studio/2022/Enterprise/VC/Tools/Llvm/x64/bin/libomp140.x86_64.dll",
        "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/Llvm/x64/bin/libomp140.x86_64.dll",
        "C:/Program Files/Microsoft Visual Studio/2022/Professional/VC/Tools/Llvm/x64/bin/libomp140.x86_64.dll",
        "C:/Program Files/LLVM/bin/libomp140.x86_64.dll",
        "C:/Program Files/LLVM/bin/libomp.dll",
    ]
    for c in candidates:
        p = Path(c)
        if p.is_file():
            print(f"[repair] Found DLL via fallback path: {p}")
            return p
    return None


def find_omp_dll() -> Path | None:
    vswhere = _find_vswhere()
    if vswhere:
        print(f"[repair] Using vswhere: {vswhere}")
        dll = _find_omp_dll_via_vswhere(vswhere)
        if dll:
            return dll
        print("[repair] vswhere did not find the DLL; trying fallback paths...")
    else:
        print("[repair] vswhere.exe not found; trying fallback paths...")
    return _find_omp_dll_fallback()


# ---------------------------------------------------------------------------
# Wheel repair
# ---------------------------------------------------------------------------

def run_delvewheel(dest: Path, wheel: Path, omp_bin: Path | None) -> bool:
    """Run delvewheel; return True if it produced at least one wheel."""
    cmd = ["delvewheel", "repair", "-w", str(dest)]
    if omp_bin:
        cmd += ["--add-path", str(omp_bin)]
    cmd.append(str(wheel))

    print(f"[repair] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[repair] delvewheel exited with code {result.returncode} (continuing...)")

    return bool(sorted(dest.glob("*.whl")))


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: repair_windows_wheel.py <dest_dir> <wheel>", file=sys.stderr)
        return 2

    dest = Path(sys.argv[1]).resolve()
    wheel = Path(sys.argv[2]).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    if not wheel.is_file():
        print(f"[repair] Source wheel not found: {wheel}", file=sys.stderr)
        return 1

    # --- Step 1: try to find the OpenMP DLL ---
    omp_dll = find_omp_dll()
    omp_bin = omp_dll.parent if omp_dll else None
    if omp_bin:
        print(f"[repair] Bundling OpenMP from: {omp_bin}")
    else:
        print("[repair] WARNING: OpenMP DLL not found; wheel will be built without it")

    # --- Step 2: run delvewheel ---
    if run_delvewheel(dest, wheel, omp_bin):
        repaired = sorted(dest.glob("*.whl"))[0]
        print(f"[repair] Repaired wheel (delvewheel): {repaired}")
        return _finalize_repaired_wheel(repaired)

    # --- Step 3: safety net — copy built wheel as-is ---
    print("[repair] delvewheel did not emit a wheel; copying built wheel as-is.")
    target = dest / wheel.name
    shutil.copy2(wheel, target)
    if not target.is_file():
        print(f"[repair] ERROR: failed to copy {wheel} -> {target}", file=sys.stderr)
        return 1

    print(f"[repair] Repaired wheel (copied as-is): {target}")
    return _finalize_repaired_wheel(target)


def _verify_bundled_openmp(wheel_path: Path) -> None:
    import zipfile

    with zipfile.ZipFile(wheel_path) as zf:
        dlls = [
            name
            for name in zf.namelist()
            if name.startswith("sequenzo.libs/")
            and name.lower().endswith(".dll")
            and "libomp140" in name.lower()
        ]
    if not dlls:
        print(
            f"[repair] ERROR: missing bundled libomp140*.dll in sequenzo.libs for {wheel_path}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(f"[OK] bundled OpenMP DLLs: {dlls}")


def _finalize_repaired_wheel(wheel_path: Path) -> int:
    print("=== delvewheel show repaired wheel ===")
    subprocess.run(["delvewheel", "show", str(wheel_path)], check=True)
    _verify_bundled_openmp(wheel_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
