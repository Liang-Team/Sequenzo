#!/usr/bin/env python3
"""
Windows wheel repair helper for cibuildwheel.

Responsibilities:
1. Locate the MSVC LLVM OpenMP DLL (libomp140.x86_64.dll) via vswhere or
   well-known fallback paths on GitHub-hosted runners.
2. Run `python -m delvewheel repair --add-path <dir>` to bundle it into the wheel.
3. If delvewheel does not produce a repaired wheel (DLL not found, or the
   wheel has no external dependencies), copy the built wheel as-is so that
   cibuildwheel always finds something in dest_dir.

Usage (called from CIBW_REPAIR_WHEEL_COMMAND_WINDOWS, cwd = project root):
    python maintenance_scripts/repair_windows_wheel.py <dest_dir> <wheel>
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

_MSVC_OMP_DLL_NAMES = (
    "libomp140.x86_64.dll",
    "libomp140.dll",
)


# ---------------------------------------------------------------------------
# delvewheel invocation
# ---------------------------------------------------------------------------

def _ensure_delvewheel() -> None:
    """Install delvewheel into the active cibuildwheel venv when missing."""
    if importlib.util.find_spec("delvewheel") is not None:
        return

    print("[repair] delvewheel not found in build venv; installing...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "delvewheel"],
        check=False,
    )
    if result.returncode != 0 or importlib.util.find_spec("delvewheel") is None:
        print("[repair] ERROR: failed to install delvewheel", file=sys.stderr)
        raise SystemExit(1)


def _delvewheel_cmd(*args: str) -> list[str]:
    """Invoke delvewheel via the active Python (Scripts/ may be off PATH)."""
    return [sys.executable, "-m", "delvewheel", *args]


def _run_delvewheel(args: list[str]) -> subprocess.CompletedProcess | None:
    cmd = _delvewheel_cmd(*args)
    print(f"[repair] Running: {' '.join(cmd)}")
    try:
        return subprocess.run(cmd, check=False)
    except FileNotFoundError as exc:
        print(f"[repair] delvewheel launch failed: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# DLL discovery
# ---------------------------------------------------------------------------

def _find_vswhere() -> Path | None:
    """Return the path to vswhere.exe, or None if not found."""
    candidates = []

    pf86 = os.environ.get("ProgramFiles(x86)") or os.environ.get("PROGRAMFILES(X86)")
    if pf86:
        candidates.append(Path(pf86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe")

    pf86b = os.environ.get("PROGRAMFILES_X86")
    if pf86b:
        candidates.append(Path(pf86b) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe")

    candidates.append(Path("C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe"))

    for p in candidates:
        if p.is_file():
            return p
    return None


def _find_omp_dll_via_vswhere(vswhere: Path) -> Path | None:
    """Locate MSVC libomp140*.dll under the active Visual Studio install."""
    for pattern in (
        r"VC\Tools\Llvm\x64\bin\libomp140.x86_64.dll",
        r"VC\Tools\Llvm\x64\bin\libomp140.dll",
    ):
        try:
            result = subprocess.run(
                [
                    str(vswhere),
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-find",
                    pattern,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception as exc:
            print(f"[repair] vswhere failed for {pattern}: {exc}", file=sys.stderr)
            continue

        for line in result.stdout.splitlines():
            p = Path(line.strip())
            if p.is_file() and p.name.lower() in _MSVC_OMP_DLL_NAMES:
                print(f"[repair] Found DLL via vswhere: {p}")
                return p

    try:
        result = subprocess.run(
            [str(vswhere), "-latest", "-products", "*", "-property", "installationPath"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        install_root = Path(result.stdout.strip())
    except Exception as exc:
        print(f"[repair] vswhere installationPath lookup failed: {exc}", file=sys.stderr)
        return None

    if install_root.is_dir():
        for name in _MSVC_OMP_DLL_NAMES:
            matches = sorted(install_root.glob(f"VC/Tools/Llvm/**/bin/{name}"))
            if matches:
                print(f"[repair] Found DLL under VS install root: {matches[0]}")
                return matches[0]

        redist_match = _find_omp_dll_via_redist(install_root)
        if redist_match:
            return redist_match

    return None


def _pick_best_redist_match(matches: list[Path]) -> Path:
    """Prefer release redist OpenMP DLLs over debug_nonredist copies."""
    release = [
        path
        for path in matches
        if "debug_nonredist" not in path.as_posix().casefold()
    ]
    pool = release or matches
    return sorted(pool)[-1]


def _find_omp_dll_via_redist(install_root: Path | None = None) -> Path | None:
    """Search VC Redist trees where MSVC ships libomp140 for /openmp:llvm."""
    roots: list[Path] = []
    if install_root is not None and install_root.is_dir():
        roots.append(install_root)
    roots.extend(
        [
            Path("C:/Program Files/Microsoft Visual Studio/2022/Enterprise"),
            Path("C:/Program Files/Microsoft Visual Studio/2022/Community"),
            Path("C:/Program Files/Microsoft Visual Studio/2022/Professional"),
        ]
    )

    seen: set[Path] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)

        for name in _MSVC_OMP_DLL_NAMES:
            for pattern in (
                f"VC/Redist/MSVC/*/x64/Microsoft.VC*.OpenMP.LLVM/{name}",
                f"VC/Redist/MSVC/*/debug_nonredist/x64/Microsoft.VC*.OpenMP.LLVM/{name}",
            ):
                matches = sorted(resolved.glob(pattern))
                if matches:
                    best = _pick_best_redist_match(matches)
                    print(f"[repair] Found DLL via VS redist: {best}")
                    return best

    return None


def _find_omp_dll_from_env() -> Path | None:
    """Use LIBOMP_DLL_DIR when CI has already resolved the OpenMP runtime."""
    env_dir = os.environ.get("LIBOMP_DLL_DIR", "").strip()
    if not env_dir:
        return None

    directory = Path(env_dir)
    for name in _MSVC_OMP_DLL_NAMES:
        candidate = directory / name
        if candidate.is_file():
            print(f"[repair] Found DLL via LIBOMP_DLL_DIR: {candidate}")
            return candidate
    return None


def _find_omp_dll_fallback() -> Path | None:
    """Walk well-known MSVC LLVM OpenMP paths on GitHub-hosted runners."""
    candidates = [
        "C:/Program Files/Microsoft Visual Studio/2022/Enterprise/VC/Tools/Llvm/x64/bin/libomp140.x86_64.dll",
        "C:/Program Files/Microsoft Visual Studio/2022/Enterprise/VC/Tools/Llvm/x64/bin/libomp140.dll",
        "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/Llvm/x64/bin/libomp140.x86_64.dll",
        "C:/Program Files/Microsoft Visual Studio/2022/Professional/VC/Tools/Llvm/x64/bin/libomp140.x86_64.dll",
        "C:/Program Files/LLVM/bin/libomp140.x86_64.dll",
    ]
    for c in candidates:
        p = Path(c)
        if p.is_file():
            print(f"[repair] Found DLL via fallback path: {p}")
            return p
    return None


def find_omp_dll() -> Path | None:
    dll = _find_omp_dll_from_env()
    if dll:
        return dll

    vswhere = _find_vswhere()
    if vswhere:
        print(f"[repair] Using vswhere: {vswhere}")
        dll = _find_omp_dll_via_vswhere(vswhere)
        if dll:
            return dll
        print("[repair] vswhere did not find libomp140*.dll; trying fallback paths...")
    else:
        print("[repair] vswhere.exe not found; trying fallback paths...")

    dll = _find_omp_dll_via_redist()
    if dll:
        return dll

    return _find_omp_dll_fallback()


# ---------------------------------------------------------------------------
# Wheel repair
# ---------------------------------------------------------------------------

def run_delvewheel(
    dest: Path,
    wheel: Path,
    omp_bin: Path | None,
    omp_dll: Path | None,
) -> bool:
    """Run delvewheel; return True if it produced at least one wheel."""
    for existing in dest.glob("*.whl"):
        existing.unlink()

    args = ["repair", "-w", str(dest)]
    if omp_bin:
        args += ["--add-path", str(omp_bin)]
    if omp_dll:
        # --add-path only helps dependency discovery; OpenMP is often loaded
        # indirectly, so force-vendor the runtime into sequenzo.libs/.
        args += ["--include", omp_dll.name]
    args.append(str(wheel))

    result = _run_delvewheel(args)
    if result is not None and result.returncode != 0:
        print(f"[repair] delvewheel exited with code {result.returncode} (continuing...)")

    return bool(sorted(dest.glob("*.whl")))


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: repair_windows_wheel.py <dest_dir> <wheel>", file=sys.stderr)
        return 2

    _ensure_delvewheel()

    dest = Path(sys.argv[1]).resolve()
    wheel = Path(sys.argv[2]).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    if not wheel.is_file():
        print(f"[repair] Source wheel not found: {wheel}", file=sys.stderr)
        return 1

    omp_dll = find_omp_dll()
    omp_bin = omp_dll.parent if omp_dll else None
    if omp_bin:
        print(f"[repair] Bundling OpenMP from: {omp_bin}")
    else:
        print("[repair] WARNING: libomp140*.dll not found; repair may miss OpenMP runtime")

    if run_delvewheel(dest, wheel, omp_bin, omp_dll):
        repaired = sorted(dest.glob("*.whl"))[0]
        print(f"[repair] Repaired wheel (delvewheel): {repaired}")
        return _finalize_repaired_wheel(repaired)

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
            if name.endswith(".dll")
            and "libomp140" in name.lower()
            and ("/" in name or "\\" in name)
            and (
                name.startswith("sequenzo.libs/")
                or ".libs/" in name.replace("\\", "/")
            )
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
    result = _run_delvewheel(["show", str(wheel_path)])
    if result is None or result.returncode != 0:
        print("[repair] ERROR: delvewheel show failed", file=sys.stderr)
        return 1
    _verify_bundled_openmp(wheel_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
