#!/usr/bin/env python3
"""
@Author  : Yuqi Liang 梁彧祺，Yapeng Wei 卫亚鹏，Xinyi Li 李欣怡
@File    : openmp_setup.py
@Time    : 07/10/2025 10:42
@Desc    :

OpenMP setup and duplicate-runtime mitigation for Sequenzo.

On Apple Silicon macOS, this module can install/configure Homebrew libomp.
On macOS and Windows Conda environments, it also tries to prevent loading
multiple incompatible OpenMP runtimes in the same Python process.
"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Windows wheels repaired with delvewheel typically bundle MSVC LLVM OpenMP here.
_WINDOWS_BUNDLED_OMP_NAMES = (
    "libomp140.x86_64.dll",
    "libomp140.dll",
)

# Conda / MKL / llvm-openmp runtimes we may want to prefer over a second copy.
_CONDA_OMP_DLL_NAMES = (
    "libomp140.x86_64.dll",
    "libomp140.dll",
    "libomp.dll",
    "libiomp5md.dll",
    "libiomp5.dll",
)

_MACOS_HOMEBREW_LIBOMP_PATHS = (
    "/opt/homebrew/opt/libomp/lib/libomp.dylib",
    "/opt/homebrew/lib/libomp.dylib",
    "/usr/local/opt/libomp/lib/libomp.dylib",
    "/usr/local/lib/libomp.dylib",
)


def _get_conda_prefix() -> str | None:
    """Return the active Conda environment prefix, if any."""
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix and os.path.isdir(prefix):
        return prefix
    return None


def _get_sequenzo_package_dir() -> Path:
    return Path(__file__).resolve().parent


def _get_sequenzo_libs_dir() -> Path:
    return _get_sequenzo_package_dir().parent / "sequenzo.libs"


def _iter_windows_wheel_openmp_dirs() -> list[Path]:
    """Directories where delvewheel may vendor OpenMP DLLs on Windows."""
    site_packages = _get_sequenzo_package_dir().parent
    candidates = [
        site_packages / "sequenzo.libs",
        site_packages,
    ]
    return [path for path in candidates if path.is_dir()]


def _find_windows_bundled_openmp_dlls(directory: Path) -> dict[str, Path]:
    found = _find_dlls(directory, _WINDOWS_BUNDLED_OMP_NAMES)
    if found:
        return found

    for path in directory.glob("libomp140*.dll"):
        if path.is_file():
            found[path.name.lower()] = path
    return found


def _iter_conda_openmp_dirs(conda_prefix: str) -> list[Path]:
    candidates = [
        Path(conda_prefix) / "Library" / "bin",
        Path(conda_prefix) / "lib",
        Path(conda_prefix) / "bin",
    ]
    return [path for path in candidates if path.is_dir()]


def _find_dlls(directory: Path, names: tuple[str, ...]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for name in names:
        path = directory / name
        if path.is_file():
            found[name.lower()] = path
    return found


def check_libomp_availability():
    """
    Check if libomp is available on the system.

    Returns:
        bool: True if libomp is available, False otherwise
    """
    if sys.platform == "darwin":
        try:
            ctypes.CDLL("libomp.dylib")
            return True
        except OSError:
            pass

        for path in ("/opt/homebrew/lib/libomp.dylib", "/usr/local/lib/libomp.dylib"):
            if os.path.exists(path):
                try:
                    ctypes.CDLL(path)
                    return True
                except OSError:
                    continue
        return False

    if sys.platform == "win32":
        for directory in _iter_conda_openmp_dirs(os.environ.get("CONDA_PREFIX", "")):
            if _find_dlls(directory, _CONDA_OMP_DLL_NAMES):
                return True
        for directory in _iter_windows_wheel_openmp_dirs():
            if _find_windows_bundled_openmp_dlls(directory):
                return True
        return False

    return True


def check_homebrew_available():
    """
    Check if Homebrew is available on the system.

    Returns:
        bool: True if Homebrew is available, False otherwise
    """
    try:
        subprocess.run(
            ["brew", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_libomp_via_homebrew():
    """
    Install libomp via Homebrew.

    Returns:
        bool: True if installation successful, False otherwise
    """
    try:
        print("🔧 Installing libomp via Homebrew...")
        subprocess.run(
            ["brew", "install", "libomp"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("[>] libomp installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[>] libomp installation failed: {e}")
        return False
    except Exception as e:
        print(f"[>] Error during installation: {e}")
        return False


def setup_openmp_environment():
    """
    Set up OpenMP environment variables for Apple Silicon.

    Returns:
        bool: True if setup successful, False otherwise
    """
    try:
        result = subprocess.run(
            ["brew", "--prefix"],
            capture_output=True,
            text=True,
            check=True,
        )
        homebrew_prefix = result.stdout.strip()

        lib_path = f"{homebrew_prefix}/lib"
        include_path = f"{homebrew_prefix}/include"

        os.environ["DYLD_LIBRARY_PATH"] = f"{lib_path}:{os.environ.get('DYLD_LIBRARY_PATH', '')}"
        os.environ["LDFLAGS"] = f"-L{lib_path} {os.environ.get('LDFLAGS', '')}"
        os.environ["CPPFLAGS"] = f"-I{include_path} {os.environ.get('CPPFLAGS', '')}"

        print("[>] OpenMP environment variables set")
        print(f"   - Library path: {lib_path}")
        print(f"   - Include path: {include_path}")
        return True

    except Exception as e:
        print(f"[>] Failed to set environment variables: {e}")
        return False


def _rewrite_macos_extension_to_conda_libomp(so_path: Path, conda_libomp: Path) -> bool:
    try:
        result = subprocess.run(
            ["otool", "-L", str(so_path)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        for brew_path in _MACOS_HOMEBREW_LIBOMP_PATHS:
            if brew_path not in result.stdout:
                continue
            subprocess.run(
                [
                    "install_name_tool",
                    "-change",
                    brew_path,
                    str(conda_libomp),
                    str(so_path),
                ],
                check=True,
                capture_output=True,
                timeout=10,
            )
            return True
    except Exception:
        return False
    return False


def _fix_duplicate_libomp_in_conda_darwin() -> None:
    """
    Fix duplicate libomp loading in Conda environments on macOS.

    When a .so extension module is linked to Homebrew's libomp but the active
    Conda environment also ships its own libomp, both copies can be loaded in
    the same process and cause memory corruption / segfaults.
    """
    conda_prefix = _get_conda_prefix()
    if conda_prefix is None:
        return

    conda_libomp = Path(conda_prefix) / "lib" / "libomp.dylib"
    if not conda_libomp.is_file():
        return

    pkg_dir = _get_sequenzo_package_dir()
    for so_path in pkg_dir.rglob("*.so"):
        _rewrite_macos_extension_to_conda_libomp(so_path, conda_libomp)


def _register_windows_dll_directory(path: Path) -> None:
    if not hasattr(os, "add_dll_directory"):
        return
    try:
        os.add_dll_directory(str(path))
    except OSError:
        pass


def _replace_bundled_openmp_with_conda_copy(
    bundled_path: Path,
    conda_path: Path,
) -> bool:
    try:
        if bundled_path.resolve() == conda_path.resolve():
            return False
        shutil.copy2(conda_path, bundled_path)
        return True
    except OSError:
        return False


def _fix_duplicate_libomp_in_conda_windows() -> None:
    """
    Mitigate duplicate OpenMP runtimes in Conda environments on Windows.

    Typical conflict:
    - Sequenzo wheel bundles ``libomp140*.dll`` in ``sequenzo.libs/``
    - NumPy / SciPy / MKL bring ``libiomp5md.dll`` from their own wheels
    - Conda may ship additional OpenMP DLLs under ``Library/bin``

    Strategy:
    1. Allow Intel OpenMP duplicates when unavoidable (MKL + other stacks).
    2. Prefer Conda's DLL directories in the Windows loader search path.
    3. When possible, replace Sequenzo's bundled MSVC OpenMP DLL with the
       copy from the active Conda environment so only one file backs the
       Sequenzo extensions.
    """
    conda_prefix = _get_conda_prefix()
    if conda_prefix is None:
        return

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    conda_dirs = _iter_conda_openmp_dirs(conda_prefix)
    conda_dlls: dict[str, Path] = {}
    for directory in conda_dirs:
        conda_dlls.update(_find_dlls(directory, _CONDA_OMP_DLL_NAMES))

    for directory in conda_dirs:
        _register_windows_dll_directory(directory)

    for directory in _iter_windows_wheel_openmp_dirs():
        if _find_windows_bundled_openmp_dlls(directory):
            _register_windows_dll_directory(directory)

    bundled_dlls: dict[str, Path] = {}
    for directory in _iter_windows_wheel_openmp_dirs():
        bundled_dlls.update(_find_windows_bundled_openmp_dlls(directory))

    if not bundled_dlls:
        return

    for bundled_name in _WINDOWS_BUNDLED_OMP_NAMES:
        bundled_path = bundled_dlls.get(bundled_name.lower())
        if bundled_path is None:
            continue

        conda_match = conda_dlls.get(bundled_name.lower())
        if conda_match is None:
            continue

        if _replace_bundled_openmp_with_conda_copy(bundled_path, conda_match):
            print(
                "[>] Aligned Sequenzo bundled OpenMP with Conda copy: "
                f"{conda_match.name}"
            )
        break


def fix_duplicate_libomp_in_conda() -> None:
    """
    Best-effort mitigation for duplicate OpenMP runtimes in Conda environments.

    - macOS: rewrite Sequenzo extension references from Homebrew libomp to the
      Conda environment's libomp.
    - Windows: register Conda DLL directories, set Intel duplicate tolerance,
      and align bundled ``sequenzo.libs`` OpenMP with Conda when available.

    The function is idempotent and never raises.
    """
    try:
        if sys.platform == "darwin":
            _fix_duplicate_libomp_in_conda_darwin()
        elif sys.platform == "win32":
            _fix_duplicate_libomp_in_conda_windows()
    except Exception:
        pass


def ensure_openmp_support():
    """
    Ensure OpenMP support is available on Apple Silicon Macs.
    This function handles the complete setup process.

    Returns:
        bool: True if OpenMP is available, False otherwise
    """
    if _get_conda_prefix() is not None:
        fix_duplicate_libomp_in_conda()

    if sys.platform != "darwin" or platform.machine() != "arm64":
        return True

    if os.environ.get("CONDA_DEFAULT_ENV"):
        print("[>] Detected Conda environment, checking OpenMP support...")

    print("[>] Detected Apple Silicon Mac, checking OpenMP support...")

    if check_libomp_availability():
        print("[>] OpenMP support is available")
        return True

    if not check_homebrew_available():
        print("""
[>] OpenMP Dependency Detection

On Apple Silicon Mac, Sequenzo requires OpenMP support for parallel computation.

Please run the following command to install OpenMP support:
    brew install libomp

If you don't have Homebrew installed, please visit https://brew.sh to install Homebrew first.
        """)
        return False

    try:
        subprocess.run(
            ["brew", "list", "libomp"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        print("[>] libomp is already installed via Homebrew")
        setup_openmp_environment()
        return True
    except subprocess.CalledProcessError:
        pass

    if install_libomp_via_homebrew():
        setup_openmp_environment()
        return True

    print("""
[>] Automatic OpenMP installation failed

Please manually run the following command:
    brew install libomp

After installation, please restart Python or re-import sequenzo.
    """)
    return False


def get_openmp_status():
    """
    Get the current OpenMP status and provide helpful information.

    Returns:
        dict: Status information about OpenMP support
    """
    return {
        "platform": sys.platform,
        "architecture": platform.machine(),
        "is_apple_silicon": sys.platform == "darwin" and platform.machine() == "arm64",
        "libomp_available": check_libomp_availability(),
        "homebrew_available": check_homebrew_available(),
        "conda_environment": _get_conda_prefix() is not None,
        "conda_prefix": _get_conda_prefix(),
        "sequenzo_libs_dir": str(_get_sequenzo_libs_dir()),
    }


if __name__ == "__main__":
    fix_duplicate_libomp_in_conda()
    success = ensure_openmp_support()
    if success:
        print("[>] OpenMP support is ready!")
    else:
        print("[>] OpenMP support unavailable, will use serial computation")

    status = get_openmp_status()
    print("\n[>] System Status:")
    print(f"   - Platform: {status['platform']}")
    print(f"   - Architecture: {status['architecture']}")
    print(f"   - Apple Silicon: {status['is_apple_silicon']}")
    print(f"   - libomp available: {status['libomp_available']}")
    print(f"   - Homebrew available: {status['homebrew_available']}")
    print(f"   - Conda environment: {status['conda_environment']}")
    print(f"   - Conda prefix: {status['conda_prefix']}")
    print(f"   - Sequenzo libs dir: {status['sequenzo_libs_dir']}")
