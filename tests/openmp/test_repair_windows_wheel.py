import importlib.util
import sys
from pathlib import Path


def _load_repair_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "maintenance_scripts" / "repair_windows_wheel.py"
    spec = importlib.util.spec_from_file_location("repair_windows_wheel", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_find_omp_dll_prefers_libomp_dll_dir(monkeypatch, tmp_path):
    repair = _load_repair_module()
    dll_dir = tmp_path / "openmp"
    dll_dir.mkdir()
    dll = dll_dir / "libomp140.x86_64.dll"
    dll.write_bytes(b"fake")

    monkeypatch.setenv("LIBOMP_DLL_DIR", str(dll_dir))
    assert repair.find_omp_dll() == dll


def test_find_omp_dll_via_redist_glob(tmp_path):
    repair = _load_repair_module()
    redist_dir = (
        tmp_path
        / "VC"
        / "Redist"
        / "MSVC"
        / "14.44.35207"
        / "x64"
        / "Microsoft.VC143.OpenMP.LLVM"
    )
    redist_dir.mkdir(parents=True)
    dll = redist_dir / "libomp140.x86_64.dll"
    dll.write_bytes(b"fake")

    assert repair._find_omp_dll_via_redist(tmp_path) == dll


def test_pick_best_redist_match_prefers_release_over_debug(tmp_path):
    repair = _load_repair_module()
    release = (
        tmp_path
        / "VC/Redist/MSVC/14.44.35207/x64/Microsoft.VC143.OpenMP.LLVM/libomp140.x86_64.dll"
    )
    debug = (
        tmp_path
        / "VC/Redist/MSVC/14.29.30133/debug_nonredist/x64/Microsoft.VC142.OpenMP.LLVM/libomp140.x86_64.dll"
    )
    release.parent.mkdir(parents=True)
    debug.parent.mkdir(parents=True)
    release.write_bytes(b"release")
    debug.write_bytes(b"debug")

    assert repair._pick_best_redist_match([debug, release]) == release
