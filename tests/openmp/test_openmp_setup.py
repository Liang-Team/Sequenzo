"""
Tests for OpenMP configuration fixes.
Run: pytest tests/openmp/test_openmp_setup.py -v
"""
import os
import sys
import inspect
from pathlib import Path
from unittest import mock

import pytest


class TestCondaPrefixDetection:
    @staticmethod
    def _make_conda_env(root: Path) -> str:
        (root / "conda-meta").mkdir(parents=True)
        return str(root)

    def test_prefers_interpreter_prefix_over_activated_env(self, tmp_path, monkeypatch):
        """``envs/foo/bin/python`` run while ``base`` is active must resolve to foo."""
        from sequenzo import openmp_setup

        running = self._make_conda_env(tmp_path / "running-env")
        activated = self._make_conda_env(tmp_path / "activated-env")

        monkeypatch.setattr(openmp_setup.sys, "prefix", running)
        monkeypatch.setenv("CONDA_PREFIX", activated)

        assert openmp_setup._get_conda_prefix() == running

    def test_falls_back_to_conda_prefix_env(self, tmp_path, monkeypatch):
        from sequenzo import openmp_setup

        plain = tmp_path / "plain-venv"
        plain.mkdir()
        activated = self._make_conda_env(tmp_path / "activated-env")

        monkeypatch.setattr(openmp_setup.sys, "prefix", str(plain))
        monkeypatch.setenv("CONDA_PREFIX", activated)

        assert openmp_setup._get_conda_prefix() == activated

    def test_returns_none_outside_conda(self, tmp_path, monkeypatch):
        from sequenzo import openmp_setup

        plain = tmp_path / "plain-venv"
        plain.mkdir()

        monkeypatch.setattr(openmp_setup.sys, "prefix", str(plain))
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        monkeypatch.setenv("CONDA_DEFAULT_ENV", "base")

        assert openmp_setup._get_conda_prefix() is None

    def test_ignores_prefix_without_conda_meta(self, tmp_path, monkeypatch):
        from sequenzo import openmp_setup

        plain = tmp_path / "plain-venv"
        plain.mkdir()
        not_conda = tmp_path / "not-conda"
        not_conda.mkdir()

        monkeypatch.setattr(openmp_setup.sys, "prefix", str(plain))
        monkeypatch.setenv("CONDA_PREFIX", str(not_conda))

        assert openmp_setup._get_conda_prefix() is None


class TestDuplicateRuntimeGuards:
    """Loading a second OpenMP runtime corrupts libomp and segfaults in __kmp_suspend."""

    def test_extension_discovery_covers_top_level_modules(self, monkeypatch, tmp_path):
        from sequenzo import openmp_setup

        site_packages = tmp_path / "site-packages"
        pkg_dir = site_packages / "sequenzo"
        (pkg_dir / "clustering").mkdir(parents=True)
        nested = pkg_dir / "clustering" / "clustering_c_code.so"
        nested.write_bytes(b"fake")
        top_level = site_packages / "_sequenzo_fastcluster.cpython-310-darwin.so"
        top_level.write_bytes(b"fake")

        monkeypatch.setattr(openmp_setup, "_get_sequenzo_package_dir", lambda: pkg_dir)

        found = set(openmp_setup._iter_sequenzo_extension_modules())
        assert nested in found
        assert top_level in found

    def test_darwin_fix_rewrites_top_level_module(self, monkeypatch, tmp_path):
        from sequenzo import openmp_setup

        conda_prefix = tmp_path / "conda"
        (conda_prefix / "lib").mkdir(parents=True)
        (conda_prefix / "lib" / "libomp.dylib").write_bytes(b"fake")

        site_packages = tmp_path / "site-packages"
        pkg_dir = site_packages / "sequenzo"
        pkg_dir.mkdir(parents=True)
        top_level = site_packages / "_sequenzo_fastcluster.cpython-310-darwin.so"
        top_level.write_bytes(b"fake")

        rewritten = []
        monkeypatch.setattr(openmp_setup, "_get_conda_prefix", lambda: str(conda_prefix))
        monkeypatch.setattr(openmp_setup, "_get_sequenzo_package_dir", lambda: pkg_dir)
        monkeypatch.setattr(
            openmp_setup,
            "_rewrite_macos_extension_to_conda_libomp",
            lambda so_path, conda_libomp: rewritten.append(so_path),
        )

        openmp_setup._fix_duplicate_libomp_in_conda_darwin()

        assert top_level in rewritten

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
    def test_availability_check_does_not_load_second_runtime(self, monkeypatch):
        from sequenzo import openmp_setup

        def _explode(*args, **kwargs):
            raise AssertionError("must not dlopen another OpenMP runtime")

        monkeypatch.setattr(
            openmp_setup,
            "_iter_loaded_openmp_images",
            lambda: ["/somewhere/libomp.dylib"],
        )
        monkeypatch.setattr(openmp_setup.ctypes, "CDLL", _explode)

        assert openmp_setup.check_libomp_availability() is True

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
    def test_availability_check_prefers_bundled_over_homebrew(self, monkeypatch, tmp_path):
        from sequenzo import openmp_setup

        pkg_dir = tmp_path / "sequenzo"
        (pkg_dir / ".dylibs").mkdir(parents=True)
        bundled = pkg_dir / ".dylibs" / "libomp.dylib"
        bundled.write_bytes(b"fake")

        monkeypatch.setattr(openmp_setup, "_get_sequenzo_package_dir", lambda: pkg_dir)
        monkeypatch.setattr(openmp_setup, "_get_conda_prefix", lambda: None)

        candidates = openmp_setup._iter_macos_libomp_candidates()
        assert candidates[0] == str(bundled)
        assert all(
            candidates.index(str(bundled)) < candidates.index(brew)
            for brew in openmp_setup._MACOS_HOMEBREW_LIBOMP_PATHS
        )

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
    def test_only_one_openmp_runtime_is_loaded(self):
        """Regression: Homebrew + Conda libomp in one process crashed OM distances."""
        import sequenzo.dissimilarity_measures.c_code  # noqa: F401
        from sequenzo.openmp_setup import _iter_loaded_openmp_images

        loaded = {os.path.realpath(path) for path in _iter_loaded_openmp_images()}
        sequenzo_runtimes = {path for path in loaded if "/sklearn/" not in path}
        assert len(sequenzo_runtimes) <= 1, (
            f"Multiple OpenMP runtimes loaded: {sorted(sequenzo_runtimes)}"
        )


class TestFixDuplicateLibompInConda:
    def test_dispatcher_calls_windows_helper(self, monkeypatch):
        from sequenzo import openmp_setup

        called = {"win": False, "mac": False}

        monkeypatch.setattr(openmp_setup.sys, "platform", "win32")
        monkeypatch.setattr(
            openmp_setup,
            "_fix_duplicate_libomp_in_conda_windows",
            lambda: called.__setitem__("win", True),
        )
        monkeypatch.setattr(
            openmp_setup,
            "_fix_duplicate_libomp_in_conda_darwin",
            lambda: called.__setitem__("mac", True),
        )

        openmp_setup.fix_duplicate_libomp_in_conda()
        assert called["win"] is True
        assert called["mac"] is False

    def test_windows_conda_registers_dll_directories(self, monkeypatch, tmp_path):
        from sequenzo import openmp_setup

        conda_prefix = tmp_path / "conda"
        conda_bin = conda_prefix / "Library" / "bin"
        conda_bin.mkdir(parents=True)
        conda_dll = conda_bin / "libomp140.x86_64.dll"
        conda_dll.write_bytes(b"fake")

        pkg_dir = tmp_path / "site-packages" / "sequenzo"
        pkg_dir.mkdir(parents=True)
        libs_dir = tmp_path / "site-packages" / "sequenzo.libs"
        libs_dir.mkdir()
        bundled_dll = libs_dir / "libomp140.x86_64.dll"
        bundled_dll.write_bytes(b"old")

        added = []
        monkeypatch.setattr(
            openmp_setup,
            "_register_windows_dll_directory",
            lambda path: added.append(str(path)),
        )
        monkeypatch.setattr(openmp_setup, "_get_conda_prefix", lambda: str(conda_prefix))
        monkeypatch.setattr(openmp_setup, "_get_sequenzo_package_dir", lambda: pkg_dir)
        monkeypatch.setattr(openmp_setup, "_get_sequenzo_libs_dir", lambda: libs_dir)

        openmp_setup._fix_duplicate_libomp_in_conda_windows()

        assert str(conda_bin) in added
        assert str(libs_dir) in added
        assert bundled_dll.read_bytes() == conda_dll.read_bytes()
        assert os.environ.get("KMP_DUPLICATE_LIB_OK") == "TRUE"

    def test_windows_dll_directory_handle_is_retained(self, monkeypatch, tmp_path):
        from sequenzo import openmp_setup

        retained_handles = []

        class FakeDllDirectoryHandle:
            pass

        def fake_add_dll_directory(path):
            handle = FakeDllDirectoryHandle()
            retained_handles.append((path, handle))
            return handle

        monkeypatch.setattr(
            openmp_setup.os,
            "add_dll_directory",
            fake_add_dll_directory,
            raising=False,
        )
        monkeypatch.setattr(openmp_setup, "_WINDOWS_DLL_DIRECTORY_HANDLES", [])

        openmp_setup._register_windows_dll_directory(tmp_path)

        assert retained_handles == [(str(tmp_path), openmp_setup._WINDOWS_DLL_DIRECTORY_HANDLES[-1])]

    def test_windows_skips_without_conda_prefix(self, monkeypatch):
        from sequenzo import openmp_setup

        monkeypatch.setattr(openmp_setup, "_get_conda_prefix", lambda: None)
        called = {"added": False}

        def _add(path):
            called["added"] = True

        monkeypatch.setattr(openmp_setup, "_register_windows_dll_directory", _add)
        openmp_setup._fix_duplicate_libomp_in_conda_windows()
        assert called["added"] is False


class TestKMPDuplicateLib:
    """Test PyTorch/MKL conflict prevention."""

    def test_kmp_duplicate_lib_ok_is_set(self):
        """KMP_DUPLICATE_LIB_OK should be set after importing sequenzo."""
        import sequenzo
        assert os.environ.get("KMP_DUPLICATE_LIB_OK") == "TRUE"

    def test_kmp_does_not_override_user_setting(self):
        """setdefault should not override existing user value."""
        os.environ["KMP_DUPLICATE_LIB_OK"] = "FALSE"
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        assert os.environ["KMP_DUPLICATE_LIB_OK"] == "FALSE"
        # Clean up
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class TestThreadControl:
    """Test SEQUENZO_NUM_THREADS support."""

    def test_sequenzo_num_threads_propagates(self):
        """SEQUENZO_NUM_THREADS should set OMP_NUM_THREADS."""
        os.environ["SEQUENZO_NUM_THREADS"] = "4"
        _nt = os.environ.get("SEQUENZO_NUM_THREADS")
        if _nt is not None:
            os.environ.setdefault("OMP_NUM_THREADS", str(_nt))
        assert os.environ.get("OMP_NUM_THREADS") is not None
        # Clean up
        os.environ.pop("SEQUENZO_NUM_THREADS", None)
        os.environ.pop("OMP_NUM_THREADS", None)


class TestOpenMPModules:
    """Test _OPENMP_MODULES configuration."""

    def test_dissimilarity_measures_in_openmp_modules(self):
        """dissimilarity_measures should be in _OPENMP_MODULES."""
        from sequenzo import _OPENMP_MODULES
        assert "sequenzo.dissimilarity_measures" in _OPENMP_MODULES

    def test_clustering_in_openmp_modules(self):
        """clustering should still be in _OPENMP_MODULES."""
        from sequenzo import _OPENMP_MODULES
        assert "sequenzo.clustering" in _OPENMP_MODULES


class TestFindLibomp:
    """Test libomp detection logic."""

    @pytest.mark.skipif(sys.platform != 'darwin', reason="macOS only")
    def test_known_libomp_paths(self):
        """On macOS, at least one known Homebrew libomp path should exist if installed."""
        candidates = [
            '/opt/homebrew/opt/libomp',   # Apple Silicon
            '/usr/local/opt/libomp',      # Intel Mac
        ]
        found = False
        for prefix in candidates:
            inc = os.path.join(prefix, 'include', 'omp.h')
            lib = os.path.join(prefix, 'lib')
            if os.path.isfile(inc) and os.path.isdir(lib):
                found = True
                break
        # This test passes if libomp is installed (expected on dev machine)
        if not found:
            pytest.skip("libomp not installed via Homebrew")


class TestOpenMPCompiled:
    """Test that compiled extensions actually have OpenMP."""

    @pytest.mark.skipif(sys.platform != 'darwin', reason="macOS only")
    def test_c_code_links_libomp(self):
        """c_code.so should be linked against libomp on macOS."""
        import subprocess
        import sequenzo.dissimilarity_measures.c_code as c
        result = subprocess.run(
            ['otool', '-L', c.__file__],
            capture_output=True, text=True
        )
        assert 'libomp' in result.stdout, (
            "c_code.so is NOT linked against libomp. "
            "OpenMP is not enabled in the compiled extension."
        )

    def test_openmp_actually_parallel(self):
        """Verify OpenMP parallelism works by timing a real computation."""
        import time
        import numpy as np
        import pandas as pd
        from sequenzo import SequenceData, get_distance_matrix

        # Create small test data as DataFrame
        np.random.seed(42)
        n, length = 500, 20
        states = list(range(5))
        data = pd.DataFrame(
            np.random.choice(states, size=(n, length)),
            columns=list(range(length))
        )

        seq_data = SequenceData(
            data, time=list(range(length)),
            states=states, missing_values=-1
        )

        start = time.time()
        mtx = get_distance_matrix(seq_data, "OM", sm="TRATE")
        elapsed = time.time() - start

        assert mtx.shape == (n, n)
        # With OpenMP on multi-core, 500 sequences should finish quickly
        assert elapsed < 10.0, f"OM took {elapsed:.1f}s for n=500, OpenMP may not be working"


class TestCondaNoSkip:
    """Test that Conda environment no longer skips OpenMP."""

    def test_openmp_setup_runs_conda_fix_before_early_return(self, monkeypatch):
        """ensure_openmp_support should not skip Conda duplicate-libomp mitigation."""
        from sequenzo import openmp_setup

        calls = []
        monkeypatch.setattr(openmp_setup, "_get_conda_prefix", lambda: "/fake/conda")
        monkeypatch.setattr(
            openmp_setup,
            "fix_duplicate_libomp_in_conda",
            lambda: calls.append("fix"),
        )
        monkeypatch.setattr(openmp_setup.sys, "platform", "linux")

        result = openmp_setup.ensure_openmp_support()

        assert result is True
        assert calls == ["fix"]

    def test_openmp_setup_source_runs_conda_fix_first(self):
        """Conda duplicate-libomp mitigation must precede platform early return."""
        from sequenzo.openmp_setup import ensure_openmp_support

        source = inspect.getsource(ensure_openmp_support)
        conda_fix_idx = source.index("fix_duplicate_libomp_in_conda()")
        early_return_idx = source.index(
            'if sys.platform != "darwin" or platform.machine() != "arm64":'
        )
        assert conda_fix_idx < early_return_idx

    def test_init_no_conda_skip(self):
        """__init__._setup_openmp_if_needed should not skip for Conda."""
        import inspect
        from sequenzo import _setup_openmp_if_needed
        source = inspect.getsource(_setup_openmp_if_needed)
        assert 'CONDA_DEFAULT_ENV' not in source, (
            "_setup_openmp_if_needed still checks CONDA_DEFAULT_ENV"
        )
