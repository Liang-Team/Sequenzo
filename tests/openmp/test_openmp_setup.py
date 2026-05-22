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
    def test_get_conda_prefix_uses_conda_prefix_only(self, tmp_path):
        from sequenzo.openmp_setup import _get_conda_prefix

        conda_prefix = tmp_path / "conda-env"
        conda_prefix.mkdir()
        with mock.patch.dict(
            os.environ,
            {"CONDA_PREFIX": str(conda_prefix), "CONDA_DEFAULT_ENV": "base"},
            clear=False,
        ):
            assert _get_conda_prefix() == str(conda_prefix)

    def test_get_conda_prefix_ignores_env_name(self):
        from sequenzo.openmp_setup import _get_conda_prefix

        env = os.environ.copy()
        env.pop("CONDA_PREFIX", None)
        env["CONDA_DEFAULT_ENV"] = "base"
        with mock.patch.dict(os.environ, env, clear=True):
            assert _get_conda_prefix() is None


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
