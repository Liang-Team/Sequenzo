import subprocess
import sys
import textwrap
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


NATIVE_IMPORT_TARGETS = [
    "_sequenzo_fastcluster",
    "sequenzo.big_data.clara.utils.get_weighted_diss",
    "sequenzo.clustering.clustering_c_code",
    "sequenzo.dissimilarity_measures.c_code",
    "sequenzo.dissimilarity_measures.utils.get_sm_trate_substitution_cost_matrix",
    "sequenzo.dissimilarity_measures.utils.seqconc",
    "sequenzo.dissimilarity_measures.utils.seqdss",
    "sequenzo.dissimilarity_measures.utils.seqdur",
    "sequenzo.dissimilarity_measures.utils.seqlength",
    "sequenzo.utils.core_distance_operations.core_distance_c_code",
]


def _run_import_script(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_public_import_contract_does_not_eagerly_load_native_extensions():
    targets = repr(NATIVE_IMPORT_TARGETS)
    result = _run_import_script(
        f"""
        import sys

        from sequenzo import SequenceData, get_distance_matrix, Cluster

        targets = {targets}
        loaded = [name for name in targets if name in sys.modules]
        if loaded:
            raise SystemExit("eager native imports: " + ", ".join(loaded))
        """
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_public_import_contract_survives_native_extension_load_failures():
    targets = repr(set(NATIVE_IMPORT_TARGETS))
    result = _run_import_script(
        f"""
        import importlib.abc
        import sys

        targets = {targets}

        class NativeImportBlocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in targets:
                    raise ImportError(
                        "DLL load failed while importing " + fullname + ": simulated missing DLL"
                    )
                return None

        sys.meta_path.insert(0, NativeImportBlocker())

        from sequenzo import SequenceData, get_distance_matrix, Cluster
        """
    )

    assert result.returncode == 0, result.stdout + result.stderr
