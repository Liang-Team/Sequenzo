import importlib
import json
import os
import subprocess
import sys
from unittest import mock

import numpy as np

from sequenzo.clustering import KMedoids


def _distance_matrix(points):
    points = np.asarray(points, dtype=np.float64)
    return np.abs(points[:, None] - points[None, :])


def test_pamonce_core_rejects_mismatched_condensed_length():
    core = importlib.import_module("sequenzo.clustering.clustering_c_code")

    with np.testing.assert_raises_regex(ValueError, "nelements"):
        core.PAMonce(
            4,
            np.zeros(4, dtype=np.float64),
            np.arange(2, dtype=np.int32),
            1,
            np.empty(0, dtype=np.float64),
        )


def test_pamonce_public_api_returns_bounded_workspace_diagnostics():
    result, diagnostics = KMedoids(
        diss=_distance_matrix([0.0, 1.0, 10.0, 11.0]),
        k=2,
        initialclust=[0, 2],
        npass=0,
        method="PAMonce",
        threads=1,
        memory_budget_mb=1,
        return_diagnostics=True,
        verbose=False,
    )

    np.testing.assert_array_equal(result, np.array([1, 1, 3, 3]))
    assert diagnostics["worker_count"] == 1
    assert diagnostics["candidate_storage_bytes"] == 0
    assert diagnostics["workspace_peak_bytes"] <= 1024 * 1024
    assert diagnostics["thread_workspace_bytes"] <= 1024 * 1024
    assert diagnostics["swap_rounds"] >= 1
    assert diagnostics["execution_path"] in {
        "fastpam",
        "reynolds",
        "reynolds_small_k",
        "fused_exact_k2_shared_ties",
        "mixed",
    }


def test_pam_methods_reject_a_thread_workspace_budget_below_one_worker():
    for method in ("PAM", "PAMonce"):
        with np.testing.assert_raises_regex(ValueError, "thread workspace"):
            KMedoids(
                diss=_distance_matrix(np.arange(100, dtype=float)),
                k=10,
                method=method,
                threads=8,
                memory_budget_mb=0.00001,
                verbose=False,
            )


def test_pamonce_budget_includes_the_bounded_candidate_buffer():
    with np.testing.assert_raises_regex(ValueError, "thread workspace"):
        KMedoids(
            diss=_distance_matrix(np.arange(100, dtype=float)),
            k=10,
            method="PAMonce",
            threads=8,
            memory_budget_mb=0.001,
            verbose=False,
        )


def test_importing_clustering_does_not_run_the_openmp_installer():
    code = r'''
import json
import sequenzo.openmp_setup as openmp_setup

calls = []
openmp_setup.check_libomp_availability = lambda: False
openmp_setup.ensure_openmp_support = lambda: calls.append("called")

import sequenzo.clustering
print(json.dumps(calls))
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.getcwd(),
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout.strip()) == []


def test_npass_executes_build_then_seeded_random_restarts(monkeypatch):
    module = importlib.import_module("sequenzo.clustering.k_medoids")
    calls = []

    class CapturePamonce:
        def __init__(self, n, diss, centroids, npass, weights, *options):
            calls.append((centroids.copy(), npass))
            self.n = n
            self.centroids = centroids.copy()

        def runclusterloop_one_based(self):
            return np.full(self.n, self.centroids[0] + 1, dtype=np.int32)

        def diagnostics(self):
            return {"worker_count": 1}

    monkeypatch.setattr(module.clustering_c_code, "PAMonce", CapturePamonce)
    module.KMedoids(
        diss=_distance_matrix(np.arange(8, dtype=float)),
        k=2,
        npass=3,
        method="PAMonce",
        random_state=17,
        verbose=False,
    )

    assert len(calls) == 3
    assert calls[0][1] == 1
    assert calls[1][1] == calls[2][1] == 0
    assert not np.array_equal(calls[1][0], calls[2][0])


def test_pamonce_updates_nearest_medoids_incrementally_after_swaps():
    _, diagnostics = KMedoids(
        diss=_distance_matrix([0.0, 1.0, 10.0, 11.0]),
        k=2,
        initialclust=[0, 1],
        npass=0,
        method="PAMonce",
        threads=1,
        return_diagnostics=True,
        verbose=False,
    )

    assert diagnostics["accepted_swaps"] >= 1
    assert diagnostics["full_assignment_rounds"] == 1
    assert diagnostics["incremental_update_rounds"] == diagnostics["accepted_swaps"]


def test_candidate_overflow_uses_exact_second_pass_instead_of_full_fallback():
    rng = np.random.default_rng(42)
    templates = rng.integers(0, 8, size=(5, 20))
    sequences = np.repeat(templates, 20, axis=0)
    mismatches = np.count_nonzero(
        sequences[:, None, :] != sequences[None, :, :],
        axis=2,
    )
    distance = np.sqrt(2.0 * mismatches)
    options = dict(
        diss=distance,
        k=5,
        npass=1,
        threads=1,
        return_diagnostics=True,
        verbose=False,
    )

    pam_result, pam_diagnostics = KMedoids(method="PAM", **options)
    once_result, once_diagnostics = KMedoids(method="PAMonce", **options)

    assert once_diagnostics["screened_candidate_highwater"] > 64
    assert once_diagnostics["two_pass_recovery_rounds"] >= 1
    assert once_diagnostics["adaptive_fallback_rounds"] == 0
    assert once_diagnostics["swap_trace"] == pam_diagnostics["swap_trace"]
    np.testing.assert_array_equal(once_result, pam_result)


def test_nonnegative_weights_use_sign_aware_verified_screen():
    rng = np.random.default_rng(1)
    points = rng.normal(size=(40, 3))
    distance = np.sqrt(
        ((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)
    )
    options = dict(
        diss=distance,
        k=5,
        npass=1,
        threads=1,
        return_diagnostics=True,
        verbose=False,
    )

    pam_result, pam_diagnostics = KMedoids(method="PAM", **options)
    once_result, once_diagnostics = KMedoids(method="PAMonce", **options)

    assert once_diagnostics["sign_aware_verified_rounds"] == (
        once_diagnostics["swap_rounds"]
    )
    assert once_diagnostics["swap_trace"] == pam_diagnostics["swap_trace"]
    np.testing.assert_array_equal(once_result, pam_result)


def test_two_medoids_use_fused_exact_path():
    distance = _distance_matrix([0.0, 1.0, 4.0, 10.0, 11.0, 15.0])
    options = dict(
        diss=distance,
        k=2,
        npass=1,
        threads=1,
        return_diagnostics=True,
        verbose=False,
    )

    pam_result, pam_diagnostics = KMedoids(method="PAM", **options)
    once_result, once_diagnostics = KMedoids(method="PAMonce", **options)

    assert once_diagnostics["execution_path"] == "fused_exact_k2_shared_ties"
    assert once_diagnostics["small_k_fused_rounds"] >= 1
    assert once_diagnostics["small_k_fused_seconds"] >= 0.0
    assert once_diagnostics["small_k_reynolds_rounds"] == 0
    assert once_diagnostics["fast_score_evaluations"] == 0
    assert once_diagnostics["swap_trace"] == pam_diagnostics["swap_trace"]
    np.testing.assert_array_equal(once_result, pam_result)


def test_pamonce_matches_classic_pam_swap_trace():
    distance = _distance_matrix([0.0, 1.0, 4.0, 10.0, 11.0, 15.0])
    options = dict(
        diss=distance,
        k=2,
        initialclust=[0, 1],
        npass=0,
        threads=1,
        return_diagnostics=True,
        verbose=False,
    )

    pam_result, pam_diagnostics = KMedoids(method="PAM", **options)
    once_result, once_diagnostics = KMedoids(method="PAMonce", **options)

    np.testing.assert_array_equal(once_result, pam_result)
    assert once_diagnostics["swap_trace"] == pam_diagnostics["swap_trace"]


def test_explicit_integer_distances_use_exact_fast_scores_without_fallback():
    distance = np.ones((12, 12), dtype=np.float64)
    np.fill_diagonal(distance, 0.0)

    _, diagnostics = KMedoids(
        diss=distance,
        k=4,
        initialclust=[0, 1, 2, 3],
        npass=0,
        method="PAMonce",
        threads=1,
        return_diagnostics=True,
        verbose=False,
    )

    assert diagnostics["exact_integer_rounds"] == diagnostics["swap_rounds"]
    assert diagnostics["exact_fixed_point_rounds"] == 0
    assert diagnostics["classic_score_evaluations"] == 0


def test_flat_fractional_distances_have_zero_width_score_intervals():
    distance = np.full((12, 12), 0.5, dtype=np.float64)
    np.fill_diagonal(distance, 0.0)

    _, diagnostics = KMedoids(
        diss=distance,
        k=4,
        initialclust=[0, 1, 2, 3],
        npass=0,
        method="PAMonce",
        threads=1,
        return_diagnostics=True,
        verbose=False,
    )

    assert diagnostics["classic_score_evaluations"] == 0
    assert diagnostics["adaptive_fallback_rounds"] == 0


def test_public_pamonce_uses_core_objective_without_python_rescan(monkeypatch):
    module = importlib.import_module("sequenzo.clustering.k_medoids")
    objective_scan = mock.Mock(wraps=module._pam_objective)
    monkeypatch.setattr(module, "_pam_objective", objective_scan)
    result, diagnostics = module.KMedoids(
        diss=_distance_matrix([0.0, 1.0, 4.0, 10.0, 11.0, 15.0]),
        k=2,
        method="PAMonce",
        threads=1,
        return_diagnostics=True,
        verbose=False,
    )

    assert objective_scan.call_count == 0
    assert len(result) == 6
    assert diagnostics["objective"] >= 0.0


def test_public_api_skips_diagnostics_when_they_are_not_requested(monkeypatch):
    module = importlib.import_module("sequenzo.clustering.k_medoids")
    collection_modes = []

    class NoDiagnosticsPamonce:
        def __init__(self, n, diss, centroids, npass, weights, *options):
            self.n = n

        def set_collect_diagnostics(self, enabled):
            collection_modes.append(enabled)

        def runclusterloop_one_based(self):
            return np.ones(self.n, dtype=np.int32)

        def objective(self):
            return 0.0

        diagnostics = mock.Mock(return_value={})

    monkeypatch.setattr(
        module.clustering_c_code, "PAMonce", NoDiagnosticsPamonce)
    result = module.KMedoids(
        diss=_distance_matrix([0.0, 1.0, 2.0, 3.0]),
        k=2,
        method="PAMonce",
        return_diagnostics=False,
        verbose=False,
    )

    assert NoDiagnosticsPamonce.diagnostics.call_count == 0
    np.testing.assert_array_equal(result, np.ones(4, dtype=np.int32))
    assert collection_modes == [False]


def test_public_api_enables_core_diagnostics_when_requested(monkeypatch):
    module = importlib.import_module("sequenzo.clustering.k_medoids")
    collection_modes = []

    class CaptureDiagnosticsPamonce:
        def __init__(self, n, diss, centroids, npass, weights, *options):
            self.n = n

        def set_collect_diagnostics(self, enabled):
            collection_modes.append(enabled)

        def runclusterloop_one_based(self):
            return np.ones(self.n, dtype=np.int32)

        def objective(self):
            return 0.0

        def diagnostics(self):
            return {"worker_count": 1}

    monkeypatch.setattr(
        module.clustering_c_code, "PAMonce", CaptureDiagnosticsPamonce)
    _, diagnostics = module.KMedoids(
        diss=_distance_matrix([0.0, 1.0, 2.0, 3.0]),
        k=2,
        method="PAMonce",
        return_diagnostics=True,
        verbose=False,
    )

    assert collection_modes == [True]
    assert diagnostics["worker_count"] == 1


def test_single_build_pass_does_not_initialize_a_random_generator(monkeypatch):
    module = importlib.import_module("sequenzo.clustering.k_medoids")
    default_rng = mock.Mock(wraps=module.np.random.default_rng)
    monkeypatch.setattr(module.np.random, "default_rng", default_rng)
    result = module.KMedoids(
        diss=_distance_matrix([0.0, 1.0, 2.0, 3.0]),
        k=2,
        method="PAMonce",
        npass=1,
        verbose=False,
    )

    assert default_rng.call_count == 0
    assert len(result) == 4


def test_build_reuses_precomputed_distance_properties():
    _, diagnostics = KMedoids(
        diss=_distance_matrix([0.0, 1.0, 2.0, 3.0]),
        k=2,
        method="PAMonce",
        npass=1,
        threads=1,
        return_diagnostics=True,
        verbose=False,
    )

    assert diagnostics["distance_properties_precomputed"] is True
