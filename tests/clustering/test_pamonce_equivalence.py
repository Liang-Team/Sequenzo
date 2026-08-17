import json
import importlib
import os
import subprocess
import sys

import numpy as np
import pytest

from sequenzo.clustering import KMedoids


def _distance_matrix(points):
    points = np.asarray(points, dtype=np.float64)
    return np.abs(points[:, None] - points[None, :])


def _run_in_fresh_process(method, threads, points, initial_medoids):
    code = f"""
import json
import numpy as np
from sequenzo.clustering import KMedoids

points = np.asarray({points!r}, dtype=np.float64)
diss = np.abs(points[:, None] - points[None, :])
result = KMedoids(
    diss=diss,
    k={len(initial_medoids)},
    initialclust={initial_medoids!r},
    npass=0,
    method={method!r},
    verbose=False,
)
print(json.dumps(result.tolist()))
"""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.getcwd(),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return np.asarray(json.loads(completed.stdout.strip()), dtype=np.int32)


def test_pamonce_matches_classic_pam_for_sub_tolerance_improvement():
    diss = _distance_matrix(np.array([0.0, 1.0, 10.0, 11.0]) * 1e-12)

    classic = KMedoids(
        diss=diss,
        k=2,
        initialclust=[0, 1],
        npass=0,
        method="PAM",
        verbose=False,
    )
    fast = KMedoids(
        diss=diss,
        k=2,
        initialclust=[0, 1],
        npass=0,
        method="PAMonce",
        verbose=False,
    )

    np.testing.assert_array_equal(fast, classic)


def test_pamonce_is_identical_with_one_and_eight_threads_on_ties():
    one_core = _run_in_fresh_process(
        method="PAMonce",
        threads=1,
        points=[0.0, 1.0, 10.0, 11.0],
        initial_medoids=[0, 1],
    )
    eight_cores = _run_in_fresh_process(
        method="PAMonce",
        threads=8,
        points=[0.0, 1.0, 10.0, 11.0],
        initial_medoids=[0, 1],
    )

    np.testing.assert_array_equal(eight_cores, one_core)


def test_classic_pam_is_identical_with_one_and_eight_threads_on_ties():
    one_core = _run_in_fresh_process(
        method="PAM",
        threads=1,
        points=[0.0, 1.0, 2.0, 3.0, 4.0],
        initial_medoids=[0, 4],
    )
    eight_cores = _run_in_fresh_process(
        method="PAM",
        threads=8,
        points=[0.0, 1.0, 2.0, 3.0, 4.0],
        initial_medoids=[0, 4],
    )

    np.testing.assert_array_equal(one_core, np.array([2, 2, 2, 5, 5]))
    np.testing.assert_array_equal(eight_cores, one_core)


def test_explicit_medoids_keep_condensed_input_without_squareform(monkeypatch):
    from scipy.spatial import distance as scipy_distance

    square = _distance_matrix([0.0, 1.0, 10.0, 11.0])
    condensed = scipy_distance.squareform(square, checks=False)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("explicit medoids must not expand condensed input")

    monkeypatch.setattr(scipy_distance, "squareform", fail_if_called)
    sys.modules.pop("sequenzo.clustering.k_medoids", None)
    module = __import__("sequenzo.clustering.k_medoids", fromlist=["KMedoids"])

    result = module.KMedoids(
        diss=condensed,
        k=2,
        initialclust=[0, 2],
        npass=0,
        method="PAMonce",
        verbose=False,
    )

    np.testing.assert_array_equal(result, np.array([1, 1, 3, 3]))


def test_kmedoids_import_does_not_eagerly_load_scipy_or_pandas():
    code = """
import json
import sys
import sequenzo.clustering.k_medoids
print(json.dumps({name: name in sys.modules for name in ("scipy", "pandas")}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.getcwd(),
        check=True,
        capture_output=True,
        text=True,
    )
    loaded = json.loads(completed.stdout.strip())

    assert loaded == {"scipy": False, "pandas": False}


def test_initial_medoid_array_is_not_modified_by_clustering():
    diss = _distance_matrix([0.0, 1.0, 10.0, 11.0])
    initial = np.array([0, 1], dtype=np.int32)
    before = initial.copy()

    KMedoids(
        diss=diss,
        k=2,
        initialclust=initial,
        npass=0,
        method="PAMonce",
        verbose=False,
    )

    np.testing.assert_array_equal(initial, before)


def test_default_unit_weights_do_not_allocate_a_dense_ones_vector(monkeypatch):
    diss = _distance_matrix([0.0, 1.0, 10.0, 11.0])

    def fail_if_called(*args, **kwargs):
        raise AssertionError("PAM unit weights should use the C++ specialization")

    monkeypatch.setattr(np, "ones", fail_if_called)
    for method in ("PAM", "PAMonce"):
        result = KMedoids(
            diss=diss,
            k=2,
            initialclust=[0, 2],
            npass=0,
            method=method,
            verbose=False,
        )
        np.testing.assert_array_equal(result, np.array([1, 1, 3, 3]))


def test_unit_weight_specialization_matches_explicit_ones():
    rng = np.random.default_rng(20260816)
    points = rng.normal(size=(24, 3))
    diss = np.sqrt(
        ((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)
    )

    for method in ("PAM", "PAMonce"):
        for initialclust, npass in (([0, 6, 12, 18], 0), (None, 1)):
            common = dict(
                diss=diss,
                k=4,
                initialclust=initialclust,
                npass=npass,
                method=method,
                verbose=False,
            )
            specialized = KMedoids(weights=None, **common)
            explicit = KMedoids(weights=np.ones(24), **common)
            np.testing.assert_array_equal(specialized, explicit)


def test_internal_random_medoids_are_not_reinterpreted_as_one_based(monkeypatch):
    module = importlib.import_module("sequenzo.clustering.k_medoids")
    captured = {}

    class CapturePam:
        def __init__(self, n, diss, centroids, npass, weights):
            captured["centroids"] = centroids.copy()
            self.n = n

        def runclusterloop_one_based(self):
            return np.ones(self.n, dtype=np.int32)

    monkeypatch.setattr(module.clustering_c_code, "PAM", CapturePam)
    module.KMedoids(
        diss=np.zeros((10, 10)),
        k=2,
        initialclust=None,
        npass=0,
        random_state=0,
        method="PAM",
        verbose=False,
    )

    np.testing.assert_array_equal(captured["centroids"], np.array([7, 6]))


def test_membership_initialized_pamonce_matches_classic_pam():
    diss = _distance_matrix([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
    membership = np.array([1, 1, 1, 2, 2, 2], dtype=np.int32)
    common = dict(
        diss=diss,
        k=2,
        initialclust=membership,
        npass=0,
        verbose=False,
    )

    classic = KMedoids(method="PAM", **common)
    fast = KMedoids(method="PAMonce", **common)
    np.testing.assert_array_equal(fast, classic)


def test_extreme_finite_weights_fall_back_to_classic_decisions():
    points = np.array([4.0, 5.0, -0.1, 1.0, 3.0, 2.5, 0.0, 10.0])
    diss = np.abs(points[:, None] - points[None, :])
    weights = np.array([
        3.3e307,
        3.3e307,
        3.3e307,
        6.6e307,
        1e100,
        1e100,
        1e100,
        1e100,
    ])
    common = dict(
        diss=diss,
        k=2,
        weights=weights,
        initialclust=[7, 8],
        npass=0,
        verbose=False,
    )

    classic = KMedoids(method="PAM", **common)
    fast = KMedoids(method="PAMonce", **common)
    np.testing.assert_array_equal(fast, classic)


def test_pam_low_level_bindings_reject_strided_arrays():
    from sequenzo.clustering import clustering_c_code

    diss = np.arange(36, dtype=np.float64).reshape(6, 6).T
    centroids = np.array([0, 3], dtype=np.int32)
    weights = np.ones(6, dtype=np.float64)
    for engine_type in (clustering_c_code.PAM, clustering_c_code.PAMonce):
        with pytest.raises(TypeError):
            engine_type(6, diss, centroids, 0, weights)


def test_negative_weight_build_remains_defined_and_equivalent():
    diss = _distance_matrix([0.0, 1.0, 4.0, 9.0, 10.0])
    common = dict(
        diss=diss,
        k=2,
        weights=np.array([-4.0, -3.0, -2.0, -1.0, -0.5]),
        initialclust=None,
        npass=1,
        verbose=False,
    )

    classic = KMedoids(method="PAM", **common)
    fast = KMedoids(method="PAMonce", **common)
    np.testing.assert_array_equal(fast, classic)


def test_pam_engines_can_emit_one_based_results_without_a_python_pass():
    from sequenzo.clustering import clustering_c_code

    diss = _distance_matrix([0.0, 1.0, 10.0, 11.0])
    weights = np.full(4, 1.0, dtype=np.float64)
    for engine_type in (clustering_c_code.PAM, clustering_c_code.PAMonce):
        engine = engine_type(
            4,
            diss,
            np.array([0, 2], dtype=np.int32),
            0,
            weights,
        )
        result = engine.runclusterloop_one_based()
        np.testing.assert_array_equal(result, np.array([1, 1, 3, 3]))


def test_pamonce_matches_classic_across_representative_geometries_and_threads():
    code = r"""
import json
import numpy as np
from sequenzo.clustering import KMedoids

rng = np.random.default_rng(20260816)
continuous = rng.normal(size=(20, 3))
discrete = rng.integers(0, 5, size=(20, 6))
duplicates = np.repeat(rng.normal(size=(10, 2)), 2, axis=0)

distance_matrices = [
    np.sqrt(((continuous[:, None, :] - continuous[None, :, :]) ** 2).sum(axis=2)),
    np.count_nonzero(discrete[:, None, :] != discrete[None, :, :], axis=2).astype(float),
    np.sqrt(((duplicates[:, None, :] - duplicates[None, :, :]) ** 2).sum(axis=2)),
]
weight_vectors = [
    np.ones(20, dtype=float),
    rng.uniform(0.25, 2.0, size=20),
]

all_results = []
for diss in distance_matrices:
    condensed = diss[np.triu_indices(20, 1)]
    for weights in weight_vectors:
        for k in (2, 4):
            explicit = np.linspace(0, 19, k, dtype=np.int32)
            for initial, npass in ((explicit, 0), (None, 1)):
                for matrix in (diss, condensed):
                    common = dict(
                        diss=matrix,
                        k=k,
                        weights=weights,
                        initialclust=initial,
                        npass=npass,
                        verbose=False,
                    )
                    classic = KMedoids(method="PAM", **common)
                    fast = KMedoids(method="PAMonce", **common)
                    np.testing.assert_array_equal(fast, classic)

                    classic_objective = float(np.sum(
                        weights * diss[np.arange(20), classic - 1]
                    ))
                    fast_objective = float(np.sum(
                        weights * diss[np.arange(20), fast - 1]
                    ))
                    assert fast_objective == classic_objective
                    all_results.append(fast.tolist())

print(json.dumps(all_results))
"""

    outputs = []
    for threads in (1, 8):
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(threads)
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.getcwd(),
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(json.loads(completed.stdout.strip()))

    assert len(outputs[0]) == 48
    assert outputs[1] == outputs[0]
