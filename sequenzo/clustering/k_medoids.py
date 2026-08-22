"""
@Author  : 李欣怡 Xinyi Li, Yuqi Liang 梁彧祺
@File    : KMedoids.py
@Time    : 2025/2/8 11:53
@Desc    : 
"""

import numpy as np

import sequenzo.clustering.clustering_c_code as clustering_c_code


def KMedoids(
    diss,
    k,
    weights=None,
    npass=1,
    initialclust=None,
    method="PAMonce",
    cluster_only=False,
    verbose=True,
    random_state=None,
    threads=None,
    memory_budget_mb=None,
    return_diagnostics=False,
):
    """Run weighted K-medoids on full or condensed distances.

    The result contains 1-based medoid row indices. Set
    ``return_diagnostics=True`` to also return execution diagnostics.
    """
    method_original = method
    if isinstance(method, str):
        method = method.lower()
        method_map = ["kmedoids", "pam", "pamonce"]
        if method in method_map:
            method = method_map.index(method) + 1

    if not (isinstance(method, int) and method in {1, 2, 3}):
        raise ValueError(f"[!] Unknown clustering method: {method_original}.")

    if threads is not None and (not isinstance(threads, (int, np.integer)) or threads < 1):
        raise ValueError("[!] 'threads' should be a positive integer or None.")
    requested_threads = 0 if threads is None else int(threads)
    if memory_budget_mb is not None and memory_budget_mb <= 0:
        raise ValueError("[!] 'memory_budget_mb' should be positive or None.")
    memory_budget_bytes = (
        0 if memory_budget_mb is None else int(float(memory_budget_mb) * 1024 * 1024)
    )

    if verbose:
        method_names = ["KMedoids", "PAM", "PAMonce"]
        method_name = method_names[method - 1]
        print(f"[>] Starting KMedoids clustering (method: {method_name}, k={k})...")

    diss = np.asarray(diss, dtype=np.float64, order="C")
    diss_condensed = None
    if diss.ndim == 1:
        diss_condensed = diss
        cond_len = len(diss)
        nelements = int((1 + np.sqrt(1 + 8 * cond_len)) / 2)
        if nelements * (nelements - 1) // 2 != cond_len:
            raise ValueError(
                "[!] 'diss' 1-D length does not correspond to a valid condensed distance vector."
            )
    elif diss.ndim == 2:
        nelements = diss.shape[0]
        if nelements != diss.shape[1]:
            raise ValueError(
                "[!] 'diss' must be square (n × n); use a 1-D condensed vector of length n(n−1)/2 for condensed form."
            )
    else:
        raise ValueError(
            "[!] 'diss' must be a square distance matrix or a 1-D condensed distance vector "
            "(see scipy.spatial.distance.squareform)."
        )

    unit_weights = weights is None
    if not unit_weights and len(weights) != nelements:
        raise ValueError(f"[!] 'weights' should be a vector of length {nelements}.")

    if npass < 0:
        raise ValueError("[!] 'npass' should be non-negative")
    if k < 2 or k > nelements:
        raise ValueError(f" [!] 'k' should be in [2, {nelements}]")

    needs_full_matrix = False
    medoids_are_internal = initialclust is None
    explicit_initial = initialclust is not None
    if explicit_initial:
        if _validate_linkage_matrix(initialclust):
            from scipy.cluster.hierarchy import cut_tree

            initialclust = cut_tree(initialclust, n_clusters=k).flatten() + 1
        if len(initialclust) == nelements:
            needs_full_matrix = True
            if diss_condensed is not None:
                from scipy.spatial.distance import squareform

                diss = squareform(diss_condensed, checks=False)
            from sequenzo.clustering.utils.disscenter import disscentertrim

            initialclust = disscentertrim(
                diss=diss,
                group=initialclust,
                medoids_index="first",
                weights=weights,
            )
            medoids_are_internal = True
            if len(initialclust) != k:
                raise ValueError(
                    f"[!] 'initialclust' should be a vector of cluster membership with k={k}."
                )
        initialclust = np.asarray(initialclust)
        if len(initialclust) != k:
            raise ValueError(
                f"[!] 'initialclust' should be a vector of medoids index of length :{k}."
            )
        if (
            not medoids_are_internal
            and initialclust.min() >= 1
            and initialclust.max() <= nelements
        ):
            initialclust = initialclust - 1
        if np.any((initialclust >= nelements) | (initialclust < 0)):
            raise ValueError(
                f"[!] Starting medoids should be 0-based indices in 0:{nelements - 1} "
                f"(R-style 1:{nelements} is also accepted)."
            )

    if diss_condensed is not None and not needs_full_matrix:
        diss_cpp = np.ascontiguousarray(diss_condensed, dtype=np.float64)
    else:
        if diss.ndim == 1:
            from scipy.spatial.distance import squareform

            diss = squareform(diss, checks=False)
        diss_cpp = np.ascontiguousarray(diss, dtype=np.float64)
    if unit_weights and method in {2, 3}:
        weights_cpp = np.empty(0, dtype=np.float64)
    elif unit_weights:
        weights_cpp = np.ones(nelements, dtype=np.float64)
    else:
        weights_cpp = np.ascontiguousarray(weights, dtype=np.float64)
    if explicit_initial:
        starts = [(np.asarray(initialclust, dtype=np.int32), 0)]
    elif npass > 0:
        starts = [(np.arange(k, dtype=np.int32), 1)]
        if npass > 1:
            rng = np.random.default_rng(random_state)
            starts.extend(
                (rng.choice(nelements, k, replace=False).astype(np.int32), 0)
                for _ in range(npass - 1)
            )
    else:
        rng = np.random.default_rng(random_state)
        starts = [(rng.choice(nelements, k, replace=False).astype(np.int32), 0)]
        if verbose:
            print("[!] npass=0 without initialclust: using random initial medoids.")

    collect_diagnostics = bool(return_diagnostics)
    compare_signatures = len(starts) > 1
    pass_diagnostics = [] if collect_diagnostics else None
    best_result = None
    best_objective = np.inf
    best_signature = None
    for pass_index, (start, core_npass) in enumerate(starts):
        init_cpp = np.array(start, dtype=np.int32, order="C", copy=True)
        if method == 1:
            engine = clustering_c_code.KMedoid(
                nelements, diss_cpp, init_cpp, core_npass, weights_cpp
            )
            result = engine.runclusterloop() + 1
            diagnostics = {}
        else:
            engine_type = clustering_c_code.PAM if method == 2 else clustering_c_code.PAMonce
            core_args = (
                nelements,
                diss_cpp,
                init_cpp,
                core_npass,
                weights_cpp,
            )
            if requested_threads or memory_budget_bytes:
                engine = engine_type(
                    *core_args, requested_threads, memory_budget_bytes
                )
            else:
                engine = engine_type(*core_args)
            if hasattr(engine, "set_collect_diagnostics"):
                engine.set_collect_diagnostics(collect_diagnostics)
            result = engine.runclusterloop_one_based()
            diagnostics = (
                dict(engine.diagnostics())
                if collect_diagnostics and hasattr(engine, "diagnostics")
                else {}
            )

        result = np.asarray(result, dtype=np.int32)
        objective = (
            float(engine.objective())
            if method in {2, 3} and hasattr(engine, "objective")
            else _pam_objective(diss_cpp, result, weights_cpp, unit_weights, nelements)
        )
        signature = (
            tuple(int(value) for value in result)
            if compare_signatures
            else None
        )
        if collect_diagnostics:
            pass_diagnostics.append(
                {
                    "pass_index": pass_index,
                    "initialization": "build" if core_npass > 0 else "explicit",
                    "objective": objective,
                    **diagnostics,
                }
            )
        if (
            best_result is None
            or objective < best_objective
            or (
                compare_signatures
                and objective == best_objective
                and signature < best_signature
            )
        ):
            best_result = result.copy()
            best_objective = objective
            best_signature = signature
            best_pass = pass_index

    memb_matrix = best_result
    if verbose:
        print("[>] Computed Successfully.")

    if return_diagnostics:
        diagnostics = dict(pass_diagnostics[best_pass])
        diagnostics.update(
            {
                "npass_requested": int(npass),
                "passes_executed": len(starts),
                "selected_pass": best_pass,
                "objective": best_objective,
                "passes": pass_diagnostics,
            }
        )
        return memb_matrix, diagnostics
    return memb_matrix


def _pam_objective(diss, result, weights, unit_weights, nelements):
    medoids = np.asarray(result, dtype=np.int64) - 1
    rows = np.arange(nelements, dtype=np.int64)
    if diss.ndim == 2:
        distances = diss[rows, medoids]
    else:
        left = np.minimum(rows, medoids)
        right = np.maximum(rows, medoids)
        indices = left * (2 * nelements - left - 1) // 2 + right - left - 1
        distances = np.zeros(nelements, dtype=np.float64)
        off_diagonal = left != right
        distances[off_diagonal] = diss[indices[off_diagonal]]
    if unit_weights:
        return float(np.sum(distances))
    return float(np.sum(np.asarray(weights) * distances))

def _validate_linkage_matrix(initialclust):
    if not isinstance(initialclust, np.ndarray):
        return False

    if initialclust.ndim != 2 or initialclust.shape[1] != 4:
        return False

    if initialclust.dtype != np.float64:
        return False

    return True
