"""K-medoids solutions and quality measures for multiple cluster counts."""
from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .k_medoids import KMedoids
from .validation.bootstrap_cluster_range import BootClusterRangeResult, boot_cluster_range
from .validation.partition_quality import (
    METRIC_ORDER,
    ClusterRangeResult,
    cluster_range_from_partitions,
)


def _condensed_subset(distance, indices, n):
    size = len(indices)
    subset = np.empty(size * (size - 1) // 2, dtype=np.float64)
    offset = 0
    for position, left in enumerate(indices[:-1]):
        right = indices[position + 1 :]
        width = right.size
        lower = np.minimum(left, right)
        upper = np.maximum(left, right)
        source = lower * (2 * n - lower - 1) // 2 + (upper - lower - 1)
        subset[offset : offset + width] = distance[source]
        offset += width
    return subset


def _weighted_bootstrap_range(
    diss, clustering, weights, n_boot, sample_size, random_state
):
    n = len(clustering)
    point = cluster_range_from_partitions(diss, clustering, weights=weights)
    if weights is None:
        probability = np.full(n, 1.0 / n)
    else:
        probability = np.asarray(weights, dtype=np.float64)
        total = float(probability.sum())
        if total <= 0 or np.any(probability < 0):
            raise ValueError("weights must be non-negative and sum to a positive value.")
        probability = probability / total
    draw_size = n if sample_size is None else int(sample_size)
    if draw_size < 2:
        raise ValueError("sample_size must be at least 2.")
    rng = np.random.default_rng(random_state)
    boot_stats = [[] for _ in clustering.columns]
    for _ in range(n_boot):
        sample = rng.choice(n, size=draw_size, replace=True, p=probability)
        indices, counts = np.unique(sample, return_counts=True)
        sampled_clustering = clustering.iloc[indices, :]

        if diss.ndim == 1:
            sampled_distance = _condensed_subset(diss, indices, n)
        else:
            sampled_distance = diss[np.ix_(indices, indices)]
        sampled = cluster_range_from_partitions(
            sampled_distance,
            sampled_clustering,
            weights=counts.astype(np.float64),
        )
        for column in range(len(boot_stats)):
            boot_stats[column].append(
                sampled.stats.iloc[column].to_numpy(dtype=np.float64)
            )

    boot = [np.vstack(values) for values in boot_stats]
    meant = np.vstack([values.mean(axis=0) for values in boot])
    stderr = np.vstack([values.std(axis=0, ddof=1) for values in boot])
    return BootClusterRangeResult(
        clustering=point.clustering,
        kvals=point.kvals,
        stats=point.stats,
        boot=boot,
        meant=pd.DataFrame(meant, index=point.stats.index, columns=METRIC_ORDER),
        stderr=pd.DataFrame(stderr, index=point.stats.index, columns=METRIC_ORDER),
    )


def k_medoids_range(
    diss: np.ndarray,
    kvals: Sequence[int],
    weights: Optional[np.ndarray] = None,
    *,
    initialclust: Optional[Union[np.ndarray, Any]] = None,
    method: Union[str, int] = "PAMonce",
    npass: int = 1,
    n_boot: int = 1,
    sample_size: Optional[int] = None,
    sampling: str = "simple",
    random_state: Optional[int] = None,
    threads: Optional[int] = None,
    memory_budget_mb: Optional[float] = None,
) -> ClusterRangeResult:
    """Run K-medoids and evaluate fixed partitions for each value of ``k``."""
    diss = np.asarray(diss, dtype=np.float64, order="C")
    if diss.ndim == 1:
        condensed_length = diss.size
        n = int((1 + np.sqrt(1 + 8 * condensed_length)) / 2)
        if n * (n - 1) // 2 != condensed_length:
            raise ValueError("diss has an invalid condensed-vector length.")
    elif diss.ndim == 2 and diss.shape[0] == diss.shape[1]:
        n = diss.shape[0]
    else:
        raise ValueError("diss must be square or a valid condensed vector.")

    kvals = [int(k) for k in kvals]
    if not kvals:
        raise ValueError("kvals must contain at least one cluster count.")
    if any(k < 2 or k > n for k in kvals):
        raise ValueError(f"each k must be in [2, {n}].")

    weights_cpp = np.empty(0, dtype=np.float64)
    if weights is not None:
        weights_cpp = np.asarray(weights, dtype=np.float64)
        if weights_cpp.shape != (n,):
            raise ValueError(f"weights must contain exactly {n} values.")
        weights_cpp = np.ascontiguousarray(weights_cpp)
    if threads is not None and (
        not isinstance(threads, (int, np.integer)) or threads < 1
    ):
        raise ValueError("threads must be a positive integer or None.")
    if memory_budget_mb is not None and memory_budget_mb <= 0:
        raise ValueError("memory_budget_mb must be positive or None.")
    requested_threads = 0 if threads is None else int(threads)
    memory_budget_bytes = (
        0
        if memory_budget_mb is None
        else int(float(memory_budget_mb) * 1024 * 1024)
    )

    shared_build_prefix = None
    is_pamonce = method == 3 or (
        isinstance(method, str) and method.lower() == "pamonce"
    )
    if initialclust is None and npass == 1 and is_pamonce:
        import sequenzo.clustering.clustering_c_code as clustering_c_code

        max_k = max(kvals)
        build_engine = clustering_c_code.PAMonce(
            n,
            np.ascontiguousarray(diss, dtype=np.float64),
            np.arange(max_k, dtype=np.int32),
            1,
            weights_cpp,
            requested_threads,
            memory_budget_bytes,
        )
        shared_build_prefix = np.asarray(
            build_engine.build_initial_medoids(), dtype=np.int32
        )
        del build_engine

    partitions = []
    for k in kvals:
        if shared_build_prefix is not None:
            initial = shared_build_prefix[:k] + 1
            current_npass = 0
        else:
            initial = initialclust
            current_npass = npass
        labels = KMedoids(
            diss=diss,
            k=k,
            weights=weights,
            npass=current_npass,
            initialclust=initial,
            method=method,
            cluster_only=True,
            verbose=False,
            random_state=random_state,
            threads=threads,
            memory_budget_mb=memory_budget_mb,
        )
        partitions.append(np.asarray(labels).reshape(-1))

    clustering = pd.DataFrame(
        {f"cluster{k}": column for k, column in zip(kvals, partitions)}
    )
    if n_boot <= 1:
        return cluster_range_from_partitions(diss, clustering, weights=weights)
    if sampling not in {"simple", "clustering"}:
        raise ValueError("sampling must be 'simple' or 'clustering'.")

    if sampling == "simple":
        return _weighted_bootstrap_range(
            diss,
            clustering,
            weights=weights,
            n_boot=n_boot,
            sample_size=sample_size,
            random_state=random_state,
        )

    if diss.ndim == 1:
        def distance_builder(indices):
            return _condensed_subset(diss, indices, n)
    else:
        def distance_builder(indices):
            return diss[np.ix_(indices, indices)]

    return boot_cluster_range(
        clustering=clustering,
        distance_matrix=diss,
        distance_builder=distance_builder,
        n_boot=n_boot,
        sample_size=sample_size or n,
        sampling=sampling,
        weights=weights,
        random_state=random_state,
    )
