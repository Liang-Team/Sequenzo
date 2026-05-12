"""
Multi-k PAM ranges with partition quality.

Mirrors WeightedCluster ``wcKMedRange``.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

from sequenzo.big_data.clara.utils.get_weighted_diss import get_weighted_diss
from sequenzo.clustering.sequenzo_fastcluster.fastcluster import linkage

from .k_medoids import KMedoids
from .utils.weightedcluster_compat import RSampleIntStream, r_k_medoids_range
from .validation.bootstrap_cluster_range import boot_cluster_range
from .validation.partition_quality import ClusterRangeResult, cluster_range_from_partitions


def k_medoids_range(
    diss: np.ndarray,
    kvals: Sequence[int],
    weights: Optional[np.ndarray] = None,
    *,
    initialclust: Optional[Union[np.ndarray, Any]] = None,
    method: str = "PAMonce",
    npass: int = 1,
    n_boot: int = 1,
    sample_size: Optional[int] = None,
    sampling: str = "simple",
    random_state: Optional[int] = None,
) -> ClusterRangeResult:
    """
    Run weighted PAM for several values of ``k`` and evaluate each partition.

    When ``n_boot == 1`` the function mirrors ``wcKMedRange`` with ``R = 1``.
    Larger values bootstrap partition quality on the supplied clustering table.
    """
    diss = np.asarray(diss, dtype=np.float64, order="C")
    if diss.ndim != 2 or diss.shape[0] != diss.shape[1]:
        raise ValueError("diss must be a square distance matrix.")

    kvals = [int(k) for k in kvals]
    if not kvals:
        raise ValueError("kvals must contain at least one cluster count.")

    if random_state is not None and initialclust is None and n_boot <= 1:
        clustering, stats = r_k_medoids_range(
            diss,
            kvals,
            weights,
            method=method,
            seed=random_state,
        )
        return ClusterRangeResult(clustering=clustering, kvals=np.asarray(kvals), stats=stats)

    partitions = []
    sample_stream = (
        RSampleIntStream(random_state, diss.shape[0], kvals) if random_state is not None else None
    )
    for k in kvals:
        initial = sample_stream.sample(k) if sample_stream is not None else None
        labels = KMedoids(
            diss=diss,
            k=k,
            weights=weights,
            npass=npass,
            initialclust=initialclust if initial is None else initial,
            method=method,
            cluster_only=True,
            verbose=False,
        )
        partitions.append(np.asarray(labels).reshape(-1))

    clustering = pd.DataFrame({f"cluster{k}": column for k, column in zip(kvals, partitions)})
    if n_boot <= 1:
        return cluster_range_from_partitions(diss, clustering, weights=weights)

    return boot_cluster_range(
        clustering=clustering,
        distance_builder=lambda idx: diss[np.ix_(idx, idx)],
        n_boot=n_boot,
        sample_size=sample_size or diss.shape[0],
        sampling=sampling,
        weights=weights,
        random_state=random_state,
    )
