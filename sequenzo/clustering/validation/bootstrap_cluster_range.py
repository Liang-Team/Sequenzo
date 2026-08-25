"""
@Author  : Yuqi Liang 梁彧祺
@File    : bootstrap_cluster_range.py
@Time    : 07/05/2025 11:45
@Desc    : 
Bootstrap cluster quality ranges (WeightedCluster ``bootclustrange``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from sequenzo.dissimilarity_measures.get_distance_matrix import get_distance_matrix

from .partition_quality import METRIC_ORDER, ClusterRangeResult, cluster_range_from_partitions


@dataclass
class BootClusterRangeResult(ClusterRangeResult):
    """Bootstrap summary of cluster quality across resamples."""

    clustering: pd.DataFrame
    kvals: np.ndarray
    stats: pd.DataFrame
    boot: List[np.ndarray]
    meant: pd.DataFrame
    stderr: pd.DataFrame


def _stratified_sample(strata: np.ndarray, sample_size: int, rng: np.random.Generator) -> np.ndarray:
    values, counts = np.unique(strata, return_counts=True)
    proportions = counts / counts.sum()
    to_sample = np.round(proportions * sample_size).astype(int)
    correction = sample_size - int(to_sample.sum())
    if correction != 0:
        adjust = rng.choice(len(to_sample), size=abs(correction), replace=False)
        to_sample[adjust] += int(np.sign(correction))
    sample_indices: List[int] = []
    for value, size in zip(values, to_sample):
        pool = np.flatnonzero(strata == value)
        sample_indices.extend(rng.choice(pool, size=int(size), replace=False).tolist())
    return np.asarray(sample_indices, dtype=int)


def _draw_bootstrap_sample(
    clustering: pd.DataFrame,
    sample_size: int,
    sampling: str,
    strata: Optional[np.ndarray],
    medoids: Optional[Sequence[int]],
    rng: np.random.Generator,
) -> np.ndarray:
    n_obs = clustering.shape[0]
    if sampling == "strata":
        if strata is None:
            raise ValueError("sampling='strata' requires strata.")
        return _stratified_sample(strata, sample_size, rng)
    if sampling == "medoids":
        if medoids is None:
            raise ValueError("sampling='medoids' requires medoids.")
        medoid_idx = np.unique(np.asarray(medoids, dtype=int) - 1)
        if np.any((medoid_idx < 0) | (medoid_idx >= n_obs)):
            raise ValueError("medoids must contain 1-based row indices.")
        remaining = sample_size - medoid_idx.size
        if remaining < 0:
            raise ValueError("sample_size cannot be smaller than the medoid count.")
        pool = np.setdiff1d(np.arange(n_obs), medoid_idx, assume_unique=True)
        extra = rng.choice(pool, size=remaining, replace=False)
        return np.sort(np.concatenate([medoid_idx, extra]))
    if sampling != "simple":
        raise ValueError("sampling must be 'simple', 'strata', or 'medoids'.")
    return rng.choice(n_obs, size=sample_size, replace=False)


def boot_cluster_range(
    clustering: Union[np.ndarray, pd.DataFrame],
    seqdata=None,
    seqdist_kwargs: Optional[Dict[str, Any]] = None,
    distance_matrix: Optional[np.ndarray] = None,
    distance_builder: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    n_boot: int = 100,
    sample_size: int = 1000,
    sampling: str = "clustering",
    strata: Optional[np.ndarray] = None,
    medoids: Optional[Sequence[int]] = None,
    weights: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
) -> BootClusterRangeResult:
    """
    Bootstrap cluster quality ranges (WeightedCluster ``bootclustrange``).

    Either ``seqdata`` + ``seqdist_kwargs`` or a ``distance_builder`` callback
    must be provided to rebuild distances on each bootstrap sample.
    ``medoids`` uses the 1-based row indices returned by ``KMedoids``.
    """
    if isinstance(clustering, pd.DataFrame):
        clustering_frame = clustering.copy()
    else:
        clustering_arr = np.asarray(clustering)
        if clustering_arr.ndim == 1:
            clustering_arr = clustering_arr.reshape(-1, 1)
        clustering_frame = pd.DataFrame(clustering_arr)

    if strata is None and sampling == "clustering":
        strata = clustering_frame.iloc[:, -1].to_numpy()
        sampling = "strata"

    seqdist_kwargs = dict(seqdist_kwargs or {})
    rng = np.random.default_rng(random_state)
    boot_stats: List[List[np.ndarray]] = [[] for _ in range(clustering_frame.shape[1])]

    for _ in range(n_boot):
        sample = _draw_bootstrap_sample(
            clustering=clustering_frame,
            sample_size=sample_size,
            sampling=sampling,
            strata=strata,
            medoids=medoids,
            rng=rng,
        )
        if distance_builder is not None:
            diss = distance_builder(sample)
        elif seqdata is not None:
            seqdist_kwargs["seqdata"] = seqdata.data.iloc[sample]
            diss = get_distance_matrix(**seqdist_kwargs)
            if isinstance(diss, pd.DataFrame):
                diss = diss.to_numpy(dtype=np.float64)
        else:
            raise ValueError("Provide seqdata/seqdist_kwargs or distance_builder.")
        cqi = cluster_range_from_partitions(
            diss,
            clustering_frame.iloc[sample, :],
            weights=None if weights is None else weights[sample],
        )
        for idx in range(clustering_frame.shape[1]):
            boot_stats[idx].append(cqi.stats.iloc[idx].to_numpy(dtype=np.float64))

    kvals = np.array([clustering_frame[col].nunique() for col in clustering_frame.columns], dtype=int)
    renamed = [f"cluster{k}" for k in kvals]
    clustering_out = clustering_frame.copy()
    clustering_out.columns = renamed

    boot_arrays = [np.vstack(values) for values in boot_stats]
    meant = np.vstack([values.mean(axis=0) for values in boot_arrays])
    stderr = np.vstack([values.std(axis=0, ddof=1) for values in boot_arrays])
    if distance_matrix is None:
        stats = pd.DataFrame(meant, index=renamed, columns=METRIC_ORDER)
    else:
        stats = cluster_range_from_partitions(
            distance_matrix,
            clustering_frame,
            weights=weights,
        ).stats

    return BootClusterRangeResult(
        clustering=clustering_out,
        kvals=kvals,
        stats=stats,
        boot=boot_arrays,
        meant=pd.DataFrame(meant, index=renamed, columns=METRIC_ORDER),
        stderr=pd.DataFrame(stderr, index=renamed, columns=METRIC_ORDER),
    )
