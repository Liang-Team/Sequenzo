"""
@Author  : Yuqi Liang 梁彧祺
@File    : fuzzy_helpers.py
@Time    : 14/05/2026
@Desc    :
High-level fuzzy clustering helpers aligned with Studer (2018) and WeightedCluster.

Provides FANNY-based clustering (``cluster::fanny``; Studer 2018 section 4.1),
most-typical-member extraction, membership summaries, and a unified entry point
that also exposes optional WeightedCluster ``wfcmdd`` extensions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd

from sequenzo.clustering.sequences_to_variables.fanny import FannyResult, fanny
from .wfcmdd_fuzzy_clustering import WfcmddResult, wfcmdd


@dataclass
class FuzzyClusterResult:
    """Unified fuzzy clustering result (FANNY or wfcmdd)."""

    membership: np.ndarray
    method: str
    memb_exp: float
    clustering: Optional[np.ndarray] = None
    objective: Optional[float] = None
    converged: Optional[bool] = None
    iterations: Optional[int] = None
    raw: Optional[Union[FannyResult, WfcmddResult]] = None

    @property
    def memb(self) -> np.ndarray:
        """Alias used by WeightedCluster-style code."""
        return self.membership


def membership_summary(
    membership: Union[np.ndarray, pd.DataFrame],
    as_dataframe: bool = True,
) -> Union[pd.DataFrame, dict]:
    """
    Descriptive statistics of fuzzy membership strengths by cluster.

    Mirrors ``summary(fclust$membership)`` in the Studer R tutorial.

    Parameters
    ----------
    membership : array-like, shape (n, k)
        Row-stochastic membership matrix.
    as_dataframe : bool, default True
        If True, return a DataFrame with rows Min, 1st Qu., Median, Mean,
        3rd Qu., Max (R ``summary`` layout). Otherwise return a nested dict.

    Returns
    -------
    pd.DataFrame or dict
    """
    if isinstance(membership, pd.DataFrame):
        frame = membership.astype(float)
    else:
        membership = np.asarray(membership, dtype=np.float64)
        if membership.ndim != 2:
            raise ValueError("membership must be a 2D matrix.")
        frame = pd.DataFrame(
            membership,
            columns=[f"V{idx + 1}" for idx in range(membership.shape[1])],
        )

    stats = frame.describe(percentiles=[0.25, 0.5, 0.75]).loc[
        ["min", "25%", "50%", "mean", "75%", "max"]
    ]
    stats.index = ["Min.", "1st Qu.", "Median", "Mean", "3rd Qu.", "Max."]
    if as_dataframe:
        return stats
    return {col: stats[col].to_dict() for col in stats.columns}


def most_typical_members(
    membership: Union[np.ndarray, pd.DataFrame],
    top_n: int = 1,
    labels: Optional[Sequence] = None,
) -> pd.DataFrame:
    """
    Identify the most typical sequence in each fuzzy cluster.

    Mirrors Studer (2018) section 4.2.1: for each cluster column, return the
    observation(s) with the highest membership strength (not PAM medoids).

    Parameters
    ----------
    membership : array-like, shape (n, k)
        Fuzzy membership matrix.
    top_n : int, default 1
        Number of top sequences to return per cluster.
    labels : sequence, optional
        Sequence identifiers (length ``n``). When omitted, row indices are used.

    Returns
    -------
    pd.DataFrame
        Columns ``cluster``, ``rank``, ``index``, ``membership``, and ``label``
        when ``labels`` is provided.
    """
    if isinstance(membership, pd.DataFrame):
        frame = membership.astype(float)
        cluster_names = [str(col) for col in frame.columns]
        values = frame.to_numpy(dtype=np.float64, copy=False)
    else:
        values = np.asarray(membership, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("membership must be a 2D matrix.")
        cluster_names = [str(idx + 1) for idx in range(values.shape[1])]

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")
    n_obs, n_clusters = values.shape
    if labels is not None and len(labels) != n_obs:
        raise ValueError("labels must have one entry per sequence.")

    rows: list[dict] = []
    for cluster_idx, cluster_name in enumerate(cluster_names):
        column = values[:, cluster_idx]
        n_keep = min(top_n, n_obs)
        if n_keep < n_obs:
            top_indices = np.argpartition(-column, n_keep - 1)[:n_keep]
            top_indices = top_indices[np.argsort(-column[top_indices])]
        else:
            top_indices = np.argsort(-column)
        for rank, row_index in enumerate(top_indices, start=1):
            entry = {
                "cluster": cluster_name,
                "rank": rank,
                "index": int(row_index),
                "membership": float(column[row_index]),
            }
            if labels is not None:
                entry["label"] = labels[row_index]
            rows.append(entry)
    return pd.DataFrame(rows)


def get_fuzzy_clusters(
    diss: np.ndarray,
    n_clusters: int,
    memb_exp: float = 1.5,
    method: Literal["fanny", "wfcmdd"] = "fanny",
    weights: Optional[np.ndarray] = None,
    wfcmdd_method: str = "FCMdd",
    **kwargs,
) -> FuzzyClusterResult:
    """
    Run fuzzy clustering on a distance matrix.

    Parameters
    ----------
    diss : np.ndarray
        Square ``(n, n)`` distance matrix.
    n_clusters : int
        Number of fuzzy clusters ``k``.
    memb_exp : float, default 1.5
        Fuzziness exponent. For ``method="fanny"`` this is passed to
        :func:`~sequenzo.clustering.sequences_to_variables.fanny.fanny` as
        ``memb_exp``. For ``method="wfcmdd"`` it is passed as ``m``.
    method : {"fanny", "wfcmdd"}, default "fanny"
        ``"fanny"`` reproduces the Studer (2018) fuzzy sequence clustering
        workflow (``cluster::fanny(diss, k, diss=TRUE, memb.exp=...)``; see
        Studer 2018 section 4.1). ``"wfcmdd"`` is an optional
        WeightedCluster-style extension for distance-based fuzzy C-medoids
        (``FCMdd``, ``NCdd``, ``PCMdd``, etc.); it is **not** the FANNY
        algorithm used in Studer (2018).
    weights : np.ndarray, optional
        Observation weights (wfcmdd only).
    wfcmdd_method : str, default "FCMdd"
        Variant for wfcmdd (``NCdd``, ``HNCdd``, ``FCMdd``, ``PCMdd``).
    **kwargs
        Extra keyword arguments forwarded to :func:`fanny` or :func:`wfcmdd`.

    Returns
    -------
    FuzzyClusterResult
    """
    diss = np.asarray(diss, dtype=np.float64)
    if method == "fanny":
        result = fanny(diss, k=n_clusters, memb_exp=memb_exp, **kwargs)
        return FuzzyClusterResult(
            membership=result.membership,
            method="fanny",
            memb_exp=result.memb_exp,
            clustering=result.clustering,
            objective=result.objective,
            converged=result.converged,
            iterations=result.iterations,
            raw=result,
        )

    if method == "wfcmdd":
        seeds = kwargs.pop("memb", None)
        if seeds is None:
            seeds = np.linspace(0, diss.shape[0] - 1, n_clusters, dtype=int)
        result = wfcmdd(
            diss,
            memb=seeds,
            weights=weights,
            method=wfcmdd_method,
            m=memb_exp,
            **kwargs,
        )
        memb = result.memb
        if memb.shape[1] > n_clusters:
            memb = memb[:, :n_clusters]
        memb = memb / np.maximum(memb.sum(axis=1, keepdims=True), 1e-15)
        clustering = np.argmax(memb, axis=1)
        return FuzzyClusterResult(
            membership=memb,
            method="wfcmdd",
            memb_exp=memb_exp,
            clustering=clustering,
            objective=result.functional,
            raw=result,
        )

    raise ValueError('method must be "fanny" or "wfcmdd".')
