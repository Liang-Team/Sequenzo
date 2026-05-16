"""
Compare multiple clustering methods across a range of k values.

Mirrors WeightedCluster ``wcCmpCluster``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform

from sequenzo.big_data.clara.utils.get_weighted_diss import get_weighted_diss
from sequenzo.clustering.sequenzo_fastcluster.fastcluster import linkage

from sequenzo.clustering.utils.weightedcluster_compat import (
    cutree_labels,
    divisive_hclust_linkage,
)

from .k_medoids_range import k_medoids_range
from .validation.partition_quality import ClusterRangeResult, cluster_range_from_partitions


HCLUST_METHODS = (
    "ward.d",
    "ward.d2",
    "single",
    "complete",
    "average",
    "mcquitty",
    "median",
    "centroid",
)
NO_WEIGHT_METHODS = ("diana", "beta.flexible")
ALL_METHODS = HCLUST_METHODS + ("pam",) + NO_WEIGHT_METHODS

_LINKAGE_METHOD = {
    "ward.d": "ward",
    "ward.d2": "ward_d2",
    "single": "single",
    "complete": "complete",
    "average": "average",
    "mcquitty": "weighted",
    "median": "median",
    "centroid": "centroid",
}


@dataclass
class ClusterRangeFamilyResult:
    """Container matching the core fields of R ``clustrangefamily`` objects."""

    results: Dict[str, ClusterRangeResult]
    allstats: pd.DataFrame
    param: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": {name: result.to_dict() for name, result in self.results.items()},
            "allstats": self.allstats,
            "param": self.param,
        }


def _resolve_methods(methods: Union[str, Sequence[str]], weights: Optional[np.ndarray]) -> List[str]:
    if isinstance(methods, str) and methods.lower() == "all":
        selected = list(ALL_METHODS) if weights is None else list(HCLUST_METHODS) + ["pam"]
    else:
        selected = [method.lower() for method in methods]

    if weights is not None and any(method in NO_WEIGHT_METHODS for method in selected):
        raise ValueError(
            "diana and beta.flexible cannot be used with weights, matching WeightedCluster."
        )
    return selected


def _linkage_input_from_diss(
    diss: np.ndarray,
    weights: Optional[np.ndarray],
) -> np.ndarray:
    """
    Build the condensed distance vector expected by fastcluster (R ``as.dist`` order).

    Passing a full square matrix to ``linkage`` can yield a different tree than
    ``hclust(as.dist(diss), method = "ward.D")``; R parity requires condensed form.
    """
    linkage_input = np.asarray(diss, dtype=np.float64, order="C").copy()
    if weights is not None:
        linkage_input = get_weighted_diss(
            linkage_input, np.asarray(weights, dtype=np.float64)
        )
    if linkage_input.ndim == 2:
        linkage_input = squareform(linkage_input, checks=False)
    return linkage_input


def hierarchical_cluster_range(
    diss: np.ndarray,
    maxcluster: int,
    *,
    method: str = "ward.d",
    weights: Optional[np.ndarray] = None,
) -> ClusterRangeResult:
    """
    Hierarchical partitions and quality for k = 2, ..., ``maxcluster``.

    Mirrors WeightedCluster ``as.clustrange(hclust, diss, ncluster = maxcluster)``.
    """
    return _hierarchical_range(diss, weights, method.lower(), maxcluster)


def _hierarchical_range(
    diss: np.ndarray,
    weights: Optional[np.ndarray],
    method: str,
    maxcluster: int,
) -> ClusterRangeResult:
    if method in NO_WEIGHT_METHODS:
        linkage_matrix = divisive_hclust_linkage(diss, method)
    else:
        linkage_input = _linkage_input_from_diss(diss, weights)
        linkage_matrix = linkage(linkage_input, method=_LINKAGE_METHOD[method])
    partitions = {
        f"cluster{k}": cutree_labels(linkage_matrix, k)
        for k in range(2, maxcluster + 1)
    }
    clustering = pd.DataFrame(partitions)
    return cluster_range_from_partitions(diss, clustering, weights=weights)


def compare_cluster_methods(
    diss: np.ndarray,
    maxcluster: int,
    weights: Optional[np.ndarray] = None,
    *,
    methods: Union[str, Sequence[str]] = "all",
    pam_combine: bool = True,
    random_state: Optional[int] = None,
) -> ClusterRangeFamilyResult:
    """
    Evaluate several clustering methods on the same distance matrix.

    The function mirrors ``wcCmpCluster`` for hierarchical and PAM methods.
    """
    if maxcluster < 2:
        raise ValueError("maxcluster should be greater than 2.")

    diss = np.asarray(diss, dtype=np.float64, order="C")
    selected = _resolve_methods(methods, weights)
    kvals = list(range(2, maxcluster + 1))
    results: Dict[str, ClusterRangeResult] = {}

    for method in selected:
        if method == "pam":
            results[method] = k_medoids_range(
                diss,
                kvals=kvals,
                weights=weights,
                random_state=random_state,
            )
            continue

        if method in NO_WEIGHT_METHODS:
            results[method] = _hierarchical_range(diss, weights, method, maxcluster)
            if pam_combine:
                linkage_matrix = divisive_hclust_linkage(diss, method)
                results[f"pam.{method}"] = k_medoids_range(
                    diss,
                    kvals=kvals,
                    weights=weights,
                    initialclust=linkage_matrix,
                    random_state=random_state,
                )
            continue

        results[method] = _hierarchical_range(diss, weights, method, maxcluster)
        if pam_combine:
            linkage_input = _linkage_input_from_diss(diss, weights)
            linkage_matrix = linkage(linkage_input, method=_LINKAGE_METHOD[method])
            results[f"pam.{method}"] = k_medoids_range(
                diss,
                kvals=kvals,
                weights=weights,
                initialclust=linkage_matrix,
                random_state=random_state,
            )

    allstats = []
    for name, result in results.items():
        frame = result.stats.copy()
        frame["method"] = name
        frame["ngroup"] = result.kvals
        allstats.append(frame.reset_index(names="Cluster"))
    combined = pd.concat(allstats, ignore_index=True)
    return ClusterRangeFamilyResult(
        results=results,
        allstats=combined,
        param={
            "method": selected,
            "pam_combine": pam_combine,
            "all_methods": list(results.keys()),
            "kvals": kvals,
        },
    )
