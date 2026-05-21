"""
@Author  : Yuqi Liang 梁彧祺
@File    : mc_clust.py
@Time    : 20/05/2026 20:40
@Desc    : Cluster comparison and quality across replicates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from sequenzo.clustering import KMedoids, k_medoids_range
from sequenzo.clustering.compare_cluster_methods import hierarchical_cluster_range
from .mc_diss import MCDissList


def _cluster_labels(diss: Union[np.ndarray, pd.DataFrame], k: int, method: str = "PAM") -> np.ndarray:
    mat = np.asarray(diss, dtype=float)
    if mat.ndim == 1:
        from scipy.spatial.distance import squareform

        mat = squareform(mat)
    if method.upper() in ("PAM", "KM"):
        return KMedoids(diss=mat, k=k, cluster_only=True)
    raise ValueError(f"Unsupported clustmeth for cluster-only: {method}")


def _ari_rand_metrics(labels_a: np.ndarray, labels_b: np.ndarray) -> dict:
    """Adjusted Rand, Rand, and related indices without aricode dependency."""
    from sklearn.metrics import adjusted_rand_score, rand_score

    a = np.asarray(labels_a, dtype=int)
    b = np.asarray(labels_b, dtype=int)
    return {
        "ARI": adjusted_rand_score(a, b),
        "RI": rand_score(a, b),
    }


def mc_clustcomp(
    clustlist: List[np.ndarray],
    clust_o: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Compare MC cluster solutions to observed clusters (R ``MCclustcomp``).

    Uses sklearn ARI/RI; Chi2 row is omitted (R replaces it with Cramer's V).
    """
    items = list(clustlist)
    if clust_o is None:
        clust_o = items[-1]
        items = items[:-1]
    clust_o = np.asarray(clust_o, dtype=int).ravel()
    cols = []
    for lab in items:
        lab = np.asarray(lab, dtype=int).ravel()
        cols.append(_ari_rand_metrics(lab, clust_o))
    frame = pd.DataFrame(cols)
    frame.index = ["ARI", "RI"]
    return frame


@dataclass
class MCClustQualResult:
    qual_tab: List[pd.DataFrame]
    qual_max: pd.DataFrame
    max_freq: pd.DataFrame
    qual_obs: Optional[pd.DataFrame]


def mc_clustqual(
    disslist: MCDissList,
    ncluster: Union[int, Sequence[int]] = 10,
    *,
    clustmeth: str = "PAM",
    weights: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> MCClustQualResult:
    """
    Cluster quality statistics over MC-replicated distance matrices (R ``MCclustqual``).

    Parameters
    ----------
    ncluster
        Maximum number of groups (integer) or explicit ``k`` values for hierarchical methods.
    """
    isobs = getattr(disslist, "obs", False)
    items = list(disslist)
    mat0 = np.asarray(items[0], dtype=float)
    if mat0.ndim == 1:
        from scipy.spatial.distance import squareform

        mat0 = squareform(mat0)
    ncases = mat0.shape[0]

    if isinstance(ncluster, int):
        kvals = list(range(2, ncluster + 1))
    else:
        kvals = [int(k) for k in ncluster]

    qlist: List[pd.DataFrame] = []
    for diss in items:
        d = np.asarray(diss, dtype=float)
        if d.ndim == 1:
            from scipy.spatial.distance import squareform

            d = squareform(d)
        if clustmeth.upper() in ("PAM", "KM"):
            res = k_medoids_range(d, kvals=kvals, weights=weights, **kwargs)
            qlist.append(res.stats)
        else:
            res = hierarchical_cluster_range(
                d, ncluster=max(kvals), method=clustmeth, weights=weights, **kwargs
            )
            qlist.append(res.stats)

    qobs = None
    if isobs:
        qobs = qlist[-1]
        qlist = qlist[:-1]

    max_rows = []
    for stats in qlist:
        st = stats.copy()
        if "HC" in st.columns:
            st = st.copy()
            st["HC"] = -st["HC"]
        row = (st.idxmax(axis=0) + 2).astype(int)
        max_rows.append(row.values)
    tabmax = np.vstack(max_rows)
    qual_max = pd.DataFrame(tabmax, columns=stats.columns)
    max_freq = pd.DataFrame(
        {c: pd.Series(tabmax[:, i]).value_counts().reindex(kvals, fill_value=0) for i, c in enumerate(qual_max.columns)}
    )

    return MCClustQualResult(
        qual_tab=qlist,
        qual_max=qual_max,
        max_freq=max_freq,
        qual_obs=qobs,
    )
