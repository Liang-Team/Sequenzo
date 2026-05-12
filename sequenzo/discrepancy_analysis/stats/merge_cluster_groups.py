"""Greedy cluster-label merging by partition quality (TraMineR: dissmergegroups)."""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any, List
def merge_cluster_groups(
    distance_matrix: np.ndarray,
    group: Union[np.ndarray, pd.Series],
    weights: Optional[np.ndarray] = None,
    target_n_groups: Optional[int] = None,
    squared: bool = False,
    measure: str = "ASW",
    crit: float = 0.2,
    ref: str = "max",
    min_group: int = 4,
    small: float = 0.05,
    silent: bool = False,
) -> Dict[str, Any]:
    """
    Iteratively merge groups using TraMineR's dissmergegroups logic.

    This implementation follows the original TraMineR behaviour:
    - Quality-driven greedy merges (default quality measure: ASW),
    - adaptive search restricted to smallest group when needed,
    - stopping by both `min_group` and quality deterioration threshold
      `crit * quality_ref` with `ref in {"initial","max","previous"}`.

    Parameters
    ----------
    distance_matrix : np.ndarray or pandas.DataFrame
        Square symmetric distance matrix of shape (n, n).
    group : array-like
        Initial group labels (length n). Can be numeric or categorical.
    weights : np.ndarray, optional
        Optional weights for each observation.
    target_n_groups : int, optional
        Compatibility alias for `min_group`: if provided, overrides `min_group`.
    squared : bool, default False
        Whether to square distances before analysis (passed to
        `single_factor_association`).

    measure : str, default "ASW"
        Cluster quality measure. Currently only "ASW" is supported.
    crit : float, default 0.2
        Maximum allowed proportion of quality loss.
    ref : {"initial", "max", "previous"}, default "max"
        Reference quality used in the deterioration threshold.
    min_group : int, default 4
        Minimal number of final groups.
    small : float, default 0.05
        If <1, interpreted as proportion of weighted sample size. While the
        smallest group is below this threshold, only merges involving that
        smallest group are evaluated.
    silent : bool, default False
        If False, merge steps are printed.

    Returns
    -------
    dict
        - 'final_group': final merged grouping as integer codes (1..K)
        - 'quality': final quality value
        - 'history': merge log
    """
    if isinstance(distance_matrix, pd.DataFrame):
        distance_matrix = distance_matrix.values
    D = np.asarray(distance_matrix, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("diss must be a square distance matrix")

    n = D.shape[0]
    g = pd.Categorical(pd.Series(group)).codes.astype(int) + 1
    if len(g) != n or np.any(g <= 0):
        raise ValueError("group must be valid and conformable with distance matrix")

    if weights is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if len(w) != n:
            raise ValueError("weights length must match distance matrix size")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")

    if squared:
        D = D ** 2

    if target_n_groups is not None:
        min_group = int(target_n_groups)

    if min_group < 1:
        raise ValueError("min_group must be >= 1")

    if measure.upper() != "ASW":
        raise ValueError("Only measure='ASW' is currently supported")

    if ref not in {"initial", "max", "previous"}:
        raise ValueError("ref must be one of {'initial', 'max', 'previous'}")

    def _asw(labels: np.ndarray) -> float:
        unique = np.unique(labels)
        if len(unique) <= 1:
            return 0.0
        s = np.zeros(n, dtype=float)
        for i in range(n):
            gi = labels[i]
            in_g = np.where(labels == gi)[0]
            in_g_other = in_g[in_g != i]
            if len(in_g_other) == 0:
                a_i = 0.0
            else:
                ww = w[in_g_other]
                denom = float(np.sum(ww))
                a_i = float(np.sum(ww * D[i, in_g_other]) / denom) if denom > 0 else 0.0

            b_i = np.inf
            for gj in unique:
                if gj == gi:
                    continue
                out_g = np.where(labels == gj)[0]
                ww = w[out_g]
                denom = float(np.sum(ww))
                if denom <= 0:
                    continue
                cand = float(np.sum(ww * D[i, out_g]) / denom)
                if cand < b_i:
                    b_i = cand
            if not np.isfinite(b_i):
                b_i = 0.0
            den = max(a_i, b_i)
            s[i] = 0.0 if den <= 0 else (b_i - a_i) / den
        return float(np.average(s, weights=w))

    N = float(np.sum(w))
    minsize = small * N if small < 1 else float(small)

    history: List[Dict[str, Any]] = []
    quality = _asw(g)
    quality_ref = quality
    final_quality = quality

    while int(np.max(g)) > min_group:
        maxgn = int(np.max(g))
        grp_sizes = np.bincount(g, weights=w, minlength=maxgn + 1)[1:]
        diff = quality_ref
        best_pair = None
        best_qual = None

        if np.min(grp_sizes) > minsize:
            pairs = [(i, j) for i in range(1, maxgn) for j in range(i + 1, maxgn + 1)]
        else:
            i = int(np.argmin(grp_sizes)) + 1
            pairs = [(min(i, j), max(i, j)) for j in range(1, maxgn + 1) if j != i]

        for i, j in pairs:
            gng = g.copy()
            gng[gng == j] = i
            gng = pd.Categorical(gng).codes.astype(int) + 1
            if len(np.unique(gng)) < 2:
                continue
            qual = _asw(gng)
            loss = quality_ref - qual
            if loss < diff:
                diff = loss
                best_pair = (i, j)
                best_qual = qual

        if best_pair is None or diff > crit * quality_ref:
            break

        i, j = best_pair
        if not silent:
            print(f"Merging groups {i} and {j}")
        g[g == j] = i
        g = pd.Categorical(g).codes.astype(int) + 1

        final_quality = float(best_qual if best_qual is not None else _asw(g))
        history.append(
            {
                "merged": (i, j),
                "quality": final_quality,
                "loss": float(diff),
                "n_groups": int(np.max(g)),
            }
        )

        if ref == "max":
            quality_ref = max(quality_ref, final_quality)
        elif ref == "previous":
            quality_ref = final_quality
        else:  # initial
            quality_ref = quality

    return {
        "final_group": pd.Series(g, name="group"),
        "quality": final_quality,
        "history": history,
    }
