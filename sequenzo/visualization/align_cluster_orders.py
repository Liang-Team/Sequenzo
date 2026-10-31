"""
@Author  : Yuqi Liang 梁彧祺
@File    : align_cluster_orders.py
@Desc    :
    Reorder cluster / medoid panels so that similar representative sequences
    occupy the same column across methods or partitions.

    Typical use with multi-row medoid comparison figures::

        orders = align_cluster_orders(medoid_mats, mode="progressive")
        # orders[i] is a permutation: display column j shows original cluster orders[i][j]

    Matching uses the Hungarian algorithm on pairwise Hamming (mismatch)
    distances between medoid state sequences.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
from scipy.optimize import linear_sum_assignment


ArrayLike = Union[np.ndarray, Sequence[np.ndarray]]


def _as_medoid_matrix(medoids: ArrayLike) -> np.ndarray:
    """Stack medoid sequences into shape ``(k, T)`` (integer / float states)."""
    arr = np.asarray(medoids)
    if arr.ndim == 1:
        # Sequence of 1d arrays of unequal type → stack manually
        rows = [np.asarray(m).reshape(-1) for m in medoids]
        if not rows:
            raise ValueError("medoids must contain at least one sequence.")
        T = rows[0].size
        if any(r.size != T for r in rows):
            raise ValueError("All medoid sequences in a partition must share length T.")
        arr = np.vstack(rows)
    elif arr.ndim != 2:
        raise ValueError(
            f"Each partition must be a (k, T) array or a sequence of length-T "
            f"vectors; got ndim={arr.ndim}."
        )
    if arr.shape[0] < 1:
        raise ValueError("Each partition must have at least one medoid.")
    return arr


def hamming_distance_matrix(
    left: np.ndarray,
    right: np.ndarray,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """
    Pairwise Hamming / mismatch distances between two sets of sequences.

    Parameters
    ----------
    left, right :
        Arrays of shape ``(k_left, T)`` and ``(k_right, T)``.
    normalize :
        If True (default), divide by ``T`` so entries lie in ``[0, 1]``.

    Returns
    -------
    np.ndarray
        Cost matrix of shape ``(k_left, k_right)``.
    """
    a = np.asarray(left)
    b = np.asarray(right)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("left and right must be 2D (k, T) arrays.")
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"Sequence lengths differ: left T={a.shape[1]}, right T={b.shape[1]}."
        )
    # Broadcast: (k_a, 1, T) vs (1, k_b, T)
    mismatches = np.sum(a[:, None, :] != b[None, :, :], axis=2).astype(float)
    if normalize and a.shape[1] > 0:
        mismatches /= float(a.shape[1])
    return mismatches


def _leaf_order_from_linkage(dist: np.ndarray) -> np.ndarray:
    """Seriate ``k`` items via average-linkage leaf order (similar items adjacent)."""
    k = dist.shape[0]
    if k <= 2:
        return np.arange(k, dtype=int)
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    # Ensure zeros on diagonal for squareform
    d = np.asarray(dist, dtype=float).copy()
    np.fill_diagonal(d, 0.0)
    condensed = squareform(d, checks=False)
    Z = linkage(condensed, method="average")
    return np.asarray(leaves_list(Z), dtype=int)


def match_clusters_to_reference(
    reference: np.ndarray,
    other: np.ndarray,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """
    Optimal assignment of ``other`` rows onto ``reference`` columns.

    Returns
    -------
    np.ndarray
        Permutation ``p`` of length ``k_other`` such that display column ``j``
        should show ``other[p[j]]``. When ``k_other == k_ref``, this is a
        bijection minimising total Hamming cost. When sizes differ, only
        ``min(k_ref, k_other)`` columns are matched and remaining indices of
        ``other`` are appended in original order.
    """
    ref = _as_medoid_matrix(reference)
    oth = _as_medoid_matrix(other)
    cost = hamming_distance_matrix(ref, oth, normalize=normalize)
    k_ref, k_oth = cost.shape
    # Pad rectangular costs so Hungarian can run (large penalty for dummy cells).
    n = max(k_ref, k_oth)
    pad = np.full((n, n), fill_value=cost.max() + 1.0 if cost.size else 1.0)
    pad[:k_ref, :k_oth] = cost
    row_ind, col_ind = linear_sum_assignment(pad)
    # Map reference column j → other row
    assignment = -np.ones(k_ref, dtype=int)
    used = set()
    for r, c in zip(row_ind, col_ind):
        if r < k_ref and c < k_oth:
            assignment[r] = c
            used.add(int(c))
    # Build permutation of other indices in reference-column order, then leftovers
    ordered: List[int] = []
    for j in range(k_ref):
        if assignment[j] >= 0:
            ordered.append(int(assignment[j]))
    for i in range(k_oth):
        if i not in used:
            ordered.append(i)
    return np.asarray(ordered, dtype=int)


def align_cluster_orders(
    partitions: Sequence[ArrayLike],
    *,
    reference: Union[int, str] = 0,
    mode: str = "to_reference",
    sort_reference: bool = True,
    normalize: bool = True,
) -> List[np.ndarray]:
    """
    Align cluster / medoid orders across several partitions for comparative plots.

    Parameters
    ----------
    partitions :
        Sequence of medoid sets. Each entry is a ``(k, T)`` array or a sequence
        of length-``T`` state vectors (one per cluster).
    reference :
        Index of the reference partition when ``mode="to_reference"`` (default
        ``0``). Ignored for ``mode="progressive"`` (which starts from partition 0).
    mode :
        - ``"to_reference"``: match every partition to the (optionally seriated)
          reference.
        - ``"progressive"``: match partition ``i`` to the already-aligned
          partition ``i-1`` (useful when consecutive methods are most comparable).
    sort_reference :
        If True, reorder the reference itself so that similar medoids sit in
        adjacent columns (average-linkage leaf order). Other partitions then
        match that seriated layout.
    normalize :
        Pass-through to :func:`hamming_distance_matrix`.

    Returns
    -------
    list of np.ndarray
        One permutation per partition. Display column ``j`` should show the
        original cluster ``orders[i][j]`` of partition ``i``.
    """
    if not partitions:
        raise ValueError("partitions must be non-empty.")
    mats = [_as_medoid_matrix(p) for p in partitions]
    n = len(mats)

    if mode not in {"to_reference", "progressive"}:
        raise ValueError(f"Unknown mode={mode!r}; use 'to_reference' or 'progressive'.")

    if isinstance(reference, str):
        if reference not in {"first", "0"}:
            raise ValueError("reference string must be 'first' (or use an int index).")
        ref_idx = 0
    else:
        ref_idx = int(reference)
    if not (0 <= ref_idx < n):
        raise ValueError(f"reference index {ref_idx} out of range for {n} partitions.")

    orders: List[Optional[np.ndarray]] = [None] * n

    # Seriate reference so similar clusters are neighbouring columns
    ref_mat = mats[ref_idx]
    if sort_reference and ref_mat.shape[0] > 1:
        d_ref = hamming_distance_matrix(ref_mat, ref_mat, normalize=normalize)
        ref_order = _leaf_order_from_linkage(d_ref)
    else:
        ref_order = np.arange(ref_mat.shape[0], dtype=int)
    orders[ref_idx] = ref_order
    aligned_ref = ref_mat[ref_order]

    if mode == "to_reference":
        for i in range(n):
            if i == ref_idx:
                continue
            # Match other onto aligned reference columns
            p = match_clusters_to_reference(aligned_ref, mats[i], normalize=normalize)
            orders[i] = p
    else:  # progressive: walk from reference outward in index order
        # Align lower indices walking left, then higher walking right
        # First bring everything relative to ref via progressive chain from ref
        # Left side
        prev = aligned_ref
        for i in range(ref_idx - 1, -1, -1):
            p = match_clusters_to_reference(prev, mats[i], normalize=normalize)
            orders[i] = p
            prev = mats[i][p]
        # Right side
        prev = aligned_ref
        for i in range(ref_idx + 1, n):
            p = match_clusters_to_reference(prev, mats[i], normalize=normalize)
            orders[i] = p
            prev = mats[i][p]

    return [np.asarray(o, dtype=int) for o in orders]  # type: ignore[misc]


def apply_cluster_order(
    medoids,
    order: Sequence[int],
    *,
    cluster_col: str = "Cluster",
    renumber: bool = True,
):
    """
    Reorder a medoid summary table (pandas DataFrame) by a permutation.

    Parameters
    ----------
    medoids :
        DataFrame with one row per cluster (already sorted or not).
    order :
        Permutation of row positions after sorting by ``cluster_col``.
    cluster_col :
        Column used to establish the original cluster order before applying
        ``order``.
    renumber :
        If True, rewrite ``cluster_col`` to ``1..k`` in display order so panel
        titles read C1…Ck left-to-right after alignment.
    """
    import pandas as pd

    if not isinstance(medoids, pd.DataFrame):
        raise TypeError("medoids must be a pandas DataFrame.")
    df = medoids.sort_values(cluster_col).reset_index(drop=True)
    order_arr = np.asarray(order, dtype=int)
    if order_arr.size != len(df):
        raise ValueError(
            f"order length {order_arr.size} does not match n_clusters={len(df)}."
        )
    out = df.iloc[order_arr].reset_index(drop=True)
    if renumber:
        out = out.copy()
        out[cluster_col] = np.arange(1, len(out) + 1)
    return out
