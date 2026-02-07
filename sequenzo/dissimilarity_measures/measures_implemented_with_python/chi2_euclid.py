"""
@Author  : Yuqi Liang 梁彧祺
@File    : chi2_euclid.py
@Time    : 2026/02/05 22:27
@Desc    : 
CHI2 and EUCLID sequence distances (TraMineR-aligned).

Implements the Chi-square and Euclidean distance between state distributions
over time windows. Logic and formulas follow TraMineR's seqdist-CHI2.R and
chisq.cpp so that Sequenzo results match TraMineR.

References:
  - TraMineR R: seqdist-CHI2.R, CHI2()
  - TraMineR C++: chisq.cpp, tmrChisq / tmrChisqRef
"""

import numpy as np
from typing import Optional, List, Tuple, Union


def _build_breaks(n_cols: int, step: int, overlap: bool) -> List[Tuple[int, int]]:
    """
    Build list of (start, end) column indices for each time window (0-based inclusive).
    Mirrors TraMineR CHI2() when breaks is NULL (R uses 1-based column indices).

    :param n_cols: Number of sequence columns (time points).
    :param step: Window step (positive integer).
    :param overlap: Whether to use overlapping windows (step must be even in TraMineR).
    :return: List of (start, end) 0-based inclusive.
    """
    breaks = []
    if step == 1:
        for i in range(n_cols):
            breaks.append((i, i))
    elif step == n_cols:
        breaks.append((0, n_cols - 1))
    else:
        # R: bb <- seq(from=1, to=ncol(seqdata), by=step); bb <- c(bb, ncol(seqdata)+1)
        bb = list(range(0, n_cols, step))
        if len(bb) > 0 and bb[-1] != n_cols - step:
            pass  # optional: warn last episode shorter
        bb.append(n_cols)
        bi = 0
        if overlap:
            # R: breaks[[1]] <- c(1, 1+step/2) -> 0-based (0, step//2)
            breaks.append((0, step // 2))
            bi += 1
        for i in range(1, len(bb)):
            # R: breaks[[bi]] <- c(bb[i-1], bb[i]-1) -> 0-based (bb[i-1], bb[i]-1)
            start = bb[i - 1]
            end = bb[i] - 1
            breaks.append((start, end))
            bi += 1
            if overlap and bi < 100:
                # R: breaks[[bi]] <- pmin(breaks[[bi-1]]+step/2, ncol(seqdata))
                prev_start, prev_end = breaks[-1]
                next_start = prev_start + step // 2
                next_end = min(prev_end + step // 2, n_cols - 1)
                if next_start <= next_end:
                    breaks.append((next_start, next_end))
                    bi += 1
    return breaks


def _dummies_one_break(
    seqdata_mat: np.ndarray,
    weights: np.ndarray,
    alphabet: np.ndarray,
    bindice: Tuple[int, int],
    norm: bool,
    euclid: bool,
    global_pdotj: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For one time window, build (n x nalph) proportion matrix and marginal row pdotj.
    Matches TraMineR dummies() and scaling of the last row.

    :param seqdata_mat: (n, n_cols) matrix of state codes (e.g. 1..nstates).
    :param weights: (n,) weights per sequence.
    :param alphabet: (nalph,) state codes to count.
    :param bindice: (start, end) 0-based inclusive column indices.
    :param norm: Whether normalization is requested (affects maxd for CHI2/EUCLID).
    :param euclid: If True, use EUCLID marginal row; else CHI2.
    :param global_pdotj: If set, use as marginal row for CHI2 (length nalph).
    :return: (mat, pdotj) where mat is (n+1, nalph), pdotj is (nalph,) for this block.
    """
    start, end = bindice
    nalph = len(alphabet)
    n = seqdata_mat.shape[0]
    bseq = seqdata_mat[:, start : end + 1]  # (n, width)

    # Count per state: mat[i, s] = sum(bseq[i,:] == alphabet[s])
    mat = np.zeros((n, nalph), dtype=np.float64)
    for s in range(nalph):
        mat[:, s] = np.sum(bseq == alphabet[s], axis=1)

    # Weighted column sums (marginals): ndot[s] = sum(weights * mat[:, s])
    ndot = np.sum(weights[:, np.newaxis] * mat, axis=0)

    # Append marginal row (will become pdotj for this block)
    mat = np.vstack([mat, ndot])

    # Normalize rows with positive sum to proportions (row sums = 1)
    row_sums = np.sum(mat, axis=1, keepdims=True)
    non0 = (row_sums.ravel() > 0)
    mat[non0] = np.where(row_sums[non0] > 0, mat[non0] / row_sums[non0], 0.0)

    # Last row: set marginal row for distance denominator (TraMineR logic)
    if euclid:
        maxd = 2.0 if norm else 1.0
        mat[-1, :] = maxd
        mat[-1, ndot == 0] = 0.0
    else:
        if global_pdotj is not None:
            mat[-1, :] = global_pdotj
            mat[-1, ndot == 0] = 0.0
        else:
            pdot = mat[-1, ndot != 0]
            if len(pdot) > 1:
                cmin0 = np.min(pdot)
                pdot_rest = pdot[pdot != cmin0]
                cmin1 = np.min(pdot_rest) if len(pdot_rest) > 0 else cmin0
                maxd = (1.0 / cmin0 + 1.0 / cmin1) if norm else 1.0
                mat[-1, :] = mat[-1, :] * maxd
        # else: leave last row as is (single state in window)

    pdotj_block = mat[-1, :].copy()
    mat = mat[:-1, :]  # (n, nalph)
    return mat, pdotj_block


def _seqmeant_proportions(
    seqdata_mat: np.ndarray,
    weights: np.ndarray,
    alphabet: np.ndarray,
) -> np.ndarray:
    """
    Weighted mean state proportions over time (one proportion per state).
    Equivalent to TraMineR seqmeant(seqdata, weighted=TRUE, prop=TRUE).
    """
    n, n_cols = seqdata_mat.shape
    nalph = len(alphabet)
    if weights is None or len(weights) != n:
        weights = np.ones(n, dtype=np.float64)
    wtot = np.sum(weights)
    # Per-sequence proportions (state counts / length)
    iseqtab = np.zeros((n, nalph), dtype=np.float64)
    for s in range(nalph):
        iseqtab[:, s] = np.sum(seqdata_mat == alphabet[s], axis=1)
    row_sums = np.sum(iseqtab, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    iseqtab = iseqtab / row_sums
    # Weighted mean
    mtime = np.sum(weights[:, np.newaxis] * iseqtab, axis=0)
    return mtime / wtot


def build_chi2_allmat_pdotj(
    seqdata_mat: np.ndarray,
    alphabet: np.ndarray,
    weights: Optional[np.ndarray] = None,
    step: int = 1,
    breaks: Optional[List[Tuple[int, int]]] = None,
    overlap: bool = False,
    norm: bool = True,
    euclid: bool = False,
    global_pdotj: Optional[Union[str, np.ndarray]] = None,
    refseq: Optional[Union[int, List[List[int]]]] = None,
) -> dict:
    """
    Build (allmat, pdotj, norm_factor, refseq_id) for CHI2/EUCLID (TraMineR-aligned).
    Used by Python loop or C++ CHI2distance. Returns dict with keys:
    allmat, pdotj, norm_factor, refseq_id (np.int32 shape (2,)), refseq_type, n_total, n1, n2.
    """
    n, n_cols = seqdata_mat.shape
    nalph = len(alphabet)
    alphabet = np.asarray(alphabet, dtype=seqdata_mat.dtype)

    if weights is None or len(weights) != n:
        weights = np.ones(n, dtype=np.float64)
    if euclid:
        weights = np.ones(n, dtype=np.float64)

    if not euclid and global_pdotj is not None:
        if global_pdotj == "obs":
            global_pdotj = _seqmeant_proportions(seqdata_mat, weights, alphabet)
        else:
            global_pdotj = np.asarray(global_pdotj, dtype=np.float64)
            if global_pdotj.size != nalph:
                raise ValueError("global_pdotj length must equal alphabet size.")
            global_pdotj = global_pdotj / np.sum(global_pdotj)
    else:
        global_pdotj = None

    if breaks is None:
        breaks = _build_breaks(n_cols, step, overlap)
    else:
        breaks = [tuple(b) for b in breaks]

    block_mats = []
    block_pdotj = []
    for (start, end) in breaks:
        mat_b, pdotj_b = _dummies_one_break(
            seqdata_mat, weights, alphabet, (start, end),
            norm=norm, euclid=euclid, global_pdotj=global_pdotj,
        )
        block_mats.append(mat_b)
        block_pdotj.append(pdotj_b)
    allmat = np.hstack(block_mats)
    pdotj = np.concatenate(block_pdotj)

    cond = pdotj > 0
    allmat = np.ascontiguousarray(allmat[:, cond], dtype=np.float64)
    pdotj = np.ascontiguousarray(pdotj[cond], dtype=np.float64)
    n_breaks = len(breaks)
    norm_factor = 1.0 / np.sqrt(n_breaks) if norm else 1.0

    refseq_type = "none"
    n1, n2 = 0, 0
    if isinstance(refseq, list) and len(refseq) >= 2:
        set1, set2 = refseq[0], refseq[1]
        n1, n2 = len(set1), len(set2)
        refseq_id = np.array([n1, n1 + n2], dtype=np.int32)
        allmat = np.vstack([allmat[set1], allmat[set2]])
        n_total = n1 + n2
        refseq_type = "sets"
    elif refseq is not None and not isinstance(refseq, list):
        ref_idx = int(refseq)
        refseq_id = np.array([ref_idx, ref_idx], dtype=np.int32)
        n_total = n
        refseq_type = "index"
    else:
        refseq_id = np.array([0, 0], dtype=np.int32)
        n_total = n

    return {
        "allmat": allmat,
        "pdotj": pdotj,
        "norm_factor": norm_factor,
        "refseq_id": refseq_id,
        "refseq_type": refseq_type,
        "n_total": n_total,
        "n1": n1,
        "n2": n2,
    }


def chi2_euclid_distances(
    seqdata_mat: np.ndarray,
    alphabet: np.ndarray,
    weights: Optional[np.ndarray] = None,
    step: int = 1,
    breaks: Optional[List[Tuple[int, int]]] = None,
    overlap: bool = False,
    norm: bool = True,
    euclid: bool = False,
    global_pdotj: Optional[Union[str, np.ndarray]] = None,
    refseq: Optional[Union[int, List[List[int]]]] = None,
    full_matrix: bool = True,
) -> Union[np.ndarray, object]:
    """
    Compute CHI2 or EUCLID distances between sequences (TraMineR-aligned).

    :param seqdata_mat: (n, n_cols) matrix of state codes (e.g. 1..nstates).
    :param alphabet: (nalph,) state codes used in seqdata_mat (e.g. [1,2,...,nstates]).
    :param weights: (n,) sequence weights; if None, use ones. For EUCLID, TraMineR uses unweighted.
    :param step: Window step when breaks is None.
    :param breaks: List of (start, end) 0-based column indices per window; if None, built from step/overlap.
    :param overlap: Use overlapping windows when building breaks (only if breaks is None).
    :param norm: If True, divide distances by sqrt(n_breaks) (TraMineR norm=TRUE / "auto").
    :param euclid: If True, compute EUCLID (same formula, different marginals); and weights ignored.
    :param global_pdotj: For CHI2 only. None, "obs" (use weighted mean state proportions), or (nalph,) array.
    :param refseq: None (pairwise), single index (0-based), or list of two index lists [set1, set2].
    :param full_matrix: If True and refseq is None, return (n,n) matrix; else return lower-triangle vector.
    :return: Distance matrix (n,n), or (n,) to-ref vector, or (n1,n2) sets matrix; or dist-like vector.
    """
    n, n_cols = seqdata_mat.shape
    nalph = len(alphabet)
    alphabet = np.asarray(alphabet, dtype=seqdata_mat.dtype)

    if weights is None or len(weights) != n:
        weights = np.ones(n, dtype=np.float64)
    if euclid:
        weights = np.ones(n, dtype=np.float64)

    # Resolve global_pdotj for CHI2
    if not euclid and global_pdotj is not None:
        if global_pdotj == "obs":
            global_pdotj = _seqmeant_proportions(seqdata_mat, weights, alphabet)
        else:
            global_pdotj = np.asarray(global_pdotj, dtype=np.float64)
            if global_pdotj.size != nalph:
                raise ValueError("global_pdotj length must equal alphabet size.")
            global_pdotj = global_pdotj / np.sum(global_pdotj)
    else:
        global_pdotj = None

    # Build breaks (0-based inclusive)
    if breaks is None:
        breaks = _build_breaks(n_cols, step, overlap)
    else:
        breaks = [tuple(b) for b in breaks]

    # Build allmat (n x total_cols) and pdotj (total_cols,) with only columns where pdotj > 0
    block_mats = []
    block_pdotj = []
    for (start, end) in breaks:
        mat_b, pdotj_b = _dummies_one_break(
            seqdata_mat, weights, alphabet, (start, end),
            norm=norm, euclid=euclid, global_pdotj=global_pdotj,
        )
        block_mats.append(mat_b)
        block_pdotj.append(pdotj_b)
    allmat = np.hstack(block_mats)   # (n, nalph * n_breaks)
    pdotj = np.concatenate(block_pdotj)

    cond = pdotj > 0
    allmat = allmat[:, cond]
    pdotj = pdotj[cond]
    n_breaks = len(breaks)

    # Handle refseq
    sets = False
    if isinstance(refseq, list) and len(refseq) >= 2:
        set1, set2 = refseq[0], refseq[1]
        n1, n2 = len(set1), len(set2)
        refseq_id = (n1, n1 + n2)
        # TraMineR: allmat <- allmat[c(refseq[[1]],refseq[[2]]),]
        allmat = np.vstack([allmat[set1], allmat[set2]])
        n_total = n1 + n2
        sets = True
    elif refseq is not None and not isinstance(refseq, list):
        ref_idx = int(refseq)
        refseq_id = (ref_idx, ref_idx)
        n_total = n
        sets = False
    else:
        refseq_id = None
        n_total = n

    # Distance formula: d(i,j) = sqrt(sum_c (allmat[i,c]-allmat[j,c])^2 / pdotj[c])
    def dist_ij(i: int, j: int) -> float:
        diff = allmat[i] - allmat[j]
        return np.sqrt(np.sum(np.where(pdotj > 0, diff * diff / pdotj, 0.0)))

    if refseq_id is None:
        # Pairwise: fill lower triangle (row > col)
        n_pairs = n_total * (n_total - 1) // 2
        dd = np.zeros(n_pairs, dtype=np.float64)
        idx = 0
        for i in range(n_total - 1):
            for j in range(i + 1, n_total):
                dd[idx] = dist_ij(i, j)
                idx += 1
    else:
        rseq1, rseq2 = refseq_id[0], refseq_id[1]
        if rseq1 < rseq2:
            # Sets: allmat is [set1 rows, set2 rows]. For each ref row in set2, distance from each set1 row.
            # tmrChisqRef: dist[i + nseq*(rseq-rseq1)] for rseq in [rseq1, rseq2), i in [0, nseq).
            nseq = n1
            na = n1 * n2
            dd = np.zeros(na, dtype=np.float64)
            for rseq in range(n2):
                ref_row = n1 + rseq
                for i in range(n1):
                    dd[i + n1 * rseq] = dist_ij(i, ref_row)
        else:
            # Single reference: refseq_id = (ref, ref). ref_row is 0-based index of reference.
            ref_row = rseq1
            dd = np.zeros(n_total, dtype=np.float64)
            for i in range(n_total):
                dd[i] = dist_ij(i, ref_row)

    if norm:
        dd = dd / np.sqrt(n_breaks)

    if refseq_id is None:
        if full_matrix:
            # Convert lower-triangle vector to full matrix
            result = np.zeros((n_total, n_total), dtype=np.float64)
            idx = 0
            for i in range(n_total - 1):
                for j in range(i + 1, n_total):
                    result[j, i] = dd[idx]
                    result[i, j] = dd[idx]
                    idx += 1
            return result
        return dd
    if isinstance(refseq, list) and len(refseq) >= 2:
        return dd.reshape((n1, n2), order='F')
    return dd
