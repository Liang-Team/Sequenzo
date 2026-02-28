"""
@Author  : Yuqi Liang 梁彧祺
@File    : seqs2vars_utils.py
@Time    : 01/03/2026 00:00
@Desc    :
Utilities for "sequences to variables" (Helske et al. 2024).
Used by sequences_to_variables and related helpers.
"""
import numpy as np


def max_distance(diss):
    """
    Maximum distance between any two sequences (Helske et al. 2024: "maximum distance between two sequences").

    Used for representativeness: R_i^k = 1 - d(i,k) / d_max.

    Parameters
    ----------
    diss : np.ndarray or array-like
        n x n distance/dissimilarity matrix (symmetric). If scipy.spatial.distance
        condensed form is passed, it will be converted to square form first.

    Returns
    -------
    float
        Maximum off-diagonal distance. If diss is condensed, max over all pairs.
    """
    diss = np.asarray(diss, dtype=float)
    if diss.ndim == 1:
        from scipy.spatial.distance import squareform
        diss = squareform(diss)
    if diss.ndim != 2 or diss.shape[0] != diss.shape[1]:
        raise ValueError("diss must be a square matrix or condensed distance vector")
    # Upper triangle (excluding diagonal) to avoid double count; diagonal is 0
    n = diss.shape[0]
    triu = np.triu_indices(n, k=1)
    d_max = np.max(diss[triu])
    return float(d_max)


def cluster_labels_to_dummies(labels, k=None, reference=0):
    """
    Convert cluster labels to dummy (one-hot) matrix for regression, with one reference category omitted.

    Helske et al. (2024) Table 1: Hard classification uses "Cluster membership" as "Dummies";
    "one cluster is typically chosen as a reference, and the respective (dummy or probability) variable is omitted."

    Parameters
    ----------
    labels : array-like of int
        Cluster assignment per observation, length n. Can be 0-based (0 .. K-1) or 1-based (1 .. K).
        If 1-based, reference is interpreted relative to min(labels)..max(labels).
    k : int, optional
        Number of clusters. If None, inferred as len(np.unique(labels)).
    reference : int, optional
        Index of the reference category to omit (0 = first category in sorted order).
        The returned columns correspond to the other K-1 categories.

    Returns
    -------
    np.ndarray
        Shape (n, K-1). Column j is 1 when the observation belongs to the (j+1)-th non-reference
        category (in sorted order), 0 otherwise.
    """
    labels = np.asarray(labels, dtype=int).ravel()
    uniq = np.unique(labels)
    if k is None:
        k = len(uniq)
    if len(uniq) != k:
        raise ValueError(f"Number of unique labels ({len(uniq)}) does not match k={k}")

    # Map labels to 0-based indices 0 .. K-1 (by sorted unique)
    label_to_idx = {u: i for i, u in enumerate(uniq)}
    idx = np.array([label_to_idx[l] for l in labels])

    # Reference in 0-based index
    ref_idx = min(reference, k - 1)
    # Columns: all except ref_idx
    col_indices = [i for i in range(k) if i != ref_idx]
    n = len(labels)
    out = np.zeros((n, k - 1), dtype=float)
    for j, c in enumerate(col_indices):
        out[:, j] = (idx == c).astype(float)
    return out
