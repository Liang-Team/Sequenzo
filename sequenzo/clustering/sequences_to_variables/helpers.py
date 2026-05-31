"""
@Author  : Yuqi Liang 梁彧祺
@File    : helpers.py
@Time    : 01/03/2026 23:10
@Desc    :
Utilities for "sequences to variables" (Helske et al. 2024).
"""
import numpy as np


def _validate_positive_integer(value, name):
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def validate_reference_index(reference, k, name="reference"):
    """Validate and normalize an omitted-reference category index."""
    k = _validate_positive_integer(k, "k")
    if not isinstance(reference, (int, np.integer)) or isinstance(reference, bool):
        raise ValueError(f"{name} must be an integer category index")
    reference = int(reference)
    if reference < 0 or reference >= k:
        raise ValueError(f"{name} must be between 0 and {k - 1}; got {reference}")
    return reference


def validate_integer_labels(labels, name="labels"):
    """Validate cluster labels without silently truncating floats or booleans."""
    raw = np.asarray(labels, dtype=object).ravel()
    if any(isinstance(value, (bool, np.bool_)) for value in raw):
        raise ValueError(f"{name} must contain integer cluster labels")
    if not all(isinstance(value, (int, np.integer)) for value in raw):
        raise ValueError(f"{name} must contain integer cluster labels")
    return np.asarray(raw, dtype=int)


def validate_name_sequence(names, expected_length, name):
    """Validate optional user-facing column-name sequences."""
    if names is None:
        return None
    if isinstance(names, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of strings, not a string")
    try:
        names = list(names)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of strings") from exc
    if len(names) != expected_length:
        raise ValueError(f"{name} must have length {expected_length}")
    if not all(isinstance(value, str) for value in names):
        raise ValueError(f"{name} must contain only strings")
    return names


def validate_diss_matrix(diss):
    """Check square distance/dissimilarity matrix conventions."""
    diss = np.asarray(diss, dtype=float)
    if diss.ndim != 2 or diss.shape[0] != diss.shape[1]:
        raise ValueError("diss must be a square matrix")
    if np.any(np.isnan(diss)):
        raise ValueError("NA values in the dissimilarity matrix are not allowed")
    if np.any(~np.isfinite(diss)):
        raise ValueError("diss must contain only finite dissimilarities")
    if np.any(diss < 0):
        raise ValueError("diss must contain nonnegative dissimilarities")
    if not np.allclose(diss, diss.T):
        raise ValueError("diss must be symmetric")
    if not np.allclose(np.diag(diss), 0.0):
        raise ValueError("diss must have zeros on the diagonal")
    return diss


def validate_membership_matrix(U, name="U"):
    """Check row-stochastic fuzzy membership matrix."""
    U = np.asarray(U, dtype=float)
    if U.ndim != 2:
        raise ValueError(f"{name} must be a 2D membership matrix")
    if np.any(~np.isfinite(U)):
        raise ValueError(f"{name} must contain only finite membership probabilities")
    if np.any(U < 0):
        raise ValueError(f"{name} must contain nonnegative membership probabilities")
    if not np.allclose(U.sum(axis=1), 1.0):
        raise ValueError(f"Rows of {name} must sum to 1")
    return U


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
        if np.any(np.isnan(diss)):
            raise ValueError("diss must not contain NA values")
        if np.any(~np.isfinite(diss)):
            raise ValueError("diss must contain only finite dissimilarities")
        if np.any(diss < 0):
            raise ValueError("diss must contain nonnegative dissimilarities")
        from scipy.spatial.distance import squareform
        diss = squareform(diss)
    else:
        diss = validate_diss_matrix(diss)
    if diss.ndim != 2 or diss.shape[0] != diss.shape[1]:
        raise ValueError("diss must be a square matrix or condensed distance vector")
    n = diss.shape[0]
    if n < 2:
        return 0.0
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
        Shape (n, K-1). Column j is 1 when the observation belongs to the j-th
        retained category after sorting unique labels and dropping the reference
        category, 0 otherwise.
    """
    labels = validate_integer_labels(labels)
    uniq = np.unique(labels)
    if k is None:
        k = len(uniq)
    else:
        k = _validate_positive_integer(k, "k")
    if len(uniq) != k:
        raise ValueError(f"Number of unique labels ({len(uniq)}) does not match k={k}")

    # Map labels to 0-based indices 0 .. K-1 (by sorted unique)
    label_to_idx = {u: i for i, u in enumerate(uniq)}
    idx = np.array([label_to_idx[l] for l in labels])

    ref_idx = validate_reference_index(reference, k)
    col_indices = [i for i in range(k) if i != ref_idx]
    n = len(labels)
    out = np.zeros((n, k - 1), dtype=float)
    for j, c in enumerate(col_indices):
        out[:, j] = (idx == c).astype(float)
    return out


def dummy_column_names(labels, k=None, reference=0, prefix="C"):
    """Column names for omitted-reference dummy encoding."""
    labels = validate_integer_labels(labels)
    categories = np.sort(np.unique(labels))
    if k is None:
        k = len(categories)
    else:
        k = _validate_positive_integer(k, "k")
    if len(categories) != k:
        raise ValueError(f"Number of unique labels ({len(categories)}) does not match k={k}")
    reference = validate_reference_index(reference, k)
    col_indices = [i for i in range(k) if i != reference]
    return [f"{prefix}_{categories[c]}" for c in col_indices]


def medoid_indices_from_kmedoids_result(
    assigned_medoid_indices: np.ndarray,
    input_base: int = 1,
) -> np.ndarray:
    """
    Sorted medoid row indices from a :func:`KMedoids` return vector.

    ``KMedoids`` assigns each observation the **1-based row index** of its
    cluster medoid (WeightedCluster convention).

    Parameters
    ----------
    assigned_medoid_indices : np.ndarray of int
        Return value of :func:`KMedoids` (medoid row index per observation).
    input_base : {0, 1}, default 1
        Indexing base of ``assigned_medoid_indices``. Use ``1`` for raw
        :func:`KMedoids` output; use ``0`` if indices are already 0-based.

    Returns
    -------
    np.ndarray of shape (K,)
        Sorted 0-based medoid row indices.
    """
    assigned_medoid_indices = validate_integer_labels(
        assigned_medoid_indices,
        name="assigned_medoid_indices",
    )
    if assigned_medoid_indices.size == 0:
        return np.array([], dtype=int)
    medoids = np.unique(assigned_medoid_indices)
    if input_base == 1:
        medoids = medoids - 1
    elif input_base != 0:
        raise ValueError("input_base must be 0 or 1")
    if np.any(medoids < 0):
        raise ValueError("Converted medoid indices contain negative values")
    return np.sort(medoids)


def cluster_labels_from_kmedoids_result(
    assigned_medoid_indices: np.ndarray,
    input_base: int = 1,
) -> np.ndarray:
    """
    0-based cluster labels from a :func:`KMedoids` return vector.

    Parameters
    ----------
    assigned_medoid_indices : np.ndarray of int
        Return value of :func:`KMedoids` (medoid row indices per row).
    input_base : {0, 1}, default 1
        Indexing base of ``assigned_medoid_indices``. Use ``1`` for raw
        :func:`KMedoids` output; use ``0`` if indices are already 0-based.

    Returns
    -------
    np.ndarray of shape (n,)
        Cluster labels ``0 .. K-1`` (ordered by medoid index).
    """
    assigned_medoid_indices = validate_integer_labels(
        assigned_medoid_indices,
        name="assigned_medoid_indices",
    )
    if assigned_medoid_indices.size == 0:
        return assigned_medoid_indices
    memb = assigned_medoid_indices.copy()
    if input_base == 1:
        memb = memb - 1
    elif input_base != 0:
        raise ValueError("input_base must be 0 or 1")
    if np.any(memb < 0):
        raise ValueError("Converted medoid indices contain negative values")
    medoids = np.sort(np.unique(memb))
    return np.searchsorted(medoids, memb)
