"""
@Author  : Yuqi Liang 梁彧祺
@File    : _utils.py
@Time    : 18/05/2026 13:40
@Desc    : 
Shared helpers for multidomain CLARA (subset sequences, distance calls).
"""

from __future__ import annotations

import os
from contextlib import redirect_stdout
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from sequenzo.define_sequence_data import SequenceData
from sequenzo.dissimilarity_measures.get_distance_matrix import get_distance_matrix
from sequenzo.multidomain.idcd import validate_multidomain_domains


def validate_domain_weights(domains: List[SequenceData]) -> Optional[np.ndarray]:
    """
    Return aligned case weights from domain 0, or ``None`` when unweighted.

    Raises if some domains carry weights and others do not, or if weights differ.
    """
    if not domains:
        raise ValueError("domains must contain at least one SequenceData object.")

    reference_weights = getattr(domains[0], "weights", None)

    for domain_index, domain in enumerate(domains[1:], start=1):
        domain_weights = getattr(domain, "weights", None)

        if reference_weights is None and domain_weights is None:
            continue

        if reference_weights is None or domain_weights is None:
            raise ValueError(
                "Either provide case weights for every domain or for none of them."
            )

        if not np.allclose(
            np.asarray(reference_weights, dtype=float),
            np.asarray(domain_weights, dtype=float),
        ):
            raise ValueError(
                f"Case weights in domain {domain_index} do not match domain 0."
            )

    if reference_weights is None:
        return None

    weights = np.asarray(reference_weights, dtype=float)
    if weights.ndim != 1:
        raise ValueError("Case weights must be one-dimensional.")
    if len(weights) != len(domains[0].data):
        raise ValueError("Case weights must match the number of cases.")
    if not np.all(np.isfinite(weights)):
        raise ValueError("Case weights must contain only finite values.")
    if np.any(weights < 0):
        raise ValueError("Case weights must be non-negative.")
    if float(np.sum(weights)) <= 0:
        raise ValueError("At least one case weight must be positive.")

    return weights


def one_based_to_zero_based(
    values: Sequence[int],
    *,
    name: str,
) -> np.ndarray:
    """Convert Sequenzo aggregation indices from one-based to zero-based."""
    arr = np.asarray(values, dtype=int)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if arr.size and np.any(arr < 1):
        raise ValueError(
            f"{name} is expected to use one-based indexing, "
            "but it contains values below 1."
        )

    return arr - 1


def _validate_distance_matrix_values(
    matrix: np.ndarray,
    *,
    label: str,
    check_symmetry: bool = False,
) -> None:
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} contains NaN or infinite values.")
    if np.any(matrix < -1e-12):
        raise ValueError(f"{label} contains negative values.")
    if check_symmetry:
        if not np.allclose(matrix, matrix.T):
            raise ValueError(f"{label} must be symmetric.")
        if not np.allclose(np.diag(matrix), 0):
            raise ValueError(f"{label} must have a zero diagonal.")

try:
    from joblib import Parallel, delayed
except ImportError:  # pragma: no cover
    Parallel = None  # type: ignore[misc, assignment]
    delayed = None  # type: ignore[misc, assignment]


def parallel_map(
    func,
    items: Sequence[Any],
    *,
    n_jobs: int = 1,
) -> List[Any]:
    """
    Run ``func(item)`` over ``items``, optionally in parallel via joblib.

    Defaults to serial execution (``n_jobs=1``) so callers can nest safely
    inside outer ``Parallel`` loops (e.g. CLARA iterations).
    """
    if not items:
        return []
    if n_jobs == 1 or len(items) == 1:
        return [func(item) for item in items]
    if Parallel is None or delayed is None:
        return [func(item) for item in items]
    return Parallel(n_jobs=n_jobs)(delayed(func)(item) for item in items)


def assert_sample_distance_shape(
    matrix: np.ndarray,
    sample_indices: Sequence[int],
) -> np.ndarray:
    """Validate an on-sample distance matrix is square with the right order."""
    matrix = np.asarray(matrix, dtype=float)
    n = len(sample_indices)
    expected = (n, n)
    if matrix.shape != expected:
        raise ValueError(
            f"Expected sample_distance_matrix shape {expected}, got {matrix.shape}."
        )
    _validate_distance_matrix_values(
        matrix,
        label="Sample distance matrix",
        check_symmetry=True,
    )
    return matrix


def assert_distance_to_medoids_shape(
    matrix: np.ndarray,
    n_sequences: int,
    medoid_indices: Sequence[int],
) -> np.ndarray:
    """Validate distances from all sequences to K medoids (N x K)."""
    matrix = np.asarray(matrix, dtype=float)
    expected = (n_sequences, len(medoid_indices))
    if matrix.shape != expected:
        raise ValueError(
            f"Expected distance_to_medoids shape {expected}, got {matrix.shape}. "
            "Check get_distance_matrix(refseq=[all_indices, medoid_indices]) orientation."
        )
    _validate_distance_matrix_values(matrix, label="Distance-to-medoid matrix")
    return matrix


def subset_sequence_data(seqdata: SequenceData, indices: Sequence[int]) -> SequenceData:
    """Return a SequenceData view containing only the requested row indices."""
    idx = np.asarray(indices, dtype=int)
    id_col = seqdata.id_col
    columns = ([id_col] if id_col else []) + list(seqdata.time)
    sub_df = seqdata.data.iloc[idx][columns].reset_index(drop=True)

    kwargs: Dict[str, Any] = {
        "data": sub_df,
        "time": seqdata.time,
        "states": seqdata.states,
        "labels": seqdata.labels,
    }
    if id_col:
        kwargs["id_col"] = id_col
    if getattr(seqdata, "weights", None) is not None:
        kwargs["weights"] = np.asarray(seqdata.weights)[idx]
    if getattr(seqdata, "void", None) is not None:
        kwargs["void"] = seqdata.void

    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull):
            return SequenceData(**kwargs)


def compute_distance_matrix(
    seqdata: SequenceData,
    dist_args: Dict[str, Any],
    *,
    refseq: Optional[Union[int, List[List[int]]]] = None,
) -> np.ndarray:
    """Call get_distance_matrix with stdout suppressed and return a NumPy array."""
    opts = dict(dist_args)
    opts["seqdata"] = seqdata
    opts["full_matrix"] = True
    if refseq is not None:
        opts["refseq"] = refseq

    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull):
            result = get_distance_matrix(opts=opts)

    if isinstance(result, pd.DataFrame):
        return result.to_numpy(dtype=float)
    if isinstance(result, pd.Series):
        return result.to_numpy(dtype=float).reshape(-1, 1)
    return np.asarray(result, dtype=float)


def build_multidomain_profile_frame(
    domains: List[SequenceData],
) -> pd.DataFrame:
    """
    Construct one complete multidomain trajectory signature per case.

    Two cases are treated as duplicates only when they have identical
    sequences in every domain and at every time position.
    """
    validate_multidomain_domains(domains)

    blocks = []

    for domain_index, domain in enumerate(domains):
        time_cols = list(domain.time)

        block = (
            domain.data.loc[:, time_cols]
            .reset_index(drop=True)
            .copy()
        )

        block.columns = [
            f"domain_{domain_index}__{time_col}"
            for time_col in time_cols
        ]

        blocks.append(block)

    return pd.concat(blocks, axis=1)


def aggregate_domains(
    domains: List[SequenceData],
    aggregation: Dict[str, Any],
) -> List[SequenceData]:
    """Subset each domain to the aggregated unique-case indices."""
    agg_idx = one_based_to_zero_based(aggregation["aggIndex"], name="aggIndex")
    return [subset_sequence_data(domain, agg_idx) for domain in domains]


def warn_large_combined_state_space(seqdata: SequenceData, threshold: int = 500) -> None:
    """Warn when the observed multidomain combined state space is very large."""
    n_states = len(seqdata.states)
    if n_states >= threshold:
        import warnings

        warnings.warn(
            f"The observed multidomain state space contains {n_states} combined states "
            f"(threshold={threshold}). IDCD distances may be slow or memory-intensive.",
            UserWarning,
            stacklevel=3,
        )


def warn_large_expanded_alphabet(seqdata: SequenceData, threshold: int = 500) -> None:
    """Backward-compatible alias for :func:`warn_large_combined_state_space`."""
    warn_large_combined_state_space(seqdata, threshold=threshold)


def check_sample_size_for_k(
    sample_size: int,
    kvals: Sequence[int],
    *,
    n_unique_cases: Optional[int] = None,
) -> None:
    """
    Validate CLARA design parameters before iterations start.

    Raises if k < 2, if the draw size is smaller than max(k), or if the number
    of unique aggregated cases in the full data is smaller than max(k).
    """
    if not kvals:
        raise ValueError("kvals must contain at least one cluster size.")

    min_k = min(kvals)
    max_k = max(kvals)

    if min_k < 2:
        raise ValueError(
            f"kvals must contain values >= 2 (got min={min_k}). "
            "Cluster-quality indices (AMS, XB, DB) require at least two clusters."
        )

    if sample_size < max_k:
        raise ValueError(
            f"sample_size ({sample_size}) must be at least max(kvals) ({max_k})."
        )

    if n_unique_cases is not None and n_unique_cases < max_k:
        raise ValueError(
            f"Only {n_unique_cases} unique aggregated cases in the data, but "
            f"max(kvals)={max_k}. Reduce k or use a dataset with more distinct sequences."
        )

    if sample_size < 5 * max_k:
        import warnings

        warnings.warn(
            f"sample_size ({sample_size}) is small for k up to {max_k}. "
            "Consider increasing sample_size for more stable medoids.",
            UserWarning,
            stacklevel=3,
        )


__all__ = [
    "subset_sequence_data",
    "compute_distance_matrix",
    "parallel_map",
    "build_multidomain_profile_frame",
    "aggregate_domains",
    "one_based_to_zero_based",
    "warn_large_combined_state_space",
    "warn_large_expanded_alphabet",
    "check_sample_size_for_k",
    "validate_multidomain_domains",
    "validate_domain_weights",
    "assert_sample_distance_shape",
    "assert_distance_to_medoids_shape",
]
