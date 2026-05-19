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
    opts.setdefault("full_matrix", True)
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


def aggregate_domains(
    domains: List[SequenceData],
    aggregation: Dict[str, Any],
) -> List[SequenceData]:
    """Subset each domain to the aggregated unique-case indices."""
    agg_idx = np.asarray(aggregation["aggIndex"], dtype=int) - 1
    return [subset_sequence_data(domain, agg_idx) for domain in domains]


def warn_large_expanded_alphabet(seqdata: SequenceData, threshold: int = 500) -> None:
    """Warn when the observed expanded alphabet is very large."""
    n_states = len(seqdata.states)
    if n_states >= threshold:
        import warnings

        warnings.warn(
            f"The multidomain expanded alphabet has {n_states} observed states "
            f"(threshold={threshold}). IDCD distances may be slow or memory-intensive.",
            UserWarning,
            stacklevel=3,
        )


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
    "aggregate_domains",
    "warn_large_expanded_alphabet",
    "check_sample_size_for_k",
    "validate_multidomain_domains",
    "assert_sample_distance_shape",
    "assert_distance_to_medoids_shape",
]
