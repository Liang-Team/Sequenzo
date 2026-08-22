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
from sequenzo.dissimilarity_measures.get_distance_matrix import (
    get_distance_matrix,
    warm_unique_sequence_cache,
)
from sequenzo.multidomain.idcd import validate_multidomain_domains

def validate_domain_weights(domains: List[SequenceData]) -> Optional[np.ndarray]:
    """
    Return aligned case weights from domain 0.

    ``SequenceData`` always stores a weight vector (default all ones when the user
    did not pass ``weights``). This helper returns ``None`` only when domain 0 has
    no ``weights`` attribute at all.

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
    if len(items) == 0:
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


def assert_condensed_distance_shape(
    vector: np.ndarray,
    sample_indices: Sequence[int],
) -> np.ndarray:
    """Validate a condensed within-sample distance vector (SciPy pdist order)."""
    vector = np.asarray(vector, dtype=float).reshape(-1)
    n = len(sample_indices)
    expected = n * (n - 1) // 2
    if vector.size != expected:
        raise ValueError(
            f"Expected condensed distance length {expected} for {n} samples, "
            f"got {vector.size}."
        )
    _validate_distance_matrix_values(vector, label="Condensed distances")
    return vector


def validate_profile_weights(
    profile_weights: Sequence[float],
    *,
    n_profiles: int,
) -> np.ndarray:
    """Validate aggregated profile frequency weights before CLARA."""
    weights = np.asarray(profile_weights, dtype=float)
    if weights.ndim != 1:
        raise ValueError("aggWeights must be one-dimensional.")
    if len(weights) != n_profiles:
        raise ValueError(
            f"aggWeights length ({len(weights)}) does not match the number of "
            f"profiles ({n_profiles})."
        )
    if not np.all(np.isfinite(weights)):
        raise ValueError("aggWeights contain NaN or infinite values.")
    if np.any(weights <= 0):
        raise ValueError("aggWeights must be strictly positive.")
    return weights


def validate_kvals(kvals: Optional[Sequence[int]]) -> List[int]:
    """Validate cluster-size values for MD-CLARA (shared by public API and engine)."""
    if kvals is None:
        return list(range(2, 11))
    if not kvals:
        raise ValueError("kvals must contain at least one value.")
    if any(not isinstance(k, (int, np.integer)) for k in kvals):
        raise TypeError("Every value in kvals must be an integer.")
    normalized = [int(k) for k in kvals]
    if any(k < 2 for k in normalized):
        raise ValueError("Every value in kvals must be at least 2.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("kvals must not contain duplicate values.")
    return normalized


def subset_sequence_data(
    seqdata: SequenceData,
    indices: Sequence[int],
    *,
    weights: Optional[Sequence[float]] = None,
) -> SequenceData:
    """Return a SequenceData view containing only the requested row indices.

    ``weights``, when given, replaces the subset of ``seqdata.weights``. MD-CLARA
    uses this to attach profile frequencies (``aggWeights``) after collapsing
    identical trajectories.
    """
    idx = np.asarray(indices, dtype=int)
    if idx.ndim != 1:
        raise ValueError("indices must be one-dimensional.")
    n_rows = len(seqdata.data)
    if np.any(idx < 0) or np.any(idx >= n_rows):
        raise IndexError(
            f"indices contain values outside the valid row range [0, {n_rows - 1}]."
        )
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
    if weights is not None:
        weight_arr = np.asarray(weights, dtype=float)
        if weight_arr.shape != (idx.size,):
            raise ValueError(
                f"weights length ({weight_arr.size}) must match the number of "
                f"subset rows ({idx.size})."
            )
        kwargs["weights"] = weight_arr
    elif getattr(seqdata, "weights", None) is not None:
        kwargs["weights"] = np.asarray(seqdata.weights)[idx]
    if getattr(seqdata, "void", None) is not None:
        kwargs["void"] = seqdata.void

    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull):
            return SequenceData(**kwargs)


def _substitution_cost_kwargs(
    method: str,
    sm_method: str,
    dist_args: Dict[str, Any],
) -> Dict[str, Any]:
    """Mirror ``get_distance_matrix`` kwargs for ``get_substitution_cost_matrix``."""
    method = method.upper()
    sm_method = sm_method.upper()
    tv = False
    cost: Optional[float] = None

    if sm_method in {"INDELS", "INDELSLOG"} and method == "DHD":
        tv = True
    elif sm_method == "TRATE":
        if method in {"OM", "HAM"}:
            cost = 2
        elif method == "DHD":
            cost = 4
            tv = True
    elif sm_method == "CONSTANT":
        cost = 1 if method == "HAM" else 2
    elif sm_method in {"FUTURE", "FEATURES"}:
        cost = 2

    kwargs: Dict[str, Any] = {
        "method": sm_method,
        "cval": dist_args.get("cval", cost),
        "miss_cost": dist_args.get("miss_cost", cost),
        "time_varying": dist_args.get("time_varying", tv),
        "weighted": dist_args.get("weighted", True),
        "transition": dist_args.get("transition", "both"),
        "lag": dist_args.get("lag", 1),
    }
    if sm_method == "FEATURES":
        for key in ("state_features", "feature_weights", "feature_type"):
            if key in dist_args:
                kwargs[key] = dist_args[key]
    return kwargs


_DATA_INDEPENDENT_SM = frozenset({"CONSTANT"})


def _sm_spec_is_data_dependent(sm: Any) -> bool:
    """True when ``sm`` still needs empirical frequencies from the data."""
    if isinstance(sm, str):
        return sm.upper() not in _DATA_INDEPENDENT_SM
    if isinstance(sm, (list, tuple)):
        return any(_sm_spec_is_data_dependent(item) for item in sm)
    return False


def _indel_auto_is_frequency_dependent(sm: Any) -> bool:
    """True when ``indel='auto'`` still needs observed frequencies to resolve."""
    return _sm_spec_is_data_dependent(sm)


def distance_params_are_data_dependent(
    strategy: str,
    distance_params: Optional[Dict[str, Any]],
) -> bool:
    """
    Whether substitution/indel costs must be estimated from observed frequencies.

    CONSTANT substitution (and an explicit substitution matrix) is
    data-independent, including ``indel='auto'``, which is then determined by
    the substitution costs themselves (typically max(sm)/2). TRATE, INDELS,
    and INDELSLOG remain data-dependent and must be frozen on the original
    weighted cases *before* unique-profile aggregation.
    """
    strategy = strategy.lower()
    params = dict(distance_params or {})
    if strategy == "dat":
        method_params = list(params.get("method_params", []))
        return any(
            _sm_spec_is_data_dependent(block.get("sm"))
            or (
                str(block.get("indel", "")).lower() == "auto"
                and _indel_auto_is_frequency_dependent(block.get("sm"))
            )
            for block in method_params
        )
    if str(params.get("indel", "")).lower() == "auto":
        return _indel_auto_is_frequency_dependent(params.get("sm"))
    return _sm_spec_is_data_dependent(params.get("sm"))


_OM_INDEL_METHODS = frozenset(
    {
        "OM",
        "OMLOC",
        "OMSLEN",
        "OMSPELL",
        "OMSPELLRS",
        "OMSTRAN",
        "OMTSPELL",
    }
)


def guard_on_demand_distance_params(dist_args: Dict[str, Any]) -> None:
    """Reject distance settings that depend on the queried data subset."""
    method = str(dist_args.get("method", "OM")).upper()
    if method in {"CHI2", "EUCLID"}:
        raise NotImplementedError(
            f"{method} is not yet supported in on-demand MD-CLARA because its "
            "distance definition may depend on the queried data subset."
        )
    if str(dist_args.get("norm", "none")).lower() == "elzingastuder":
        raise NotImplementedError(
            "norm='ElzingaStuder' is not yet supported in on-demand MD-CLARA "
            "unless a fixed global normalization reference is supplied."
        )


def _normalize_generated_indel_for_explicit_use(
    seqdata: SequenceData,
    method: str,
    generated_indel: Any,
) -> Any:
    """
    Convert seqcost-generated indels into the explicit-vector format expected
    by ``get_distance_matrix()``.
    """
    if method.upper() not in _OM_INDEL_METHODS:
        return generated_indel

    if np.isscalar(generated_indel):
        return generated_indel

    arr = np.asarray(generated_indel, dtype=float)
    if arr.ndim != 1:
        raise ValueError(
            "Frozen state-dependent indel costs must be one-dimensional."
        )

    n_states = len(seqdata.states)
    if arr.size == n_states + 1:
        arr = arr[1:]

    if arr.size != n_states:
        raise ValueError(
            f"Expected {n_states} explicit indel costs, got {arr.size}."
        )

    return arr


def _explicit_indel_from_constant(
    seqdata: SequenceData,
    method: str,
    dist_args: Dict[str, Any],
) -> Any:
    """Resolve ``indel='auto'`` for CONSTANT substitution (not frequency-based)."""
    from sequenzo.dissimilarity_measures import get_substitution_cost_matrix

    sm_kw = _substitution_cost_kwargs(method, "CONSTANT", dist_args)
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull):
            costs = get_substitution_cost_matrix(seqdata, **sm_kw)
    return _normalize_generated_indel_for_explicit_use(
        seqdata, method, costs["indel"]
    )


def freeze_seqdist_costs(
    seqdata: SequenceData,
    dist_args: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Replace data-dependent ``sm`` method strings with matrices estimated on
    ``seqdata``, so subsample and full-data distance queries share one metric.
    """
    frozen = dict(dist_args)
    guard_on_demand_distance_params(frozen)
    method = str(frozen.get("method", "OM")).upper()
    sm = frozen.get("sm")
    indel = frozen.get("indel")

    sm_method: Optional[str] = None
    if isinstance(sm, str):
        sm_method = sm.upper()
        if sm_method == "CONSTANT":
            if str(indel).lower() == "auto":
                frozen["indel"] = _explicit_indel_from_constant(seqdata, method, frozen)
            return frozen
    elif sm is None:
        if method == "HAM":
            sm_method = "CONSTANT"
        elif method == "DHD":
            sm_method = "TRATE"
        else:
            if str(indel).lower() == "auto":
                raise ValueError(
                    "indel='auto' requires a substitution-cost specification."
                )
            return frozen
    else:
        if str(indel).lower() == "auto":
            frozen["indel"] = float(np.nanmax(np.asarray(sm, dtype=float))) / 2.0
        return frozen

    if sm_method is None:
        return frozen

    from sequenzo.dissimilarity_measures import get_substitution_cost_matrix

    sm_kw = _substitution_cost_kwargs(method, sm_method, frozen)
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull):
            costs = get_substitution_cost_matrix(seqdata, **sm_kw)

    sm_out = costs["sm"]
    if isinstance(sm_out, pd.DataFrame):
        sm_out = sm_out.to_numpy(dtype=float)
    else:
        sm_out = np.asarray(sm_out, dtype=float)
    frozen["sm"] = sm_out
    if str(indel).lower() == "auto":
        frozen["indel"] = _normalize_generated_indel_for_explicit_use(
            seqdata,
            method,
            costs["indel"],
        )
    return frozen


def compute_distance_matrix(
    seqdata: SequenceData,
    dist_args: Dict[str, Any],
    *,
    refseq: Optional[Union[int, List[List[int]]]] = None,
    full_matrix: bool = True,
) -> np.ndarray:
    """Call get_distance_matrix and return a NumPy array."""
    opts = dict(dist_args)
    opts["seqdata"] = seqdata
    opts["full_matrix"] = full_matrix
    opts["as_numpy"] = True
    opts["quiet"] = True
    if refseq is not None:
        opts["refseq"] = refseq

    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull):
            result = get_distance_matrix(opts=opts)
    return np.asarray(result, dtype=float)


def warm_distance_query_caches(seqdata: SequenceData) -> None:
    """Precompute integer values, lengths, and unique sequences for CLARA queries."""
    warm_unique_sequence_cache(seqdata)


def _n_sequence_rows(seqdata: SequenceData) -> int:
    values = getattr(seqdata, "seqdata", None)
    if values is not None:
        return int(np.asarray(values).shape[0])
    return int(len(seqdata.data))


def pairwise_distances_on_indices(
    seqdata: SequenceData,
    dist_args: Dict[str, Any],
    sample_indices: Sequence[int],
    *,
    condensed: bool = False,
) -> np.ndarray:
    """
    Within-sample distances without copying ``seqdata`` into a new SequenceData.

    The full-object path is used when the sample is exactly ``0..N-1``, so a
    complete pairwise matrix does not go through ``refseq=[all, all]`` (which
    would duplicate unique sequences). Subsamples use ``refseq=[idx, idx]``
    on the parent object, which unique-ifies only the requested rows.
    Substitution costs must already be frozen on ``seqdata`` (see
    ``freeze_seqdist_costs``); otherwise subsample and full-data metrics can
    diverge.
    """
    idx = np.asarray(sample_indices, dtype=int).reshape(-1)
    n_full = _n_sequence_rows(seqdata)
    covers_all = idx.size == n_full and np.array_equal(idx, np.arange(n_full))

    if covers_all:
        matrix = compute_distance_matrix(
            seqdata,
            dist_args,
            full_matrix=not condensed,
        )
        if condensed:
            return assert_condensed_distance_shape(matrix, sample_indices)
        return assert_sample_distance_shape(matrix, sample_indices)

    matrix = compute_distance_matrix(
        seqdata,
        dist_args,
        refseq=[idx, idx],
        full_matrix=not condensed,
    )
    if condensed:
        if matrix.ndim == 1:
            return assert_condensed_distance_shape(matrix, sample_indices)
        from scipy.spatial.distance import squareform

        matrix = assert_sample_distance_shape(matrix, sample_indices)
        return assert_condensed_distance_shape(
            squareform(matrix, checks=False),
            sample_indices,
        )
    return assert_sample_distance_shape(matrix, sample_indices)


def warn_nested_parallelism(
    *,
    n_jobs: int,
    n_jobs_domains: int,
) -> None:
    """Warn when CLARA repetitions and DAT domain work are both parallelized."""
    reps_parallel = n_jobs != 1
    domains_parallel = n_jobs_domains != 1
    if reps_parallel and domains_parallel:
        import warnings

        warnings.warn(
            "Both CLARA repetitions (n_jobs) and DAT domain-level distance calls "
            "(n_jobs_domains) are parallelized. This may oversubscribe CPUs and "
            "increase peak memory. Prefer n_jobs=-1 with n_jobs_domains=1, or "
            "n_jobs=1 with n_jobs_domains=-1.",
            UserWarning,
            stacklevel=4,
        )


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
    """
    Subset each domain to unique multidomain profiles.

    Profile frequencies (``aggregation['aggWeights']``) are written onto every
    aggregated ``SequenceData.weights``. That keeps data-dependent substitution
    and indel costs (TRATE, INDELS, INDELSLOG) frequency-weighted if they are
    estimated after aggregation. Prefer estimating those costs on the original
    weighted data first (see ``make_aggregated_distance_provider``).
    """
    agg_idx = one_based_to_zero_based(aggregation["aggIndex"], name="aggIndex")
    agg_weights = np.asarray(aggregation["aggWeights"], dtype=float)
    if agg_weights.shape != (agg_idx.size,):
        raise ValueError(
            "aggWeights length does not match the number of unique profiles."
        )
    return [
        subset_sequence_data(domain, agg_idx, weights=agg_weights)
        for domain in domains
    ]


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
    "freeze_seqdist_costs",
    "guard_on_demand_distance_params",
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
    "validate_profile_weights",
    "validate_kvals",
    "distance_params_are_data_dependent",
    "assert_sample_distance_shape",
    "assert_condensed_distance_shape",
    "assert_distance_to_medoids_shape",
    "warn_nested_parallelism",
]
