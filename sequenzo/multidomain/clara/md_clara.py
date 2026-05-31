"""
@Author  : Yuqi Liang 梁彧祺
@File    : md_clara.py
@Time    : 18/05/2026 18:12
@Desc    : 
Public API for scalable multidomain CLARA (IDCD, CAT, DAT).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from sequenzo.big_data.clara.utils.aggregatecases import DataFrameAggregator
from sequenzo.define_sequence_data import SequenceData

from .clara_engine import clara_from_distance_provider
from .distance_providers import make_distance_provider
from .results import MDClaraResult
from ._utils import (
    aggregate_domains,
    build_multidomain_profile_frame,
    validate_domain_weights,
    validate_multidomain_domains,
)

_VALID_CRITERIA = frozenset({"distance"})


def _validate_kvals(kvals: Sequence[int]) -> List[int]:
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


def _normalize_criteria(criteria: Sequence[str]) -> Tuple[str, ...]:
    normalized = tuple(c.lower() for c in criteria)
    if len(normalized) != 1:
        raise ValueError(
            "md_clara currently supports exactly one clustering criterion per run. "
            f"Got {normalized!r}. Pass e.g. criteria=('distance',). "
            "Multi-criterion results (result.by_criterion) are planned for a later release."
        )
    if normalized[0] not in _VALID_CRITERIA:
        raise ValueError(
            f"Unknown criterion {normalized[0]!r}. "
            f"Choose one of: {sorted(_VALID_CRITERIA)}."
        )
    return normalized


def _resolve_effective_sample_size(
    sample_size: Optional[int],
    *,
    kvals: Sequence[int],
    n_unique_profiles: int,
) -> tuple[Optional[int], int]:
    """Map requested ``b`` to effective subsample size capped by ``N*``."""
    requested_sample_size = sample_size
    max_k = max(kvals)

    if sample_size is None:
        effective = min(40 + 2 * max_k, n_unique_profiles)
    elif sample_size > n_unique_profiles:
        warnings.warn(
            f"sample_size={sample_size} exceeds the number of unique "
            f"multidomain profiles ({n_unique_profiles}). "
            f"Using sample_size={n_unique_profiles}.",
            UserWarning,
            stacklevel=3,
        )
        effective = n_unique_profiles
    else:
        effective = sample_size

    return requested_sample_size, effective


def md_clara(
    domains: List[SequenceData],
    strategy: str = "idcd",
    R: int = 100,
    sample_size: Optional[int] = None,
    kvals: Optional[Sequence[int]] = None,
    method: str = "crisp",
    distance_params: Optional[Dict[str, Any]] = None,
    criteria: Sequence[str] = ("distance",),
    stability: bool = False,
    random_state: Optional[int] = None,
    n_jobs: int = -1,
    verbose: bool = True,
) -> MDClaraResult:
    """
    Scalable multidomain CLARA with IDCD, CAT, or DAT dissimilarity.

    First stable release: ``method='crisp'`` and a single entry in ``criteria``.

    For ``strategy='dat'``, optional ``distance_params['n_jobs_domains']``
    parallelizes per-domain distance calls inside the provider. Use ``1`` when
    ``n_jobs`` parallelizes CLARA iterations (default); use ``-1`` when
    ``n_jobs=1`` and DAT domain work should use multiple cores.

    For large datasets, consider a conservative ``n_jobs`` value because parallel
    CLARA iterations may increase peak memory use.
    """
    if R < 1:
        raise ValueError("R must be at least 1.")
    if stability and R < 2:
        raise ValueError("stability=True requires R >= 2.")

    validate_multidomain_domains(domains)
    strategy = strategy.lower()
    method = method.lower()
    if method != "crisp":
        raise ValueError(
            "md_clara currently supports method='crisp' only. "
            "Fuzzy and representativeness clustering will be added later."
        )

    criteria_tuple = _normalize_criteria(criteria)

    if kvals is None:
        kvals = list(range(2, 11))
    else:
        kvals = _validate_kvals(kvals)

    reference = domains[0]
    reference_weights = validate_domain_weights(domains)
    multidomain_profiles = build_multidomain_profile_frame(domains)
    ac = DataFrameAggregator().aggregate(
        multidomain_profiles,
        weights=reference_weights,
    )
    agg_domains = aggregate_domains(domains, ac)

    n_unique_profiles = len(ac["aggWeights"])
    requested_sample_size, effective_sample_size = _resolve_effective_sample_size(
        sample_size,
        kvals=kvals,
        n_unique_profiles=n_unique_profiles,
    )

    provider = make_distance_provider(
        agg_domains,
        strategy=strategy,
        distance_params=distance_params,
    )
    if len(ac["aggWeights"]) != provider.n_sequences():
        raise ValueError(
            "Aggregation size does not match the number of profiles represented "
            "by the distance provider."
        )

    raw = clara_from_distance_provider(
        provider,
        reference_seqdata=reference,
        aggregation=ac,
        R=R,
        sample_size=effective_sample_size,
        kvals=kvals,
        method=method,
        criteria=criteria_tuple,
        stability=stability,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    raw["requested_sample_size"] = requested_sample_size
    raw["effective_sample_size"] = effective_sample_size
    raw["n_unique_profiles"] = n_unique_profiles

    return _to_md_clara_result(
        raw,
        strategy=strategy,
        method=method,
        kvals=kvals,
        distance_params=distance_params,
        R=R,
        stability_requested=stability,
    )


def _to_md_clara_result(
    raw: Dict[str, Any],
    *,
    strategy: str,
    method: str,
    kvals: Sequence[int],
    distance_params: Optional[Dict[str, Any]],
    R: int,
    stability_requested: bool,
) -> MDClaraResult:
    """Convert engine output into :class:`MDClaraResult`."""
    if "clustering" not in raw:
        raise ValueError(
            "Unexpected CLARA engine output (missing 'clustering'). "
            "Ensure only one criterion was requested."
        )

    effective_sample_size = raw.get("effective_sample_size", raw.get("sample_size"))
    criterion = raw.get("criterion", "distance")

    stats = raw["stats"].copy()
    stats["R"] = R
    stats["sample_size"] = effective_sample_size
    stats["strategy"] = strategy

    best_by_k: Dict[int, Dict[str, Any]] = {}
    medoids: Dict[int, np.ndarray] = {}
    stability_out: Dict[int, Dict[str, Any]] = {}

    for k in kvals:
        cluster_info = raw["clara"][k]
        best_by_k[k] = cluster_info
        medoids[k] = np.asarray(cluster_info["medoids"])
        if stability_requested and cluster_info.get("stability") is not None:
            stability_out[k] = cluster_info["stability"]

    settings = {
        "strategy": strategy,
        "method": method,
        "kvals": list(kvals),
        "distance_params": distance_params,
        "R": R,
        "sample_size": effective_sample_size,
        "requested_sample_size": raw.get("requested_sample_size"),
        "effective_sample_size": effective_sample_size,
        "n_unique_profiles": raw.get("n_unique_profiles"),
        "criteria": [criterion],
        "stability": stability_requested,
    }

    return MDClaraResult(
        strategy=strategy,
        method=method,
        kvals=list(kvals),
        best_by_k=best_by_k,
        clustering=raw["clustering"],
        stats=stats,
        medoids=medoids,
        settings=settings,
        stability=stability_out if stability_out else None,
        membership=None,
    )


__all__ = ["md_clara"]
