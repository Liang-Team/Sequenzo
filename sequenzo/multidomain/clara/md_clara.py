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
from .diagnostics import (
    dat_domain_contributions,
    summarize_combined_state_space,
    summarize_subsample_coverage,
)
from .distance_providers import DATDistanceProvider, make_distance_provider
from .results import MDClaraResult
from ._utils import (
    aggregate_domains,
    build_multidomain_profile_frame,
    validate_domain_weights,
    validate_kvals,
    validate_multidomain_domains,
    warn_nested_parallelism,
)

_VALID_CRITERIA = frozenset({"distance"})

_OM12 = {"method": "OM", "sm": "CONSTANT", "indel": 1, "norm": "none"}


def _default_distance_params(strategy: str, n_domains: int) -> Dict[str, Any]:
    """Classical OM(1,2) defaults aligned with the paper's main experiments."""
    strategy = strategy.lower()
    if strategy == "idcd":
        return dict(_OM12)
    if strategy == "cat":
        return {
            "method": "OM",
            "sm": ["CONSTANT"] * n_domains,
            "indel": 1,
            "norm": "none",
        }
    if strategy == "dat":
        return {
            "method_params": [dict(_OM12) for _ in range(n_domains)],
            "link": "sum",
        }
    raise ValueError("strategy must be one of: 'idcd', 'cat', 'dat'")


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
    subsample_diagnostics: bool = False,
    rare_profile_threshold: int = 5,
    combined_state_space: bool = False,
    dat_domain_contribution: bool = False,
    use_medoid_cache: bool = False,
    condensed_subsample: bool = True,
) -> MDClaraResult:
    """
    Scalable multidomain CLARA with IDCD, CAT, or DAT dissimilarity.

    When ``distance_params`` is omitted, distances use classical OM(1,2):
    ``method='OM'``, ``sm='CONSTANT'``, ``indel=1``, ``norm='none'`` (CAT/DAT
    replicate this per domain). Cluster labels in ``result.clustering`` are
    one-based integers (``1``, ``2``, ...); medoid indices remain zero-based.

    First stable release: ``method='crisp'`` and a single entry in ``criteria``.

    For ``strategy='dat'``, optional ``distance_params['n_jobs_domains']``
    parallelizes per-domain distance calls inside the provider. Use ``1`` when
    ``n_jobs`` parallelizes CLARA iterations (default); use ``-1`` when
    ``n_jobs=1`` and DAT domain work should use multiple cores.

    For large datasets, consider a conservative ``n_jobs`` value because parallel
    CLARA iterations may increase peak memory use.

    Set ``combined_state_space=True`` to attach an IDCD/CAT state-space summary.

    Optimization ablation (within-subsample distances only; all-to-medoid
    matrices are always ``N* x K``):

    - ``condensed_subsample=False``: square ``b x b`` subsample matrices (DAT
      holds one square matrix per domain before combining).
    - ``condensed_subsample=True``: condensed subsample vectors (DAT combines
      condensed domain vectors).
    - ``use_medoid_cache=True``: reuse medoid columns across ``kvals`` within
      each repetition (may increase peak memory).

    Typical ablation grid::

        condensed_subsample=False, use_medoid_cache=False
        condensed_subsample=True,  use_medoid_cache=False
        condensed_subsample=True,  use_medoid_cache=True
    """
    if n_jobs == 0:
        raise ValueError("n_jobs must not be 0.")
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

    kvals = validate_kvals(kvals)

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

    params = (
        dict(distance_params)
        if distance_params is not None
        else _default_distance_params(strategy, len(domains))
    )
    n_jobs_domains = int(params.get("n_jobs_domains", 1))
    if strategy == "dat":
        if n_jobs_domains == 0:
            raise ValueError("distance_params['n_jobs_domains'] must not be 0.")
        warn_nested_parallelism(n_jobs=n_jobs, n_jobs_domains=n_jobs_domains)

    provider = make_distance_provider(
        agg_domains,
        strategy=strategy,
        distance_params=params,
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
        subsample_diagnostics=subsample_diagnostics,
        rare_profile_threshold=rare_profile_threshold,
        use_medoid_cache=use_medoid_cache,
        condensed_subsample=condensed_subsample,
    )

    raw["requested_sample_size"] = requested_sample_size
    raw["effective_sample_size"] = effective_sample_size
    raw["n_unique_profiles"] = n_unique_profiles

    state_space_summary = None
    if combined_state_space and strategy in {"idcd", "cat"}:
        ch_sep = params.get("ch_sep", "+")
        state_space_summary = summarize_combined_state_space(
            domains,
            ch_sep=ch_sep,
        )

    route_diagnostics: Dict[str, Any] = {}
    if raw.get("medoid_cache_stats"):
        route_diagnostics["medoid_cache"] = raw["medoid_cache_stats"]

    subsample_table = raw.get("subsample_diagnostics")
    if subsample_table is not None and not subsample_table.empty:
        route_diagnostics["subsample_coverage_summary"] = summarize_subsample_coverage(
            subsample_table
        )

    domain_contributions = None
    if dat_domain_contribution and strategy == "dat" and isinstance(provider, DATDistanceProvider):
        profile_weights = np.asarray(ac["aggWeights"], dtype=float)
        domain_contributions = {}
        for k in kvals:
            cluster_info = raw["clara"][k]
            domain_contributions[k] = dat_domain_contributions(
                provider,
                medoids=cluster_info["medoids_agg"],
                clustering=cluster_info["profile_clustering"],
                profile_weights=profile_weights,
            )
        route_diagnostics["dat_domain_contributions"] = domain_contributions

    return _to_md_clara_result(
        raw,
        strategy=strategy,
        method=method,
        kvals=kvals,
        distance_params=params,
        R=R,
        stability_requested=stability,
        combined_state_space=state_space_summary,
        subsample_diagnostics=subsample_table,
        route_diagnostics=route_diagnostics or None,
        condensed_subsample=condensed_subsample,
        use_medoid_cache=use_medoid_cache,
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
    combined_state_space: Optional[Dict[str, Any]] = None,
    subsample_diagnostics: Optional[Any] = None,
    route_diagnostics: Optional[Dict[str, Any]] = None,
    condensed_subsample: bool = True,
    use_medoid_cache: bool = False,
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
        "subsample_diagnostics": subsample_diagnostics is not None,
        "combined_state_space": combined_state_space is not None,
        "condensed_subsample": raw.get("condensed_subsample", condensed_subsample),
        "use_medoid_cache": raw.get("use_medoid_cache", use_medoid_cache),
    }
    if combined_state_space is not None:
        settings["combined_state_space_summary"] = combined_state_space

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
        combined_state_space=combined_state_space,
        subsample_diagnostics=subsample_diagnostics,
        route_diagnostics=route_diagnostics,
    )


__all__ = ["md_clara"]
