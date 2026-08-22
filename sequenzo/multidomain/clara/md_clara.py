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

from .clara_engine import (
    STUDER_BOOTSTRAP_POLICIES,
    clara_from_distance_provider,
    normalize_sampling_policy,
)
from .diagnostics import (
    dat_domain_contributions,
    summarize_combined_state_space,
    summarize_subsample_coverage,
)
from .distance_providers import DATDistanceProvider, make_aggregated_distance_provider
from .results import MDClaraResult
from ._utils import (
    build_multidomain_profile_frame,
    distance_params_are_data_dependent,
    one_based_to_zero_based,
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
    n_original: int,
    sampling_policy: str,
) -> tuple[Optional[int], int]:
    """Map requested ``b`` to effective subsample size."""
    requested_sample_size = sample_size
    max_k = max(kvals)
    policy = normalize_sampling_policy(sampling_policy)

    # With-replacement bootstrap may draw more than N* unique profiles.
    if policy in STUDER_BOOTSTRAP_POLICIES:
        if sample_size is None:
            return requested_sample_size, 40 + 2 * max_k
        return requested_sample_size, int(sample_size)

    cap = n_original if policy == "case_sample_then_aggregate" else n_unique_profiles

    if sample_size is None:
        effective = min(40 + 2 * max_k, cap)
    elif sample_size > cap:
        warnings.warn(
            f"sample_size={sample_size} exceeds the "
            f"{'number of cases' if policy == 'case_sample_then_aggregate' else 'number of unique multidomain profiles'} "
            f"({cap}). Using sample_size={cap}.",
            UserWarning,
            stacklevel=3,
        )
        effective = cap
    else:
        effective = sample_size

    return requested_sample_size, effective


def _count_mixed_latent_profiles(
    case_labels: np.ndarray,
    disagg: np.ndarray,
    n_profiles: int,
) -> int:
    """How many unique profiles mix more than one case-level latent label."""
    mixed = 0
    for u in range(int(n_profiles)):
        labs = case_labels[disagg == u]
        if labs.size and np.unique(labs).size > 1:
            mixed += 1
    return mixed


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
    use_medoid_cache: bool = True,
    condensed_subsample: bool = True,
    sampling_policy: str = "studer_bootstrap",
    subsample_schedule: Optional[Sequence[Any]] = None,
    case_true_labels: Optional[np.ndarray] = None,
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

    Two paper implementations (within-subsample distances; all-to-medoid
    matrices remain ``N* x K``):

    - Unoptimized: ``condensed_subsample=False, use_medoid_cache=False``.
      Square ``b x b`` subsample matrices; DAT holds one square matrix per
      domain before combining; medoid columns are recomputed for each ``K``.
    - Optimized (default): ``condensed_subsample=True, use_medoid_cache=True``.
      Condensed subsample vectors (DAT combines condensed domain vectors,
      then the engine expands once for PAM); medoid columns are reused across
      ``kvals`` within each repetition.

    An intermediate condensed-only path (``use_medoid_cache=False``) is kept
    for engineering ablation.

    Sampling policies (the full-data objective always uses frequencies
    a_u; only the candidate-medoid search draw changes):

    - ``studer_bootstrap`` (default): frequency-proportional with replacement;
      subsample PAM uses this-round draw counts (``seqclararange``).
    - ``uniform_unique_profiles``: profile-balanced sampling of distinct
      profiles without replacement; subsample PAM uses full frequencies a_u.
    - ``frequency_proportional_profiles``: PPS without replacement; full a_u.
    - ``case_sample_then_aggregate``: sample original cases without
      replacement, then collapse; subsample PAM still uses full a_u.
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

    sampling_policy = normalize_sampling_policy(sampling_policy)

    reference = domains[0]
    reference_weights = validate_domain_weights(domains)
    multidomain_profiles = build_multidomain_profile_frame(domains)
    ac = DataFrameAggregator().aggregate(
        multidomain_profiles,
        weights=reference_weights,
    )

    n_unique_profiles = len(ac["aggWeights"])
    n_original = int(reference.seqdata.shape[0])
    requested_sample_size, effective_sample_size = _resolve_effective_sample_size(
        sample_size,
        kvals=kvals,
        n_unique_profiles=n_unique_profiles,
        n_original=n_original,
        sampling_policy=sampling_policy,
    )

    params = (
        dict(distance_params)
        if distance_params is not None
        else _default_distance_params(strategy, len(domains))
    )
    costs_frozen_on_original = distance_params_are_data_dependent(strategy, params)
    n_jobs_domains = int(params.get("n_jobs_domains", 1))
    if strategy == "dat":
        if n_jobs_domains == 0:
            raise ValueError("distance_params['n_jobs_domains'] must not be 0.")
        warn_nested_parallelism(n_jobs=n_jobs, n_jobs_domains=n_jobs_domains)

    provider = make_aggregated_distance_provider(
        domains,
        strategy=strategy,
        distance_params=params,
        aggregation=ac,
    )
    if len(ac["aggWeights"]) != provider.n_sequences():
        raise ValueError(
            "Aggregation size does not match the number of profiles represented "
            "by the distance provider."
        )

    profile_true_labels = None
    n_mixed_latent_profiles = 0
    if case_true_labels is not None:
        case_arr = np.asarray(case_true_labels)
        if case_arr.shape != (n_original,):
            raise ValueError("case_true_labels must have length N (original cases).")
        agg_idx = one_based_to_zero_based(ac["aggIndex"], name="aggIndex")
        disagg = one_based_to_zero_based(ac["disaggIndex"], name="disaggIndex")
        profile_true_labels = case_arr[agg_idx]
        n_mixed_latent_profiles = _count_mixed_latent_profiles(
            case_arr, disagg, n_unique_profiles
        )
        if n_mixed_latent_profiles:
            warnings.warn(
                f"{n_mixed_latent_profiles} unique profile(s) mix latent labels "
                "after aggregation. Profile-level rare-cluster coverage uses the "
                "representative case; case-level ARI_true is unaffected.",
                UserWarning,
                stacklevel=2,
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
        sampling_policy=sampling_policy,
        subsample_schedule=subsample_schedule,
        profile_true_labels=profile_true_labels,
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
        sampling_policy=sampling_policy,
        costs_frozen_on_original=costs_frozen_on_original,
        random_state=random_state,
        n_mixed_latent_profiles=n_mixed_latent_profiles,
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
    use_medoid_cache: bool = True,
    sampling_policy: str = "studer_bootstrap",
    costs_frozen_on_original: bool = False,
    random_state: Optional[int] = None,
    n_mixed_latent_profiles: int = 0,
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
        "sampling_policy": raw.get("sampling_policy", sampling_policy),
        "costs_frozen_on_original": costs_frozen_on_original,
        "random_state": random_state,
        "n_mixed_latent_profiles": n_mixed_latent_profiles,
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
