"""
@Author  : Yuqi Liang 梁彧祺
@File    : diagnostics.py
@Time    : 31/05/2026 10:11
@Desc    : 
Multidomain-specific diagnostics for MD-CLARA (Phase 4).
"""

from __future__ import annotations

import warnings
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from sequenzo.big_data.clara.clara import adjustedRandIndex, jaccardCoef
from sequenzo.define_sequence_data import SequenceData
from sequenzo.multidomain.idcd import create_idcd_sequence_from_domains, validate_multidomain_domains

from .distance_providers import DATDistanceProvider
from .results import MDClaraResult


def _labels_contingency(
    left: np.ndarray,
    right: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    df = pd.DataFrame({"left": left, "right": right})
    if weights is not None:
        df["w"] = weights
        return df.groupby(["left", "right"])["w"].sum().unstack(fill_value=0)
    return pd.crosstab(df["left"], df["right"])


def _validate_comparison_weights(
    weights: np.ndarray,
    *,
    reference_len: int,
) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1:
        raise ValueError("weights must be one-dimensional.")
    if len(weights) != reference_len:
        raise ValueError("weights length must match clustering rows.")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must contain only finite values.")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative.")
    return weights


def _pairwise_partition_metrics(
    left: np.ndarray,
    right: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        raise ValueError("Label vectors must have the same length.")

    if weights is None:
        ari = float(adjusted_rand_score(left, right))
        tab = pd.crosstab(left, right)
        jaccard = float(jaccardCoef(tab))
    else:
        tab = _labels_contingency(left, right, weights=weights)
        ari = float(adjustedRandIndex(tab))
        jaccard = float(jaccardCoef(tab))

    return {"ari": ari, "jaccard": jaccard}


def compare_md_clara_strategies(
    results: Dict[str, MDClaraResult],
    *,
    k: int,
    weights: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Compare cluster partitions across IDCD, CAT, and DAT MD-CLARA runs.

    Parameters
    ----------
    results
        Mapping from strategy name (e.g. ``'idcd'``) to :class:`MDClaraResult`.
    k
        Cluster count to compare.
    weights
        Optional case weights aligned with clustering rows.
    """
    if len(results) < 2:
        raise ValueError("Provide at least two strategy results to compare.")

    strategies = sorted(results.keys())
    rows: List[Dict[str, Any]] = []

    reference_index: Optional[pd.Index] = None
    reference_len: Optional[int] = None
    labels_by_strategy: Dict[str, np.ndarray] = {}
    for name, result in results.items():
        clustering = result.clustering
        column = f"Cluster {k}"
        if column not in clustering.columns:
            raise KeyError(f"Strategy {name!r} has no clustering stored for k={k}.")
        if reference_index is None:
            reference_index = clustering.index
            reference_len = len(reference_index)
        elif not clustering.index.equals(reference_index):
            raise ValueError(
                f"Strategy {name!r} has a different case order. "
                "Align clustering rows before comparing strategies."
            )
        labels = np.asarray(clustering.loc[reference_index, column], dtype=float)
        labels_by_strategy[name] = labels

    if weights is not None:
        weights = _validate_comparison_weights(weights, reference_len=reference_len)

    for left_name, right_name in combinations(strategies, 2):
        metrics = _pairwise_partition_metrics(
            labels_by_strategy[left_name],
            labels_by_strategy[right_name],
            weights=weights,
        )
        rows.append(
            {
                "strategy_left": left_name,
                "strategy_right": right_name,
                "k": k,
                "ari": metrics["ari"],
                "jaccard": metrics["jaccard"],
            }
        )

    return pd.DataFrame(rows)


def summarize_combined_state_space(
    domains: List[SequenceData],
    *,
    rare_threshold: int = 5,
    ch_sep: str = "+",
    warn_threshold_states: int = 500,
    warn_rare_share: float = 0.5,
    warn_coverage: float = 0.25,
) -> Dict[str, Any]:
    """
    Summarize the observed versus theoretical multidomain combined state space.

    Frequencies are reported in two ways:

    * **Case frequency**: number of cases that experience a combined state at
      least once (used for ``rare_state_share`` and singleton warnings).
    * **Position frequency**: number of (case, time) positions where the state
      appears.

    Useful before IDCD or CAT clustering to assess sparsity.
    """
    validate_multidomain_domains(domains)
    md = create_idcd_sequence_from_domains(domains, ch_sep=ch_sep, quiet=True)

    theoretical = int(np.prod([len(domain.states) for domain in domains]))
    observed_states = list(md.states)
    n_observed = len(observed_states)

    time_cols = list(md.time)
    seq_block = md.data.loc[:, time_cols]
    state_to_index = {state: idx for idx, state in enumerate(observed_states)}
    case_counts = np.zeros(n_observed, dtype=float)
    position_counts = np.zeros(n_observed, dtype=float)

    for row in seq_block.to_numpy():
        seen_in_case = set()
        for value in row:
            idx = state_to_index.get(value)
            if idx is None:
                continue
            position_counts[idx] += 1.0
            if idx not in seen_in_case:
                case_counts[idx] += 1.0
                seen_in_case.add(idx)

    coverage = n_observed / theoretical if theoretical > 0 else np.nan
    rare_mask = case_counts < rare_threshold
    rare_share = float(np.mean(rare_mask)) if n_observed else np.nan
    singleton_case_states = int(np.sum(case_counts == 1))
    singleton_position_states = int(np.sum(position_counts == 1))

    freq_pairs = sorted(
        zip(observed_states, case_counts),
        key=lambda item: item[1],
        reverse=True,
    )
    top_states = [
        {
            "state": state,
            "case_frequency": float(case_freq),
            "position_frequency": float(position_counts[state_to_index[state]]),
        }
        for state, case_freq in freq_pairs[:10]
    ]

    summary: Dict[str, Any] = {
        "theoretical_combined_states": theoretical,
        "observed_combined_states": n_observed,
        "coverage": coverage,
        "rare_threshold": rare_threshold,
        "rare_state_share_basis": "case_frequency",
        "rare_state_share": rare_share,
        "singleton_case_states": singleton_case_states,
        "singleton_position_states": singleton_position_states,
        "min_case_frequency": float(np.min(case_counts)) if n_observed else np.nan,
        "median_case_frequency": float(np.median(case_counts)) if n_observed else np.nan,
        "max_case_frequency": float(np.max(case_counts)) if n_observed else np.nan,
        "min_position_frequency": float(np.min(position_counts)) if n_observed else np.nan,
        "median_position_frequency": float(np.median(position_counts)) if n_observed else np.nan,
        "max_position_frequency": float(np.max(position_counts)) if n_observed else np.nan,
        "top_combined_states": top_states,
        "ch_sep": ch_sep,
    }

    if n_observed >= warn_threshold_states:
        warnings.warn(
            f"The observed multidomain state space contains {n_observed} combined "
            f"states (threshold={warn_threshold_states}). IDCD and CAT computations "
            "may become slower or harder to interpret.",
            UserWarning,
            stacklevel=2,
        )
    if n_observed and rare_share >= warn_rare_share:
        warnings.warn(
            f"{rare_share:.1%} of observed combined states appear in fewer than "
            f"{rare_threshold} cases. Interpret IDCD/CAT typologies with care.",
            UserWarning,
            stacklevel=2,
        )
    if (
        n_observed >= warn_threshold_states
        and coverage < warn_coverage
    ):
        warnings.warn(
            f"Combined-state coverage is {coverage:.1%} "
            f"({n_observed}/{theoretical}). The representation is sparse.",
            UserWarning,
            stacklevel=2,
        )

    return summary


def dat_domain_contributions(
    provider: DATDistanceProvider,
    *,
    medoids: Sequence[int],
    clustering: Sequence[int],
    profile_weights: Sequence[float],
    domain_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Decompose DAT within-cluster dissimilarity by domain.

    Uses DAT-weighted domain component matrices from
    :meth:`DATDistanceProvider.weighted_per_domain_distance_to_medoids`, so
    contribution shares sum to 1 for both ``link='sum'`` and ``link='mean'``.

    ``clustering`` must be zero-based cluster indices aligned with profiles.
    """
    medoids = np.asarray(medoids, dtype=int)
    clustering = np.asarray(clustering, dtype=int)
    profile_weights = np.asarray(profile_weights, dtype=float)

    if clustering.ndim != 1:
        raise ValueError("clustering must be one-dimensional.")
    if len(clustering) != provider.n_sequences():
        raise ValueError("clustering length must match provider.n_sequences().")
    if len(profile_weights) != provider.n_sequences():
        raise ValueError("profile_weights length must match provider.n_sequences().")
    if clustering.min() < 0 or clustering.max() >= len(medoids):
        raise ValueError("clustering indices must refer to medoids positions.")

    per_domain = provider.weighted_per_domain_distance_to_medoids(medoids)
    dat_matrix = np.sum(np.stack(per_domain, axis=0), axis=0)

    names = list(domain_names) if domain_names is not None else provider.domain_names
    if len(names) != len(per_domain):
        raise ValueError("domain_names length must match the number of domains.")

    cluster_ids = list(range(len(medoids)))
    rows: List[Dict[str, Any]] = []

    def _append_rows(cluster_label: Union[int, str], mask: np.ndarray) -> None:
        weights = profile_weights[mask]
        if weights.size == 0 or float(np.sum(weights)) <= 0:
            return

        col_idx = clustering[mask]
        dat_nearest = dat_matrix[mask, col_idx]
        dat_weighted = float(np.sum(dat_nearest * weights))
        if dat_weighted <= 0:
            for domain_name in names:
                rows.append(
                    {
                        "domain": domain_name,
                        "cluster": cluster_label,
                        "weighted_distance": 0.0,
                        "contribution_share": np.nan,
                        "contribution_defined": False,
                    }
                )
            return

        for domain_name, domain_matrix in zip(names, per_domain):
            domain_nearest = domain_matrix[mask, col_idx]
            domain_weighted = float(np.sum(domain_nearest * weights))
            rows.append(
                {
                    "domain": domain_name,
                    "cluster": cluster_label,
                    "weighted_distance": domain_weighted,
                    "contribution_share": domain_weighted / dat_weighted,
                    "contribution_defined": True,
                }
            )

    _append_rows("all", np.ones(provider.n_sequences(), dtype=bool))
    for cluster_id in cluster_ids:
        _append_rows(cluster_id, clustering == cluster_id)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    for cluster_label in frame["cluster"].unique():
        subset = frame["cluster"] == cluster_label
        defined = frame.loc[subset, "contribution_defined"]
        if not bool(defined.all()):
            continue
        share_sum = float(frame.loc[subset, "contribution_share"].sum())
        if not np.isclose(share_sum, 1.0, atol=1e-6):
            raise RuntimeError(
                f"Contribution shares for cluster {cluster_label!r} sum to "
                f"{share_sum:.6f}, expected 1."
            )

    return frame


def _single_domain_dist_args(
    strategy: str,
    distance_params: Optional[Dict[str, Any]],
    *,
    domain_index: int,
) -> Dict[str, Any]:
    """Map multidomain distance params to single-domain ``clara`` dist_args."""
    from .md_clara import _default_distance_params

    strategy = strategy.lower()
    params = (
        dict(distance_params)
        if distance_params is not None
        else _default_distance_params(strategy, max(domain_index + 1, 1))
    )

    if strategy == "idcd":
        return {k: v for k, v in params.items() if k != "ch_sep"}
    if strategy == "cat":
        out = dict(params)
        for key in ("sm", "cweight"):
            val = out.get(key)
            if isinstance(val, (list, tuple)):
                out[key] = val[domain_index]
        return out
    if strategy == "dat":
        method_params = list(params.get("method_params", []))
        if not method_params:
            return _default_distance_params("idcd", 1)
        return dict(method_params[domain_index])
    raise ValueError(f"Unknown strategy {strategy!r}.")


def _best_clustering_single_domain(
    seqdata: SequenceData,
    *,
    strategy: str,
    distance_params: Optional[Dict[str, Any]],
    domain_index: int,
    k: int,
    R: int,
    sample_size: int,
    method: str,
    criteria: Sequence[str],
) -> np.ndarray:
    """Run standard CLARA on one domain (leave-one-domain-out when only one remains)."""
    from sequenzo.big_data.clara.clara import clara

    dist_args = _single_domain_dist_args(
        strategy,
        distance_params,
        domain_index=domain_index,
    )
    raw = clara(
        seqdata,
        R=R,
        sample_size=sample_size,
        kvals=[k],
        method=method,
        dist_args=dist_args,
        criteria=list(criteria),
        stability=False,
    )
    column = f"Cluster {k}"
    if column not in raw["clustering"].columns:
        raise KeyError(f"Single-domain CLARA did not return {column!r}.")
    return np.asarray(raw["clustering"][column], dtype=int)


def leave_one_domain_out_sensitivity(
    domains: List[SequenceData],
    *,
    strategy: str,
    k: int,
    R: int,
    sample_size: int,
    distance_params: Optional[Dict[str, Any]] = None,
    random_state: Optional[int] = None,
    n_jobs: int = 1,
    method: str = "crisp",
    criteria: Sequence[str] = ("distance",),
    stability: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Rerun MD-CLARA without each domain and compare partitions to the full model.

    The same ``random_state`` and CLARA tuning parameters are reused across
    runs; subsamples are not guaranteed to match because unique profile sets
    change when a domain is removed.

    When only one domain remains (e.g. two-domain data and one domain omitted),
    the reduced fit uses standard single-domain ``clara`` on the retained domain
    with strategy-consistent distance parameters (``reduced_model='clara_single_domain'``).
    """
    validate_multidomain_domains(domains)
    if len(domains) < 2:
        raise ValueError("At least two domains are required for leave-one-domain-out.")

    from .md_clara import md_clara

    full_result = md_clara(
        domains,
        strategy=strategy,
        R=R,
        sample_size=sample_size,
        kvals=[k],
        method=method,
        distance_params=distance_params,
        criteria=criteria,
        stability=stability,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    full_labels = full_result.best_clustering(k)
    full_requested_b = full_result.settings.get("requested_sample_size", sample_size)
    full_effective_b = full_result.settings.get("effective_sample_size", sample_size)
    full_n_profiles = full_result.settings.get("n_unique_profiles")

    rows: List[Dict[str, Any]] = []
    for omit_index in range(len(domains)):
        reduced = [domain for idx, domain in enumerate(domains) if idx != omit_index]
        reduced_params = _reduce_distance_params(
            distance_params,
            strategy=strategy,
            omitted_index=omit_index,
        )
        if len(reduced) >= 2:
            reduced_result = md_clara(
                reduced,
                strategy=strategy,
                R=R,
                sample_size=sample_size,
                kvals=[k],
                method=method,
                distance_params=reduced_params,
                criteria=criteria,
                stability=False,
                random_state=random_state,
                n_jobs=n_jobs,
                verbose=verbose,
            )
            reduced_labels = reduced_result.best_clustering(k)
            reduced_requested_b = reduced_result.settings.get(
                "requested_sample_size", sample_size
            )
            reduced_effective_b = reduced_result.settings.get(
                "effective_sample_size", sample_size
            )
            reduced_n_profiles = reduced_result.settings.get("n_unique_profiles")
            reduced_model = "md_clara"
        else:
            kept_index = next(i for i in range(len(domains)) if i != omit_index)
            reduced_labels = _best_clustering_single_domain(
                reduced[0],
                strategy=strategy,
                distance_params=distance_params,
                domain_index=kept_index,
                k=k,
                R=R,
                sample_size=sample_size,
                method=method,
                criteria=criteria,
            )
            reduced_requested_b = sample_size
            reduced_effective_b = sample_size
            reduced_n_profiles = None
            reduced_model = "clara_single_domain"
        metrics = _pairwise_partition_metrics(full_labels, reduced_labels)
        rows.append(
            {
                "omitted_domain": f"domain_{omit_index}",
                "omitted_domain_index": omit_index,
                "strategy": strategy.lower(),
                "k": k,
                "ari_vs_all_domains": metrics["ari"],
                "jaccard_vs_all_domains": metrics["jaccard"],
                "full_requested_sample_size": full_requested_b,
                "full_effective_sample_size": full_effective_b,
                "full_n_unique_profiles": full_n_profiles,
                "reduced_requested_sample_size": reduced_requested_b,
                "reduced_effective_sample_size": reduced_effective_b,
                "reduced_n_unique_profiles": reduced_n_profiles,
                "reduced_model": reduced_model,
            }
        )

    return pd.DataFrame(rows)


def _reduce_distance_params(
    distance_params: Optional[Dict[str, Any]],
    *,
    strategy: str,
    omitted_index: int,
) -> Optional[Dict[str, Any]]:
    if distance_params is None:
        return None

    params = dict(distance_params)
    strategy = strategy.lower()

    if strategy == "dat":
        method_params = list(params.get("method_params", []))
        if method_params:
            params["method_params"] = [
                p for idx, p in enumerate(method_params) if idx != omitted_index
            ]
        domain_weights = params.get("domain_weights")
        if domain_weights is not None:
            params["domain_weights"] = [
                w for idx, w in enumerate(domain_weights) if idx != omitted_index
            ]
    elif strategy == "cat":
        sm = params.get("sm")
        if isinstance(sm, (list, tuple)):
            params["sm"] = [s for idx, s in enumerate(sm) if idx != omitted_index]
        cweight = params.get("cweight")
        if isinstance(cweight, (list, tuple)):
            params["cweight"] = [w for idx, w in enumerate(cweight) if idx != omitted_index]

    return params


def summarize_subsample_coverage(
    subsample_diagnostics: pd.DataFrame,
) -> Dict[str, float]:
    """Summarize repetition-level rare-profile subsample coverage."""
    if subsample_diagnostics.empty:
        return {}

    coverage = subsample_diagnostics["rare_profile_coverage"].dropna()
    weight_share = subsample_diagnostics["sampled_weight_share"].dropna()

    summary: Dict[str, float] = {
        "mean_rare_profile_coverage": float(coverage.mean()) if len(coverage) else np.nan,
        "min_rare_profile_coverage": float(coverage.min()) if len(coverage) else np.nan,
        "mean_sampled_weight_share": float(weight_share.mean()) if len(weight_share) else np.nan,
        "n_repetitions": float(len(subsample_diagnostics)),
    }
    return summary


__all__ = [
    "compare_md_clara_strategies",
    "summarize_combined_state_space",
    "dat_domain_contributions",
    "leave_one_domain_out_sensitivity",
    "summarize_subsample_coverage",
]
