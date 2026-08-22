"""
@Author  : Yuqi Liang 梁彧祺
@File    : clara_engine.py
@Time    : 18/05/2026 17:23
@Desc    : 
Provider-aware CLARA engine for multidomain sequence analysis.

This module mirrors :mod:`sequenzo.big_data.clara.clara` but delegates distance
computation to a :class:`~sequenzo.multidomain.clara.distance_providers.DistanceProvider`.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from joblib import Parallel, delayed
from sequenzo.big_data.clara.clara import adjustedRandIndex, jaccardCoef
from sequenzo.big_data.clara.utils.davies_bouldin import (
    davies_bouldin_internal,
    fuzzy_davies_bouldin_internal,
)
from sequenzo.big_data.clara.utils.get_weighted_diss import get_weighted_diss
from sequenzo.clustering.k_medoids import KMedoids
from sequenzo.clustering.sequenzo_fastcluster.fastcluster import linkage
from sequenzo.clustering.sequences_to_variables.helske_regression_variables import (
    medoid_indices_from_kmedoids_result,
)
from sequenzo.define_sequence_data import SequenceData

from .distance_providers import DistanceProvider
from ._utils import (
    check_sample_size_for_k,
    one_based_to_zero_based,
    validate_kvals,
    validate_profile_weights,
)


SAMPLING_POLICIES = (
    "uniform_unique_profiles",
    "case_sample_then_aggregate",
    "frequency_proportional_profiles",
    "studer_bootstrap",
)

SAMPLING_POLICY_ALIASES = {
    "frequency_proportional_with_replacement": "studer_bootstrap",
}

STUDER_BOOTSTRAP_POLICIES = frozenset({"studer_bootstrap"})


def normalize_sampling_policy(name: str) -> str:
    """Canonical sampling-policy name, including aliases."""
    policy = str(name).lower()
    policy = SAMPLING_POLICY_ALIASES.get(policy, policy)
    if policy not in SAMPLING_POLICIES:
        raise ValueError(
            f"sampling_policy must be one of {list(SAMPLING_POLICIES)}; "
            f"got {name!r}."
        )
    return policy


def _pbm_index(k: int, total_diss: float, max_intermedoid: float) -> float:
    """
    PBM index of Studer et al. (2026) with E1=1.

    PBM = ((1/K) * (1/E) * D)^2 where E is the weighted total
    within-cluster distance and D is the maximum inter-medoid distance.
    """
    if total_diss <= 0:
        return float(np.inf)
    return float(((1.0 / k) * (1.0 / total_diss) * max_intermedoid) ** 2)


def _quality_from_distances(
    diss_to_medoids: np.ndarray,
    medoids: np.ndarray,
    *,
    method: str,
    weights: np.ndarray,
    m: float = 1.5,
) -> Tuple[float, float, float, float, float, float, np.ndarray]:
    """
    Compute total distance, average distance, DB, XB, PBM, AMS, and cluster labels.

    ``weights`` are profile frequencies (``aggWeights``), not normalized probabilities.
    """
    weight_sum = float(np.sum(weights))
    alphabeta = np.array([np.sort(row)[:2] for row in diss_to_medoids])
    denom = np.maximum(alphabeta[:, 1], alphabeta[:, 0])
    sil = np.divide(
        alphabeta[:, 1] - alphabeta[:, 0],
        denom,
        out=np.zeros_like(denom, dtype=float),
        where=denom > 0,
    )

    if method == "fuzzy":
        mexp = -1.0 / (m - 1.0)
        memb = np.power(diss_to_medoids, mexp)
        zero_dist = diss_to_medoids == 0.0
        all_med = np.sum(zero_dist, axis=1) > 0
        memb[all_med, :] = 0.0
        memb[zero_dist] = 1.0
        memb = memb / memb.sum(axis=1, keepdims=True)
        total_diss = float(
            np.sum(np.sum(np.power(memb, m) * diss_to_medoids, axis=1) * weights)
        )
        avg_diss = total_diss / weight_sum if weight_sum > 0 else float(np.mean(diss_to_medoids))
        db = fuzzy_davies_bouldin_internal(
            diss_to_medoids, memb, medoids, weights=weights
        )["db"]
        highest_memb = np.sort(memb, axis=1)[:, -2:]
        crispness = np.power(highest_memb[:, 1] - highest_memb[:, 0], 1.0)
        crisp_sum = float(np.sum(crispness * weights))
        pbm = _pbm_index(len(medoids), total_diss, float(np.max(diss_to_medoids[medoids])))
        ams = (
            float(np.sum(crispness * sil * weights) / crisp_sum)
            if crisp_sum > 0
            else 0.0
        )
        labels = memb
    else:
        labels = np.argmin(diss_to_medoids, axis=1)
        nearest = np.min(diss_to_medoids, axis=1)
        total_diss = float(np.sum(nearest * weights))
        avg_diss = (
            total_diss / weight_sum
            if weight_sum > 0
            else float(np.mean(nearest))
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            db = davies_bouldin_internal(
                diss=diss_to_medoids,
                clustering=labels,
                medoids=medoids,
                weights=weights,
            )["db"]
        pbm = _pbm_index(len(medoids), total_diss, float(np.max(diss_to_medoids[medoids])))
        ams = (
            float(np.sum(sil * weights) / weight_sum)
            if weight_sum > 0
            else float(np.mean(sil))
        )

    distmed = diss_to_medoids[medoids, :]
    distmed_flat = distmed[np.triu_indices_from(distmed, k=1)]
    minsep = float(np.min(distmed_flat)) if distmed_flat.size else 1.0
    xb = avg_diss / minsep if minsep > 0 else np.inf

    return total_diss, avg_diss, db, pbm, ams, xb, labels


def _weighted_condensed_for_linkage(
    diss_square: np.ndarray,
    sample_weights: np.ndarray,
) -> np.ndarray:
    """Apply profile-frequency weights on a square matrix and return condensed input."""
    weighted_square = get_weighted_diss(np.array(diss_square, dtype=float, copy=True), sample_weights)
    return squareform(weighted_square, checks=False)


def _within_sample_distances_for_clustering(
    provider: DistanceProvider,
    local_indices: np.ndarray,
    sample_weights: np.ndarray,
    *,
    condensed_subsample: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sample dissimilarities for KMedoids and weighted Ward linkage.

    ``condensed_subsample`` controls how the provider computes and combines
    within-subsample distances (DAT can add condensed domain vectors). Ward
    initialization needs a square matrix, so the engine expands at most once
    and passes that unweighted square matrix to KMedoids. Previously the
    condensed path expanded for weighting, condensed again, then expanded a
    second time inside KMedoids.

    Returns ``(diss_for_kmedoids, linkage_input)``: square unweighted distances
    and condensed weighted distances for linkage.
    """
    if condensed_subsample:
        diss_condensed = provider.sample_distances(local_indices, condensed=True)
        diss_square = squareform(np.asarray(diss_condensed, dtype=float), checks=False)
        del diss_condensed
    else:
        diss_square = np.asarray(
            provider.sample_distances(local_indices, condensed=False),
            dtype=float,
            order="C",
        )
    linkage_input = _weighted_condensed_for_linkage(diss_square, sample_weights)
    return diss_square, linkage_input


def _assemble_distance_to_medoids(
    provider: DistanceProvider,
    medoids: np.ndarray,
    medoid_column_cache: Dict[int, np.ndarray],
) -> Tuple[np.ndarray, int, int]:
    """
    Build N* x K all-to-medoid matrix, reusing cached columns when possible.

    Returns (matrix, cache_hits, cache_misses).
    """
    medoid_list = [int(m) for m in medoids]
    new_medoids = [m for m in medoid_list if m not in medoid_column_cache]
    cache_misses = len(new_medoids)
    cache_hits = len(medoid_list) - cache_misses

    if new_medoids:
        new_matrix = provider.distance_to_medoids(new_medoids)
        for col_idx, medoid in enumerate(new_medoids):
            medoid_column_cache[medoid] = new_matrix[:, col_idx]

    diss_full = np.column_stack([medoid_column_cache[m] for m in medoid_list])
    return diss_full, cache_hits, cache_misses


def _stability_pair_labels(
    left_labels: np.ndarray,
    right_labels: np.ndarray,
    weights: np.ndarray,
) -> Tuple[float, float]:
    """Weighted ARI and Jaccard between two profile-level partitions."""
    df = pd.DataFrame(
        {
            "left": left_labels,
            "right": right_labels,
            "w": weights,
        }
    )
    tab = df.groupby(["left", "right"])["w"].sum().unstack(fill_value=0)
    return adjustedRandIndex(tab), jaccardCoef(tab)


def _unique_with_counts_first_occurrence(draws: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collapse a with-replacement draw, preserving first-occurrence order.

    Matches ``wcAggregateCases`` on the bootstrap index vector: unique ids in
    the order they first appear, with counts equal to draw multiplicities.
    """
    uniques, first_idx, counts = np.unique(
        np.asarray(draws, dtype=int),
        return_index=True,
        return_counts=True,
    )
    order = np.argsort(first_idx)
    return uniques[order], counts[order].astype(float)


def _draw_profile_sample(
    rng: np.random.Generator,
    *,
    sampling_policy: str,
    n_profiles: int,
    sample_size: int,
    profile_weights: np.ndarray,
    disagg: np.ndarray,
    max_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw unique-profile indices and the weights used inside subsample PAM.

    Full-data evaluation always uses ``profile_weights`` (``aggWeights``).
    Subsample PAM weights differ by policy:

    * ``studer_bootstrap`` (default) uses the number of times each profile
      was drawn in this repetition (Studer et al. 2026 / ``seqclararange``);
    * the other policies use the full-data frequencies a_u of the drawn
      unique profiles.
    """
    policy = normalize_sampling_policy(sampling_policy)

    if policy == "uniform_unique_profiles":
        if sample_size > n_profiles:
            raise ValueError(
                f"sample_size ({sample_size}) cannot exceed the number of "
                f"unique multidomain profiles ({n_profiles}) when sampling "
                "without replacement."
            )
        indices = rng.choice(n_profiles, size=sample_size, replace=False)
        return indices, profile_weights[indices]

    if policy == "frequency_proportional_profiles":
        draw = min(sample_size, n_profiles)
        probs = profile_weights / float(np.sum(profile_weights))
        indices = rng.choice(n_profiles, size=draw, replace=False, p=probs)
        return indices, profile_weights[indices]

    if policy == "case_sample_then_aggregate":
        n_original = int(disagg.size)
        n_draw = min(int(sample_size), n_original)
        last = np.empty(0, dtype=int)
        for _ in range(25):
            cases = rng.choice(n_original, size=n_draw, replace=False)
            last = np.unique(disagg[cases]).astype(int, copy=False)
            if last.size >= max_k:
                return last, profile_weights[last]
        raise ValueError(
            "case_sample_then_aggregate could not obtain "
            f"max(kvals)={max_k} unique profiles from {n_draw} cases "
            f"(last draw had {last.size}). Increase sample_size."
        )

    if policy == "studer_bootstrap":
        # One draw, as in seqclararange: sample.int(..., replace=TRUE) then
        # wcAggregateCases. No redraw if the unique count is below K.
        probs = profile_weights / float(np.sum(profile_weights))
        draws = rng.choice(n_profiles, size=int(sample_size), replace=True, p=probs)
        indices, counts = _unique_with_counts_first_occurrence(draws)
        if int(indices.size) < max_k:
            raise ValueError(
                f"studer_bootstrap draw of size {sample_size} yielded only "
                f"{int(indices.size)} unique profiles, but max(kvals)={max_k}. "
                "Increase sample_size; seqclararange does not redraw."
            )
        return indices, counts

    raise ValueError(
        f"Unknown sampling_policy {sampling_policy!r}. "
        f"Choose one of: {list(SAMPLING_POLICIES)}."
    )


def _subsample_coverage_row(
    *,
    repetition: int,
    local_indices: np.ndarray,
    profile_weights: np.ndarray,
    rare_profile_threshold: int,
    sampling_policy: str,
    requested_sample_size: int,
    subsample_pam_weights: Optional[np.ndarray] = None,
    profile_true_labels: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """One repetition's rare-profile (and optional latent-cluster) coverage."""
    sampled_weights = profile_weights[local_indices]
    pam_w = (
        np.asarray(subsample_pam_weights, dtype=float)
        if subsample_pam_weights is not None
        else sampled_weights
    )
    rare_mask = profile_weights < rare_profile_threshold
    rare_indices = np.flatnonzero(rare_mask)
    n_rare = int(rare_indices.size)
    if n_rare:
        sampled_rare = int(np.intersect1d(local_indices, rare_indices).size)
        rare_coverage = sampled_rare / n_rare
    else:
        sampled_rare = 0
        rare_coverage = np.nan

    total_weight = float(np.sum(profile_weights))
    sampled_weight_share = (
        float(np.sum(sampled_weights) / total_weight) if total_weight > 0 else np.nan
    )

    row: Dict[str, Any] = {
        "repetition": repetition,
        "sampling_policy": sampling_policy,
        "requested_sample_size": int(requested_sample_size),
        "n_draws": int(requested_sample_size),
        "sampled_profiles": int(local_indices.size),
        "subsample_pam_weight_sum": float(np.sum(pam_w)),
        "max_subsample_pam_weight": float(np.max(pam_w)) if pam_w.size else np.nan,
        "max_draw_multiplicity": (
            float(np.max(pam_w))
            if pam_w.size and normalize_sampling_policy(sampling_policy) in STUDER_BOOTSTRAP_POLICIES
            else (1.0 if local_indices.size else np.nan)
        ),
        "sampled_rare_profiles": sampled_rare,
        "rare_profile_coverage": rare_coverage,
        "sampled_weight_share": sampled_weight_share,
        "min_sampled_weight": float(np.min(sampled_weights)),
        "median_sampled_weight": float(np.median(sampled_weights)),
        "max_sampled_weight": float(np.max(sampled_weights)),
    }

    if profile_true_labels is not None:
        labels = np.asarray(profile_true_labels)
        sampled_labels = labels[local_indices]
        unique_labels = np.unique(labels)
        for lab in unique_labels:
            lab_i = int(lab)
            in_full = labels == lab
            in_sample = sampled_labels == lab
            n_profiles_full = int(np.sum(in_full))
            n_profiles_sample = int(np.sum(in_sample))
            weight_full = float(np.sum(profile_weights[in_full]))
            weight_sample = float(np.sum(sampled_weights[in_sample]))
            row[f"cluster_{lab_i}_profiles_sampled"] = n_profiles_sample
            row[f"cluster_{lab_i}_profiles_full"] = n_profiles_full
            row[f"cluster_{lab_i}_weight_sampled"] = weight_sample
            row[f"cluster_{lab_i}_weight_full"] = weight_full
            row[f"cluster_{lab_i}_absent"] = int(n_profiles_sample == 0)
        rare_lab = int(unique_labels[np.argmin(
            [float(np.sum(profile_weights[labels == lab])) for lab in unique_labels]
        )])
        row["rare_latent_cluster"] = rare_lab
        row["rare_latent_profiles_sampled"] = int(row[f"cluster_{rare_lab}_profiles_sampled"])
        row["rare_latent_weight_sampled"] = float(row[f"cluster_{rare_lab}_weight_sampled"])
        row["rare_latent_absent"] = int(row[f"cluster_{rare_lab}_absent"])

    return row


def _scheduled_sample(
    entry: Any,
    *,
    sampling_policy: str,
    sample_size: int,
    n_agg: int,
    profile_weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Replay a stored subsample.

    For ``studer_bootstrap`` the entry must keep draw multiplicities: either a
    dict with ``indices`` and ``pam_weights`` (or ``counts``), or a length-``b``
    array of raw with-replacement profile draws. A unique-index list without
    weights is rejected because it would silently replace counts with ``a_u``.
    """
    policy = normalize_sampling_policy(sampling_policy)
    if isinstance(entry, dict):
        if "indices" not in entry:
            raise ValueError("scheduled subsample dict must contain 'indices'.")
        indices = np.asarray(entry["indices"], dtype=int)
        if "pam_weights" in entry:
            weights = np.asarray(entry["pam_weights"], dtype=float)
        elif "counts" in entry:
            weights = np.asarray(entry["counts"], dtype=float)
        else:
            raise ValueError(
                "scheduled subsample dict must contain 'pam_weights' or 'counts'."
            )
        if indices.shape != weights.shape:
            raise ValueError("scheduled indices and PAM weights must have the same shape.")
        if np.any(indices < 0) or np.any(indices >= n_agg):
            raise IndexError("scheduled subsample indices are out of range for N*.")
        return indices, weights

    arr = np.asarray(entry, dtype=int)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("scheduled subsample indices must be a non-empty 1-d array.")
    if np.any(arr < 0) or np.any(arr >= n_agg):
        raise IndexError("scheduled subsample indices are out of range for N*.")

    if policy in STUDER_BOOTSTRAP_POLICIES:
        if arr.size != int(sample_size):
            raise ValueError(
                "studer_bootstrap subsample_schedule must be raw draws of length "
                f"sample_size={sample_size} (got {arr.size}), or dicts with "
                "indices and pam_weights. Unique profile lists drop multiplicities."
            )
        return _unique_with_counts_first_occurrence(arr)

    indices = np.unique(arr)
    return indices, profile_weights[indices]


def _run_single_iteration(
    provider: DistanceProvider,
    *,
    sample_size: int,
    kvals: Sequence[int],
    method: str,
    aggregation: Dict[str, Any],
    rng: np.random.Generator,
    repetition_index: int = 0,
    subsample_diagnostics: bool = False,
    rare_profile_threshold: int = 5,
    use_medoid_cache: bool = True,
    condensed_subsample: bool = True,
    sampling_policy: str = "studer_bootstrap",
    scheduled_entry: Optional[Any] = None,
    profile_true_labels: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """One CLARA iteration: sample, cluster the sample, assign all sequences."""
    n_agg = provider.n_sequences()
    profile_weights = validate_profile_weights(
        aggregation["aggWeights"],
        n_profiles=n_agg,
    )
    max_k = max(kvals)
    disagg = one_based_to_zero_based(aggregation["disaggIndex"], name="disaggIndex")

    if scheduled_entry is not None:
        local_indices, sample_weights = _scheduled_sample(
            scheduled_entry,
            sampling_policy=sampling_policy,
            sample_size=sample_size,
            n_agg=n_agg,
            profile_weights=profile_weights,
        )
    else:
        if (
            sampling_policy == "uniform_unique_profiles"
            and sample_size > n_agg
        ):
            raise ValueError(
                f"sample_size ({sample_size}) cannot exceed the number of "
                f"unique multidomain profiles ({n_agg}) when sampling without replacement."
            )
        local_indices, sample_weights = _draw_profile_sample(
            rng,
            sampling_policy=sampling_policy,
            n_profiles=n_agg,
            sample_size=sample_size,
            profile_weights=profile_weights,
            disagg=disagg,
            max_k=max_k,
        )
        local_indices = np.asarray(local_indices, dtype=int)
        sample_weights = np.asarray(sample_weights, dtype=float)

    n_unique_sample = len(local_indices)

    if n_unique_sample < max_k:
        raise ValueError(
            f"Only {n_unique_sample} sampled profiles, but max(kvals)={max_k}."
        )

    diss_sample, linkage_input = _within_sample_distances_for_clustering(
        provider,
        local_indices,
        sample_weights,
        condensed_subsample=condensed_subsample,
    )
    hc = linkage(linkage_input, method="ward")

    outputs: List[Dict[str, Any]] = []
    if method != "crisp":
        raise ValueError("Only method='crisp' is supported in this CLARA engine version.")

    medoid_column_cache: Dict[int, np.ndarray] = {}
    total_cache_hits = 0
    total_cache_misses = 0
    peak_cached_columns = 0

    for k in kvals:
        # KMedoids accepts condensed input; with linkage init it expands internally.
        clustering = KMedoids(
            diss=diss_sample,
            k=k,
            initialclust=hc,
            weights=sample_weights,
            verbose=False,
        )  # diss_sample is square; Ward init would expand condensed input.
        medoid_rows = medoid_indices_from_kmedoids_result(clustering)
        medoids = local_indices[medoid_rows]

        if use_medoid_cache:
            diss_full, cache_hits, cache_misses = _assemble_distance_to_medoids(
                provider,
                medoids,
                medoid_column_cache,
            )
        else:
            diss_full = provider.distance_to_medoids(medoids)
            cache_hits = 0
            cache_misses = len(medoids)
        total_cache_hits += cache_hits
        total_cache_misses += cache_misses
        if use_medoid_cache:
            peak_cached_columns = max(peak_cached_columns, len(medoid_column_cache))

        total_diss, avg_diss, db, pbm, ams, xb, labels = _quality_from_distances(
            diss_full,
            medoids,
            method="crisp",
            weights=profile_weights,
        )

        outputs.append(
            {
                "total_diss": total_diss,
                "avg_diss": avg_diss,
                "db": db,
                "pbm": pbm,
                "ams": ams,
                "xb": xb,
                "clustering": labels,
                "medoids": medoids,
            }
        )

    del diss_sample, linkage_input

    result: Dict[str, Any] = {"k_results": outputs}
    if subsample_diagnostics:
        result["subsample"] = _subsample_coverage_row(
            repetition=repetition_index,
            local_indices=local_indices,
            profile_weights=profile_weights,
            rare_profile_threshold=rare_profile_threshold,
            sampling_policy=sampling_policy,
            requested_sample_size=sample_size,
            subsample_pam_weights=sample_weights,
            profile_true_labels=profile_true_labels,
        )
    result["medoid_cache"] = {
        "hits": total_cache_hits,
        "misses": total_cache_misses,
        "peak_cached_columns": peak_cached_columns,
    }
    return result


def clara_from_distance_provider(
    provider: DistanceProvider,
    *,
    reference_seqdata: SequenceData,
    aggregation: Optional[Dict[str, Any]] = None,
    R: int = 100,
    sample_size: Optional[int] = None,
    kvals: Optional[Sequence[int]] = None,
    method: str = "crisp",
    criteria: Sequence[str] = ("distance",),
    stability: bool = False,
    random_state: Optional[int] = None,
    n_jobs: int = -1,
    verbose: bool = True,
    subsample_diagnostics: bool = False,
    rare_profile_threshold: int = 5,
    use_medoid_cache: bool = True,
    condensed_subsample: bool = True,
    sampling_policy: str = "studer_bootstrap",
    subsample_schedule: Optional[Sequence[Any]] = None,
    profile_true_labels: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Run CLARA using a distance provider instead of a full distance matrix.

    Parameters
    ----------
    provider
        Object implementing sample and medoid-distance queries.
    reference_seqdata
        Sequence object used for case aggregation and ID alignment (typically
        the first domain before aggregation).
    condensed_subsample
        If ``True``, within-subsample distances use condensed storage (and DAT
        combines domain-level condensed vectors). If ``False``, providers return
        square ``b x b`` matrices (unoptimized path for ablation benchmarks).
    use_medoid_cache
        Reuse all-to-medoid distance columns across ``kvals`` within each
        repetition when ``True`` (default; the optimized public path).
    sampling_policy
        ``studer_bootstrap`` (default: frequency-proportional with replacement;
        subsample PAM uses draw multiplicities, matching ``seqclararange``);
        ``uniform_unique_profiles`` (profile-balanced: distinct profiles
        uniformly, subsample PAM uses full frequencies a_u);
        ``case_sample_then_aggregate``; or
        ``frequency_proportional_profiles``.
        Alias: ``frequency_proportional_with_replacement``.
    subsample_schedule
        Optional list of ``R`` stored draws used instead of the RNG. For
        ``studer_bootstrap`` each entry must be raw length-``b`` draws or a
        dict with ``indices`` and ``pam_weights``. Unique-index lists are
        valid for the other policies.
    profile_true_labels
        Optional length-``N*`` latent labels for subsample coverage diagnostics.
    """
    if n_jobs == 0:
        raise ValueError("n_jobs must not be 0.")

    kvals = validate_kvals(kvals)

    if sample_size is None:
        sample_size = 40 + 2 * max(kvals)

    if R < 1:
        raise ValueError("R must be at least 1.")
    if stability and R < 2:
        raise ValueError("stability=True requires R >= 2.")

    sampling_policy = normalize_sampling_policy(sampling_policy)

    if aggregation is None:
        raise ValueError(
            "aggregation must be provided explicitly for multidomain CLARA. "
            "Use md_clara() as the public API."
        )
    ac = dict(aggregation)

    profile_weights = validate_profile_weights(
        ac["aggWeights"],
        n_profiles=provider.n_sequences(),
    )

    check_sample_size_for_k(
        sample_size,
        kvals,
        n_unique_cases=len(profile_weights),
    )

    schedule: Optional[List[Any]] = None
    if subsample_schedule is not None:
        if len(subsample_schedule) != R:
            raise ValueError(
                f"subsample_schedule length ({len(subsample_schedule)}) must equal R={R}."
            )
        schedule = list(subsample_schedule)

    if profile_true_labels is not None:
        labels_arr = np.asarray(profile_true_labels)
        if labels_arr.shape != (provider.n_sequences(),):
            raise ValueError(
                "profile_true_labels must have length equal to the number of unique profiles."
            )
        profile_true_labels = labels_arr

    method = method.lower()
    if method != "crisp":
        raise ValueError(
            "clara_from_distance_provider currently supports method='crisp' only. "
            "Fuzzy and representativeness will be added in a later release."
        )

    criteria = tuple(c.lower() for c in criteria)
    if len(criteria) != 1 or criteria[0] != "distance":
        raise ValueError(
            "MD-CLARA selects the best repetition by minimizing total nearest-medoid "
            "distance. Pass criteria=('distance',) only."
        )
    criterion = "distance"

    if verbose:
        print("[>] Starting multidomain CLARA with distance provider.")
        print(f"  - Strategy sample size: {sample_size}, iterations: {R}")
        print(f"  - Sampling policy: {sampling_policy}")
        print(
            f"  - Within-subsample storage: "
            f"{'condensed' if condensed_subsample else 'square'}; "
            f"medoid cache: {use_medoid_cache}"
        )

    rng = np.random.default_rng(random_state)
    iteration_seeds = rng.integers(0, np.iinfo(np.int64).max, size=R)

    def _iteration(rep_index: int, seed: int) -> Dict[str, Any]:
        iter_rng = np.random.default_rng(seed)
        scheduled = None if schedule is None else schedule[rep_index]
        return _run_single_iteration(
            provider,
            sample_size=sample_size,
            kvals=kvals,
            method="crisp",
            aggregation=ac,
            rng=iter_rng,
            repetition_index=rep_index,
            subsample_diagnostics=subsample_diagnostics,
            rare_profile_threshold=rare_profile_threshold,
            use_medoid_cache=use_medoid_cache,
            condensed_subsample=condensed_subsample,
            sampling_policy=sampling_policy,
            scheduled_entry=scheduled,
            profile_true_labels=profile_true_labels,
        )

    if verbose:
        print("[>] Running CLARA iterations...")

    if n_jobs == 1:
        results = [
            _iteration(i, int(iteration_seeds[i])) for i in range(R)
        ]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_iteration)(i, int(iteration_seeds[i])) for i in range(R)
        )

    if verbose:
        print("  - Done.")
        print("[>] Aggregating iterations for each k...")

    subsample_rows: List[Dict[str, Any]] = []
    cache_hits_total = 0
    cache_misses_total = 0
    peak_cached_columns_total = 0

    collected: List[List[Dict[str, Any]]] = [[] for _ in kvals]
    for iter_result in results:
        if subsample_diagnostics and "subsample" in iter_result:
            subsample_rows.append(iter_result["subsample"])
        cache_info = iter_result.get("medoid_cache", {})
        cache_hits_total += int(cache_info.get("hits", 0))
        cache_misses_total += int(cache_info.get("misses", 0))
        peak_cached_columns_total = max(
            peak_cached_columns_total,
            int(cache_info.get("peak_cached_columns", 0)),
        )
        for k_idx, item in enumerate(iter_result["k_results"]):
            collected[k_idx].append(item)

    kret: List[Dict[str, Any]] = []

    for k_index, k_value in enumerate(kvals):
        bucket = collected[k_index]
        total_all = [d["total_diss"] for d in bucket]
        avg_all = [d["avg_diss"] for d in bucket]
        db_all = [d["db"] for d in bucket]
        pbm_all = [d["pbm"] for d in bucket]
        ams_all = [d["ams"] for d in bucket]
        xb_all = [d["xb"] for d in bucket]
        clustering_all = [d["clustering"] for d in bucket]
        med_all = [d["medoids"] for d in bucket]

        objective = total_all
        best = int(np.argmin(total_all))

        if stability:
            comparison_indices = [j for j in range(R) if j != best]

            def _stability_pair(j: int) -> Tuple[float, float]:
                left_labels = clustering_all[j]
                right_labels = clustering_all[best]
                df = pd.DataFrame(
                    {
                        "left": left_labels,
                        "right": right_labels,
                        "w": ac["aggWeights"],
                    }
                )
                tab = df.groupby(["left", "right"])["w"].sum().unstack(fill_value=0)
                return adjustedRandIndex(tab), jaccardCoef(tab)

            if n_jobs == 1:
                arilist = [_stability_pair(j) for j in comparison_indices]
            else:
                arilist = Parallel(n_jobs=n_jobs)(
                    delayed(_stability_pair)(j) for j in comparison_indices
                )
            arimatrix = pd.DataFrame(arilist, columns=["ARI", "JC"])
            ari_vs_best = arimatrix.iloc[:, 0].to_numpy(dtype=float)
            jc_vs_best = arimatrix.iloc[:, 1].to_numpy(dtype=float)
            ari08 = int(np.sum(ari_vs_best >= 0.8))
            jc08 = int(np.sum(jc_vs_best >= 0.8))
            n_comparisons = len(comparison_indices)

            all_pair_ari: List[float] = []
            all_pair_jc: List[float] = []
            for i in range(R):
                for j in range(i + 1, R):
                    ari_ij, jc_ij = _stability_pair_labels(
                        clustering_all[i], clustering_all[j], ac["aggWeights"]
                    )
                    all_pair_ari.append(ari_ij)
                    all_pair_jc.append(jc_ij)

            n_top = max(1, int(np.ceil(0.2 * R)))
            top_order = np.argsort(np.asarray(total_all, dtype=float))[:n_top]
            top_vs_best = [int(idx) for idx in top_order if int(idx) != best]
            if top_vs_best:
                top_pairs = [
                    _stability_pair_labels(
                        clustering_all[j], clustering_all[best], ac["aggWeights"]
                    )
                    for j in top_vs_best
                ]
                mean_ari_top20 = float(np.mean([p[0] for p in top_pairs]))
                mean_jc_top20 = float(np.mean([p[1] for p in top_pairs]))
            else:
                mean_ari_top20 = np.nan
                mean_jc_top20 = np.nan

            stability_info = {
                "ari": ari_vs_best,
                "jc": jc_vs_best,
                "ari08": ari08,
                "jc08": jc08,
                "mean_ari": float(np.mean(ari_vs_best)) if n_comparisons else np.nan,
                "mean_jc": float(np.mean(jc_vs_best)) if n_comparisons else np.nan,
                "n_comparisons": n_comparisons,
                "mean_ari_all": float(np.mean(all_pair_ari)) if all_pair_ari else np.nan,
                "mean_jc_all": float(np.mean(all_pair_jc)) if all_pair_jc else np.nan,
                "mean_ari_top20": mean_ari_top20,
                "mean_jc_top20": mean_jc_top20,
                "n_ari_ge_0.9": int(np.sum(ari_vs_best >= 0.9)),
                "n_ari_ge_0.8": ari08,
                "n_ari_ge_0.7": int(np.sum(ari_vs_best >= 0.7)),
                "n_jc_ge_0.8": jc08,
            }
        else:
            arimatrix = None
            ari08 = np.nan
            jc08 = np.nan
            stability_info = None

        best_clustering = clustering_all[best]
        disag = np.full(reference_seqdata.seqdata.shape[0], -1, dtype=int)
        for row_idx, agg_idx in enumerate(
            one_based_to_zero_based(ac["disaggIndex"], name="disaggIndex")
        ):
            disag[row_idx] = int(best_clustering[agg_idx]) + 1

        evol = np.minimum.accumulate(objective)

        bestcluster = {
            "medoids": one_based_to_zero_based(
                np.asarray(ac["aggIndex"], dtype=int)[med_all[best]],
                name="aggIndex",
            ),
            "medoids_agg": med_all[best],
            "profile_clustering": best_clustering,
            "clustering": disag,
            "membership": None,
            "evol_diss": evol,
            "iter_objective": objective,
            "objective": objective[best],
            "iteration": best,
            "arimatrix": arimatrix,
            "stability": stability_info,
            "criteria": criterion,
            "method": method,
            "total_diss": total_all[best],
            "avg_dist": avg_all[best],
            "pbm": pbm_all[best],
            "db": db_all[best],
            "xb": xb_all[best],
            "ams": ams_all[best],
            "ari08": ari08,
            "jc08": jc08,
            "R": R,
            "k": k_value,
            "sample_size": sample_size,
        }

        kret.append(
            {
                "k": k_value,
                "criteria": criterion,
                "stats": [
                    bestcluster["total_diss"],
                    bestcluster["avg_dist"],
                    bestcluster["pbm"],
                    bestcluster["db"],
                    bestcluster["xb"],
                    bestcluster["ams"],
                    bestcluster["ari08"],
                    bestcluster["jc08"],
                    best,
                ],
                "bestcluster": bestcluster,
            }
        )

    packaged = _package_clara_output(
        kret,
        kvals=kvals,
        criterion=criterion,
        method=method,
        stability=stability,
        reference_seqdata=reference_seqdata,
        sample_size=sample_size,
        R=R,
    )
    if subsample_diagnostics and subsample_rows:
        packaged["subsample_diagnostics"] = pd.DataFrame(subsample_rows)
    packaged["medoid_cache_stats"] = {
        "hits": cache_hits_total,
        "misses": cache_misses_total,
        "peak_cached_columns": peak_cached_columns_total,
    }
    packaged["condensed_subsample"] = condensed_subsample
    packaged["use_medoid_cache"] = use_medoid_cache
    packaged["sampling_policy"] = sampling_policy
    return packaged


def _package_clara_output(
    kret: List[Dict[str, Any]],
    *,
    kvals: Sequence[int],
    criterion: str,
    method: str,
    stability: bool,
    reference_seqdata: SequenceData,
    sample_size: int,
    R: int,
) -> Dict[str, Any]:
    """Build clustering/stats structures for a single criterion."""
    clustering = pd.DataFrame(
        -1,
        index=reference_seqdata.ids,
        columns=[f"Cluster {k}" for k in kvals],
        dtype=int,
    )
    stats = np.full((len(kvals), 9), -1.0, dtype=float)
    clara_dict: Dict[int, Any] = {}

    for entry in kret:
        k = entry["k"]
        pos = kvals.index(k)
        stats[pos, :] = np.array(entry["stats"], dtype=float)
        clara_dict[k] = entry["bestcluster"]
        clustering.iloc[:, pos] = entry["bestcluster"]["clustering"]

    stats_df = pd.DataFrame(
        stats,
        columns=[
            "total_diss",
            "avg_dist",
            "pbm",
            "db",
            "xb",
            "ams",
            "ari08",
            "jc08",
            "best_iter",
        ],
    )
    stats_df.insert(0, "k", list(kvals))
    stats_df["criterion"] = criterion

    return {
        "kvals": list(kvals),
        "clara": clara_dict,
        "clustering": clustering,
        "stats": stats_df,
        "stability": stability,
        "method": method,
        "sample_size": sample_size,
        "R": R,
        "criterion": criterion,
    }


__all__ = [
    "clara_from_distance_provider",
    "_assemble_distance_to_medoids",
    "SAMPLING_POLICIES",
    "STUDER_BOOTSTRAP_POLICIES",
    "normalize_sampling_policy",
    "_pbm_index",
    "_draw_profile_sample",
]
