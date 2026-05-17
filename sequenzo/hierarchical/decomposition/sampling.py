"""
@Author  : 梁彧祺 Yuqi Liang
@File    : sampling.py
@Time    : 15/04/2026 11:40
@Desc    :
    Sampling-based hierarchical analysis for large relational sequence sets.

    Three sampling strategies:

    - ``sampling_unit="sequence"`` (default): subsample sequences (up to
      ``max_sequences``), compute a full distance matrix on that subset, then draw
      random pairwise comparisons from its upper triangle.

    - ``sampling_unit="pair"``: draw unordered sequence pairs directly from the
      full set of ``n_pairs`` sequences and compute only those distances (no full
      matrix). Currently supports ``method="HAM"`` only.

    - ``sampling_unit="structural"``: stratified quotas for same level-1,
      same level-2, and baseline pairs (see :func:`sample_structural_pairwise_distances`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd

from ..data import RelationalSequenceData, RelationalSequenceRecord
from ..distances import compute_relational_distance_matrix
from .marginal import (
    HierarchicalDecompositionResult,
    LevelDiscrepancyResult,
    permutation_test_level_effect,
    sequence_discrepancy_by_level,
)

SamplingUnit = Literal["sequence", "pair", "structural"]


@dataclass
class SampledPairwiseDistances:
    """
    Random sample of unordered pair distances with hierarchical flags.

    Attributes
    ----------
    i_index, j_index : ndarray
        Row indices into the **original** full pair list (length ``n_pairs``).
    distance : ndarray
        Sampled pairwise distances.
    level_1_ids, level_2_ids : ndarray
        Full-length identifier arrays for all ``n_pairs`` sequences (not subsampled).
    n_pairs : int
        Total number of pair-level sequences in the original data.
    n_sampled : int
        Number of pairwise comparisons in this sample.
    sampling_unit : str
        ``"sequence"`` or ``"pair"`` — see :func:`describe_sampling_scheme`.
    """

    i_index: np.ndarray
    j_index: np.ndarray
    distance: np.ndarray
    level_1_ids: np.ndarray
    level_2_ids: np.ndarray
    n_pairs: int
    n_sampled: int
    method: str = ""
    sampling_unit: SamplingUnit = "sequence"
    details: Dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        """Human-readable summary of how this sample was constructed."""
        return describe_sampling_scheme(self)


def sampling_scheme_description(sampling_unit: SamplingUnit = "sequence") -> str:
    """
    Static description of a sampling design (no :class:`SampledPairwiseDistances` needed).

    Use when documenting approximation limits in benchmarks or supplementary text.
    """
    if sampling_unit == "sequence":
        return (
            "Sampling unit: sequence\n"
            "Mode: sequence subsampling + pairwise sampling within subsample.\n"
            "Note: This is NOT direct sampling from all unordered pairs among all "
            "sequences. Rare level combinations may be under-represented if missed "
            "by the sequence subsample."
        )
    if sampling_unit == "structural":
        return (
            "Sampling unit: structural\n"
            "Mode: stratified pair sampling for hierarchical decomposition.\n"
            "Draws target counts for: same level-1 / different level-2, "
            "different level-1 / same level-2, and cross-level baseline pairs."
        )
    return (
        "Sampling unit: pair\n"
        "Mode: direct pair sampling from all unordered sequence pairs.\n"
        "Each draw picks (i, j) with i < j among all original sequences."
    )


def describe_sampling_scheme(sample: SampledPairwiseDistances) -> str:
    """
    Explain whether the sample used sequence subsampling or direct pair sampling.
    """
    unit = sample.sampling_unit
    d = sample.details
    lines = [
        f"Sampling unit: {unit}",
        f"Original sequences (pairs): {sample.n_pairs:,}",
        f"Sampled pairwise comparisons: {sample.n_sampled:,}",
        f"Distance method: {sample.method or '—'}",
    ]
    if unit == "sequence":
        lines.extend(
            [
                "",
                "Mode: sequence subsampling + pairwise sampling within subsample.",
                (
                    f"  • Sequences used for distance computation: "
                    f"{d.get('n_sequences_in_matrix', '—'):,}"
                ),
                (
                    f"  • Pairwise draws from upper triangle of "
                    f"{d.get('n_sequences_in_matrix', '—')}×"
                    f"{d.get('n_sequences_in_matrix', '—')} matrix."
                ),
                "",
                "Note: This is NOT direct sampling from all unordered pairs among",
                f"all {sample.n_pairs:,} sequences. Rare regions/CPCs may be under-",
                "represented if they are missed by the sequence subsample.",
            ]
        )
    elif unit == "structural":
        sc = d.get("strata_counts", {})
        lines.extend(
            [
                "",
                "Mode: stratified structural pair sampling.",
                f"  • Same level-1 draws: {sc.get('same_level_1', '—'):,}",
                f"  • Same level-2 draws: {sc.get('same_level_2', '—'):,}",
                f"  • Baseline draws: {sc.get('baseline', '—'):,}",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Mode: direct pair sampling from all unordered sequence pairs.",
                "  • Each draw picks (i, j) with i < j among all original sequences.",
                "  • Distances computed one pair at a time (no full n×n matrix).",
            ]
        )
        if d.get("duplicate_pairs_in_sample", 0) > 0:
            lines.append(
                f"  • Duplicate index pairs in sample: {d['duplicate_pairs_in_sample']}"
            )
    return "\n".join(lines)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _hamming_distance_records(
    rec_i: RelationalSequenceRecord,
    rec_j: RelationalSequenceRecord,
    *,
    normalized: bool = True,
) -> float:
    """Position-wise Hamming distance between two sequences (lightweight)."""
    s1, s2 = rec_i.sequence, rec_j.sequence
    if len(s1) != len(s2):
        raise ValueError(
            "Pairwise HAM requires equal sequence length; align time grids first."
        )
    n = len(s1)
    if n == 0:
        return 0.0
    diff = 0
    valid = 0
    for a, b in zip(s1, s2):
        if _is_missing(a) or _is_missing(b):
            continue
        valid += 1
        if a != b:
            diff += 1
    if valid == 0:
        return 0.0
    return float(diff / valid) if normalized else float(diff)


def _subsample_sequence_data(
    sequence_data: RelationalSequenceData,
    max_sequences: int,
    random_state: Optional[int],
) -> tuple[RelationalSequenceData, np.ndarray]:
    """Return a subset of sequences and the selected row indices into the original list."""
    n = sequence_data.n_pairs
    if n <= max_sequences:
        return sequence_data, np.arange(n)

    rng = np.random.default_rng(random_state)
    idx = np.sort(rng.choice(n, size=max_sequences, replace=False))
    records = [sequence_data.records[i] for i in idx]
    sub = RelationalSequenceData(
        records=records,
        level_1_col=sequence_data.level_1_col,
        level_2_col=sequence_data.level_2_col,
        time_col=sequence_data.time_col,
        state_col=sequence_data.state_col,
        pair_separator=sequence_data.pair_separator,
    )
    return sub, idx


def _sample_via_sequence_subsampling(
    sequence_data: RelationalSequenceData,
    original_n: int,
    original_level_1_ids: np.ndarray,
    original_level_2_ids: np.ndarray,
    *,
    method: str,
    representation: str,
    n_pair_samples: Optional[int],
    max_sequences: int,
    random_state: Optional[int],
    compute_full_if_small: bool,
    distance_params: Dict[str, Any],
) -> SampledPairwiseDistances:
    """Sequence subsample → full matrix → random upper-triangle pairs."""
    n = original_n
    rng = np.random.default_rng(random_state)

    if not compute_full_if_small or n > max_sequences:
        sub_data, row_map = _subsample_sequence_data(
            sequence_data, max_sequences, random_state
        )
    else:
        sub_data = sequence_data
        row_map = np.arange(n)

    dist = compute_relational_distance_matrix(
        sub_data,
        method=method,
        representation=representation,
        **distance_params,
    )
    matrix = dist.matrix
    m = matrix.shape[0]
    n_possible = m * (m - 1) // 2
    n_draw = min(n_pair_samples or n_possible, n_possible)

    tri_i, tri_j = np.triu_indices(m, k=1)
    if n_draw < n_possible:
        pick = rng.choice(len(tri_i), size=n_draw, replace=False)
        tri_i, tri_j = tri_i[pick], tri_j[pick]

    distances = matrix[tri_i, tri_j]

    return SampledPairwiseDistances(
        i_index=row_map[tri_i],
        j_index=row_map[tri_j],
        distance=distances,
        level_1_ids=original_level_1_ids,
        level_2_ids=original_level_2_ids,
        n_pairs=original_n,
        n_sampled=len(distances),
        method=dist.method,
        sampling_unit="sequence",
        details={
            "n_sequences_in_matrix": m,
            "row_map": row_map,
            "subsampled_from_n": n,
            "n_pair_samples_requested": n_pair_samples,
        },
    )


def _sample_via_direct_pairs(
    sequence_data: RelationalSequenceData,
    original_n: int,
    original_level_1_ids: np.ndarray,
    original_level_2_ids: np.ndarray,
    *,
    method: str,
    n_pair_samples: int,
    random_state: Optional[int],
) -> SampledPairwiseDistances:
    """Draw (i, j) directly from all unordered pairs; compute HAM per pair."""
    method_upper = method.upper()
    if method_upper not in {"HAM", "DHD"}:
        raise ValueError(
            f"sampling_unit='pair' currently supports method='HAM' (or 'DHD') only; "
            f"got {method!r}. Use sampling_unit='sequence' for other methods."
        )

    rng = np.random.default_rng(random_state)
    n_possible = original_n * (original_n - 1) // 2
    n_draw = min(n_pair_samples, n_possible)

    i_list = []
    j_list = []
    dist_list = []
    seen: set[tuple[int, int]] = set()
    n_duplicate = 0
    records = sequence_data.records
    max_attempts = n_draw * 20
    attempts = 0

    while len(dist_list) < n_draw and attempts < max_attempts:
        attempts += 1
        i = int(rng.integers(0, original_n - 1))
        j = int(rng.integers(i + 1, original_n))
        key = (i, j)
        if key in seen:
            n_duplicate += 1
            continue
        seen.add(key)
        d = _hamming_distance_records(records[i], records[j])
        i_list.append(i)
        j_list.append(j)
        dist_list.append(d)

    if len(dist_list) < n_draw:
        warnings_msg = (
            f"Could only draw {len(dist_list)} unique pairs after {attempts} attempts "
            f"(requested {n_draw})."
        )
        import warnings

        warnings.warn(warnings_msg, UserWarning, stacklevel=2)

    return SampledPairwiseDistances(
        i_index=np.asarray(i_list, dtype=int),
        j_index=np.asarray(j_list, dtype=int),
        distance=np.asarray(dist_list, dtype=float),
        level_1_ids=original_level_1_ids,
        level_2_ids=original_level_2_ids,
        n_pairs=original_n,
        n_sampled=len(dist_list),
        method=method_upper,
        sampling_unit="pair",
        details={
            "n_possible_pairs": n_possible,
            "n_pair_samples_requested": n_pair_samples,
            "duplicate_pairs_in_sample": n_duplicate,
            "sampling_attempts": attempts,
        },
    )


def _draw_index_pair(
    rng: np.random.Generator,
    pool: list[int],
) -> Optional[tuple[int, int]]:
    if len(pool) < 2:
        return None
    i, j = rng.choice(pool, size=2, replace=False)
    return int(min(i, j)), int(max(i, j))


def sample_structural_pairwise_distances(
    sequence_data: RelationalSequenceData,
    *,
    n_same_level_1: int = 200_000,
    n_same_level_2: int = 200_000,
    n_baseline: int = 400_000,
    method: str = "HAM",
    random_state: Optional[int] = 123,
) -> SampledPairwiseDistances:
    """
    Stratified pairwise sampling for scalable hierarchical decomposition.

    Draws separate quotas for:

    - same level-1, different level-2
    - different level-1, same level-2
    - different level-1, different level-2 (baseline)

    Currently supports ``method="HAM"`` (normalized Hamming).
    """
    method_upper = method.upper()
    if method_upper not in {"HAM", "DHD"}:
        raise ValueError(
            "sample_structural_pairwise_distances supports method='HAM' or 'DHD' only."
        )

    rng = np.random.default_rng(random_state)
    records = sequence_data.records
    n = len(records)
    l1 = sequence_data.level_1_ids
    l2 = sequence_data.level_2_ids

    from collections import defaultdict

    l1_index: Dict[Any, List[int]] = defaultdict(list)
    l2_index: Dict[Any, List[int]] = defaultdict(list)
    for idx in range(n):
        l1_index[l1[idx]].append(idx)
        l2_index[l2[idx]].append(idx)

    l1_pools = [p for p in l1_index.values() if len(p) >= 2]
    l2_pools = [p for p in l2_index.values() if len(p) >= 2]

    i_list: list[int] = []
    j_list: list[int] = []
    dist_list: list[float] = []
    seen: set[tuple[int, int]] = set()
    strata_counts = {
        "same_level_1": 0,
        "same_level_2": 0,
        "baseline": 0,
    }

    def _append_pair(i: int, j: int, stratum: str) -> bool:
        key = (i, j) if i < j else (j, i)
        if key in seen:
            return False
        seen.add(key)
        d = _hamming_distance_records(records[i], records[j])
        i_list.append(key[0])
        j_list.append(key[1])
        dist_list.append(d)
        strata_counts[stratum] += 1
        return True

    def _fill_stratum(target: int, stratum: str, draw_fn) -> None:
        attempts = 0
        max_attempts = max(target * 30, 1000)
        while strata_counts[stratum] < target and attempts < max_attempts:
            attempts += 1
            pair = draw_fn()
            if pair is None:
                break
            _append_pair(pair[0], pair[1], stratum)

    def _draw_same_l1() -> Optional[tuple[int, int]]:
        if not l1_pools:
            return None
        pool = l1_pools[int(rng.integers(0, len(l1_pools)))]
        pair = _draw_index_pair(rng, pool)
        if pair is None:
            return None
        i, j = pair
        if l2[i] == l2[j]:
            return None
        return i, j

    def _draw_same_l2() -> Optional[tuple[int, int]]:
        if not l2_pools:
            return None
        pool = l2_pools[int(rng.integers(0, len(l2_pools)))]
        pair = _draw_index_pair(rng, pool)
        if pair is None:
            return None
        i, j = pair
        if l1[i] == l1[j]:
            return None
        return i, j

    def _draw_baseline() -> Optional[tuple[int, int]]:
        i = int(rng.integers(0, n - 1))
        j = int(rng.integers(i + 1, n))
        if l1[i] == l1[j] or l2[i] == l2[j]:
            return None
        return i, j

    _fill_stratum(int(n_same_level_1), "same_level_1", _draw_same_l1)
    _fill_stratum(int(n_same_level_2), "same_level_2", _draw_same_l2)
    _fill_stratum(int(n_baseline), "baseline", _draw_baseline)

    return SampledPairwiseDistances(
        i_index=np.asarray(i_list, dtype=int),
        j_index=np.asarray(j_list, dtype=int),
        distance=np.asarray(dist_list, dtype=float),
        level_1_ids=sequence_data.level_1_ids.copy(),
        level_2_ids=sequence_data.level_2_ids.copy(),
        n_pairs=n,
        n_sampled=len(dist_list),
        method=method_upper,
        sampling_unit="structural",
        details={
            "n_same_level_1_requested": n_same_level_1,
            "n_same_level_2_requested": n_same_level_2,
            "n_baseline_requested": n_baseline,
            "strata_counts": strata_counts,
        },
    )


def _mean_distance_for_label(structural: pd.DataFrame, label_substr: str) -> float:
    mask = structural["comparison_type"].str.contains(label_substr, case=False, regex=False)
    if not mask.any():
        return float("nan")
    return float(structural.loc[mask, "mean_distance"].iloc[0])


def _approximate_level_discrepancy(
    structural: pd.DataFrame,
    *,
    grouping_variable: str,
    same_label: str,
    baseline_label: str = "baseline",
) -> LevelDiscrepancyResult:
    """Contrast-based pseudo-R² from stratified structural distance means."""
    same_mean = _mean_distance_for_label(structural, same_label)
    baseline_mean = _mean_distance_for_label(structural, baseline_label)
    if not np.isfinite(same_mean) or not np.isfinite(baseline_mean):
        pseudo_r2 = float("nan")
    elif baseline_mean <= 0:
        pseudo_r2 = 0.0
    else:
        pseudo_r2 = float(np.clip((baseline_mean - same_mean) / baseline_mean, 0.0, 1.0))

    return LevelDiscrepancyResult(
        grouping_variable=grouping_variable,
        total_discrepancy=baseline_mean if np.isfinite(baseline_mean) else float("nan"),
        within_group_discrepancy=same_mean if np.isfinite(same_mean) else float("nan"),
        between_group_discrepancy=float("nan"),
        pseudo_r2=pseudo_r2,
        pseudo_f=float("nan"),
        p_value=float("nan"),
        n_groups=0,
        details={
            "approximation": "structural_contrast",
            "same_mean": same_mean,
            "baseline_mean": baseline_mean,
        },
    )


def hierarchical_sequence_discrepancy_from_sample(
    sample: SampledPairwiseDistances,
    *,
    level_1_name: str = "level_1",
    level_2_name: str = "level_2",
) -> HierarchicalDecompositionResult:
    """
    Build a decomposition summary from :class:`SampledPairwiseDistances`.

    Intended for ``sampling_unit="structural"``. Marginal pseudo-R² values are
    contrast-based structural approximations, **not** the exact Gower Type-III
    additive decomposition from a full distance matrix. The joint explained share
    and residual share are diagnostic proxies derived from stratified mean
    distances (``1 - min(same L1, same L2) / baseline``), not exact
    ``pi_joint`` / ``pi_res`` from :func:`hierarchical_sequence_discrepancy`.
    """
    structural = summarize_distance_by_structure_sampled(sample)
    level_1 = _approximate_level_discrepancy(
        structural,
        grouping_variable=level_1_name,
        same_label="Same level-1",
    )
    level_2 = _approximate_level_discrepancy(
        structural,
        grouping_variable=level_2_name,
        same_label="Different level-1, same level-2",
    )

    same_l1 = _mean_distance_for_label(structural, "Same level-1")
    same_l2 = _mean_distance_for_label(structural, "same level-2")
    baseline = _mean_distance_for_label(structural, "baseline")
    if np.isfinite(baseline) and baseline > 0:
        joint_proxy = float(
            np.clip(1.0 - min(same_l1, same_l2) / baseline, 0.0, 1.0)
        )
        residual_share = float(np.clip(1.0 - joint_proxy, 0.0, 1.0))
    else:
        joint_proxy = float("nan")
        residual_share = float("nan")

    return HierarchicalDecompositionResult(
        level_1=level_1,
        level_2=level_2,
        joint_pseudo_r2=joint_proxy,
        residual_share=residual_share,
        structural_summary=structural,
        additive=None,
        crossed=None,
        method="structural_sample",
    )


def sample_pairwise_distances(
    sequence_data: RelationalSequenceData,
    *,
    method: str = "HAM",
    representation: str = "state",
    n_pair_samples: Optional[int] = 1_000_000,
    max_sequences: int = 4_000,
    random_state: Optional[int] = 123,
    compute_full_if_small: bool = True,
    sampling_unit: SamplingUnit = "sequence",
    n_same_level_1: int = 200_000,
    n_same_level_2: int = 200_000,
    n_baseline: int = 400_000,
    **distance_params: Any,
) -> SampledPairwiseDistances:
    """
    Estimate large-scale distance structure without a full ``n_pairs × n_pairs`` matrix.

    Parameters
    ----------
    sequence_data : RelationalSequenceData
    method, representation
        Used for distance computation. ``representation`` applies only when
        ``sampling_unit="sequence"``.
    n_pair_samples : int, optional
        Target number of unordered pairwise comparisons to draw.
    max_sequences : int
        When ``sampling_unit="sequence"``, cap on sequences for the intermediate
        full distance matrix.
    random_state : int, optional
        RNG seed.
    compute_full_if_small : bool
        When ``sampling_unit="sequence"``, use all sequences without subsampling
        if ``n_pairs <= max_sequences``.
    sampling_unit : ``"sequence"``, ``"pair"``, or ``"structural"``
        - ``"sequence"``: subsample sequences, build a matrix on the subsample,
          then sample pairwise distances within it. Does **not** draw uniformly
          from all ``n_pairs choose 2`` pairs in the full data.
        - ``"pair"``: draw ``(i, j)`` directly from all original sequences and
          compute each distance individually (HAM only; no full matrix).
        - ``"structural"``: stratified quotas for same level-1 / same level-2 /
          baseline pairs (see :func:`sample_structural_pairwise_distances`).
    n_same_level_1, n_same_level_2, n_baseline : int
        Quotas when ``sampling_unit="structural"`` (ignored otherwise).

    Returns
    -------
    SampledPairwiseDistances
        ``i_index`` / ``j_index`` index into the **original** sequence list;
        ``level_1_ids`` / ``level_2_ids`` are always the full original arrays.

    See Also
    --------
    describe_sampling_scheme
        Text summary of the sampling design.
    sample_structural_pairwise_distances
        Direct structural stratified sampling.
    """
    if sampling_unit == "structural":
        return sample_structural_pairwise_distances(
            sequence_data,
            n_same_level_1=n_same_level_1,
            n_same_level_2=n_same_level_2,
            n_baseline=n_baseline,
            method=method,
            random_state=random_state,
        )

    original_level_1_ids = sequence_data.level_1_ids.copy()
    original_level_2_ids = sequence_data.level_2_ids.copy()
    original_n = sequence_data.n_pairs

    if sampling_unit == "pair":
        n_draw = n_pair_samples or 1_000_000
        return _sample_via_direct_pairs(
            sequence_data,
            original_n,
            original_level_1_ids,
            original_level_2_ids,
            method=method,
            n_pair_samples=n_draw,
            random_state=random_state,
        )

    return _sample_via_sequence_subsampling(
        sequence_data,
        original_n,
        original_level_1_ids,
        original_level_2_ids,
        method=method,
        representation=representation,
        n_pair_samples=n_pair_samples,
        max_sequences=max_sequences,
        random_state=random_state,
        compute_full_if_small=compute_full_if_small,
        distance_params=distance_params,
    )


def summarize_distance_by_structure_sampled(
    sample: SampledPairwiseDistances,
) -> pd.DataFrame:
    """Structural distance summary from :class:`SampledPairwiseDistances`."""
    l1_i = sample.level_1_ids[sample.i_index]
    l1_j = sample.level_1_ids[sample.j_index]
    l2_i = sample.level_2_ids[sample.i_index]
    l2_j = sample.level_2_ids[sample.j_index]

    same_l1 = l1_i == l1_j
    same_l2 = l2_i == l2_j
    diff_l1 = ~same_l1
    diff_l2 = ~same_l2

    buckets = {
        "Same level-1, different level-2": same_l1 & diff_l2,
        "Different level-1, same level-2": diff_l1 & same_l2,
        "Different level-1, different level-2 (baseline)": diff_l1 & diff_l2,
        "Same level-1 and level-2 (duplicate pair)": same_l1 & same_l2,
    }

    rows = []
    for label, mask in buckets.items():
        if mask.sum() == 0:
            rows.append(
                {
                    "comparison_type": label,
                    "mean_distance": np.nan,
                    "std_distance": np.nan,
                    "n_pairs": 0,
                }
            )
        else:
            d = sample.distance[mask]
            rows.append(
                {
                    "comparison_type": label,
                    "mean_distance": float(np.mean(d)),
                    "std_distance": float(np.std(d, ddof=1)) if len(d) > 1 else 0.0,
                    "n_pairs": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows)


def sequence_discrepancy_by_level_sampled(
    sequence_data: RelationalSequenceData,
    group_labels: np.ndarray,
    *,
    grouping_variable: str = "group",
    max_sequences: int = 4_000,
    random_state: Optional[int] = 123,
    R: int = 0,
    squared: bool = False,
    **distance_params: Any,
) -> LevelDiscrepancyResult:
    """
    Pseudo-R² for one level using **sequence subsampling** (scalable).

    Builds a full distance matrix on at most ``max_sequences`` sequences, then
    runs standard discrepancy analysis. This is sequence subsampling, not direct
    pairwise sampling from all sequences.
    """
    sub, idx = _subsample_sequence_data(
        sequence_data, max_sequences, random_state
    )
    dist = compute_relational_distance_matrix(
        sequence_data=sub,
        **distance_params,
    )
    sub_groups = np.asarray(group_labels)[idx]
    result = sequence_discrepancy_by_level(
        dist,
        sub_groups,
        grouping_variable=grouping_variable,
        R=R,
        squared=squared,
    )
    result.details["sampling_unit"] = "sequence"
    result.details["subsample_indices"] = idx
    result.details["max_sequences"] = max_sequences
    return result


def permutation_test_level_effect_sampled(
    sequence_data: RelationalSequenceData,
    group_labels: np.ndarray,
    *,
    n_perm: int = 999,
    max_sequences: int = 4_000,
    random_state: Optional[int] = 123,
    squared: bool = False,
    **distance_params: Any,
) -> Dict[str, Any]:
    """Permutation test for one level using sequence subsampling."""
    sub, idx = _subsample_sequence_data(
        sequence_data, max_sequences, random_state
    )
    dist = compute_relational_distance_matrix(sub, **distance_params)
    out = permutation_test_level_effect(
        dist,
        np.asarray(group_labels)[idx],
        n_perm=n_perm,
        random_state=random_state,
        squared=squared,
    )
    out["sampling_unit"] = "sequence"
    out["subsample_indices"] = idx
    out["max_sequences"] = max_sequences
    return out
