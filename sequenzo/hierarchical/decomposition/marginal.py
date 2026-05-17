"""
@Author  : 梁彧祺 Yuqi Liang
@File    : marginal.py
@Time    : 09/04/2026 15:48
@Desc    :
    Decomposition of sequence dissimilarity by hierarchical structure.

    Reuses discrepancy-analysis pseudo-R² machinery where appropriate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from sequenzo.discrepancy_analysis import (
    overall_discrepancy,
    single_factor_association,
)
from sequenzo.discrepancy_analysis.stats.multifactor_association import (
    build_multifactor_design,
    distance_multifactor_anova,
)

from .crossed import (
    AdditiveDecompositionResult,
    CrossedDecompositionResult,
    additive_sequence_discrepancy,
    crossed_sequence_discrepancy,
)
from ..distances import RelationalDistanceMatrix


@dataclass
class StructuralDistanceSummary:
    """Mean distances by pair relationship type."""

    comparison_type: str
    mean_distance: float
    n_pairs: int
    std_distance: float = np.nan


@dataclass
class LevelDiscrepancyResult:
    """Pseudo-R² style discrepancy for one grouping level."""

    grouping_variable: str
    total_discrepancy: float
    within_group_discrepancy: float
    between_group_discrepancy: float
    pseudo_r2: float
    pseudo_f: float
    p_value: float
    n_groups: int
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HierarchicalDecompositionResult:
    """Combined hierarchical decomposition outputs."""

    level_1: LevelDiscrepancyResult
    level_2: LevelDiscrepancyResult
    joint_pseudo_r2: float
    residual_share: float
    structural_summary: pd.DataFrame
    multifactor_table: Optional[pd.DataFrame] = None
    additive: Optional[AdditiveDecompositionResult] = None
    crossed: Optional[CrossedDecompositionResult] = None
    method: str = "additive"


def _upper_triangle_pairs(
    matrix: np.ndarray,
    level_1_ids: np.ndarray,
    level_2_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract upper-triangle distances and level labels for unordered pairs."""
    n = matrix.shape[0]
    i_idx, j_idx = np.triu_indices(n, k=1)
    distances = matrix[i_idx, j_idx]
    l1_i = level_1_ids[i_idx]
    l1_j = level_1_ids[j_idx]
    l2_i = level_2_ids[i_idx]
    l2_j = level_2_ids[j_idx]
    return distances, (l1_i, l1_j), (l2_i, l2_j)


def summarize_distance_by_structure(
    distance_matrix: Union[RelationalDistanceMatrix, np.ndarray],
    level_1_ids: Optional[np.ndarray] = None,
    level_2_ids: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Compare mean distances for same-level-1, same-level-2, and baseline pairs.

    Parameters
    ----------
    distance_matrix : RelationalDistanceMatrix or ndarray
        If ndarray, pass ``level_1_ids`` and ``level_2_ids`` explicitly.
    level_1_ids, level_2_ids : ndarray, optional
        Required when ``distance_matrix`` is a bare ndarray.

    Returns
    -------
    pandas.DataFrame
        Rows: comparison types; columns: mean_distance, std_distance, n_pairs.
    """
    if isinstance(distance_matrix, RelationalDistanceMatrix):
        matrix = distance_matrix.matrix
        level_1_ids = distance_matrix.level_1_ids
        level_2_ids = distance_matrix.level_2_ids
    else:
        matrix = np.asarray(distance_matrix, dtype=float)
        if level_1_ids is None or level_2_ids is None:
            raise ValueError(
                "level_1_ids and level_2_ids are required when passing a raw matrix."
            )

    distances, (l1_i, l1_j), (l2_i, l2_j) = _upper_triangle_pairs(
        matrix, level_1_ids, level_2_ids
    )

    same_l1 = l1_i == l1_j
    same_l2 = l2_i == l2_j
    diff_l1 = ~same_l1
    diff_l2 = ~same_l2

    buckets = {
        "same_level_1_diff_level_2": same_l1 & diff_l2,
        "diff_level_1_same_level_2": diff_l1 & same_l2,
        "diff_level_1_diff_level_2": diff_l1 & diff_l2,
        "same_level_1_same_level_2": same_l1 & same_l2,
    }

    labels = {
        "same_level_1_diff_level_2": "Same level-1, different level-2",
        "diff_level_1_same_level_2": "Different level-1, same level-2",
        "diff_level_1_diff_level_2": "Different level-1, different level-2 (baseline)",
        "same_level_1_same_level_2": "Same level-1 and level-2 (duplicate pair)",
    }

    rows = []
    for key, mask in buckets.items():
        if mask.sum() == 0:
            rows.append(
                {
                    "comparison_type": labels[key],
                    "mean_distance": np.nan,
                    "std_distance": np.nan,
                    "n_pairs": 0,
                }
            )
        else:
            d = distances[mask]
            rows.append(
                {
                    "comparison_type": labels[key],
                    "mean_distance": float(np.mean(d)),
                    "std_distance": float(np.std(d, ddof=1)) if len(d) > 1 else 0.0,
                    "n_pairs": int(mask.sum()),
                }
            )

    return pd.DataFrame(rows)


def sequence_discrepancy_by_level(
    distance_matrix: Union[RelationalDistanceMatrix, np.ndarray],
    group_labels: np.ndarray,
    *,
    grouping_variable: str = "group",
    weights: Optional[np.ndarray] = None,
    R: int = 0,
    squared: bool = False,
) -> LevelDiscrepancyResult:
    """
    Pseudo-R² discrepancy for one grouping variable (level-1 or level-2).

    Wraps :func:`~sequenzo.discrepancy_analysis.single_factor_association`.
    """
    if isinstance(distance_matrix, RelationalDistanceMatrix):
        matrix = distance_matrix.matrix
    else:
        matrix = np.asarray(distance_matrix, dtype=float)

    group_labels = np.asarray(group_labels)
    assoc = single_factor_association(
        distance_matrix=matrix,
        group=group_labels,
        weights=weights,
        R=R,
        squared=squared,
    )

    anova = assoc["anova_table"]
    sc_tot = float(anova.loc["Total", "SS"])
    sc_res = float(anova.loc["Res", "SS"])
    sc_exp = float(anova.loc["Exp", "SS"])

    n_groups = int((assoc["groups"]["n"] > 0).sum() - 1)

    return LevelDiscrepancyResult(
        grouping_variable=grouping_variable,
        total_discrepancy=sc_tot,
        within_group_discrepancy=sc_res,
        between_group_discrepancy=sc_exp,
        pseudo_r2=float(assoc["pseudo_r2"]),
        pseudo_f=float(assoc["pseudo_f"]) if not np.isnan(assoc["pseudo_f"]) else np.nan,
        p_value=float(assoc["pseudo_f_pval"]) if R > 0 else np.nan,
        n_groups=n_groups,
        details=assoc,
    )


def hierarchical_sequence_discrepancy(
    distance_matrix: RelationalDistanceMatrix,
    *,
    level_1_name: str = "level_1",
    level_2_name: str = "level_2",
    weights: Optional[np.ndarray] = None,
    R: int = 0,
    squared: bool = False,
    include_multifactor: bool = False,
    include_additive: bool = True,
    include_crossed: bool = False,
) -> HierarchicalDecompositionResult:
    """
    Decompose sequence dissimilarity across two hierarchical levels.

    Main results (default):
    - Marginal pseudo-R² for level-1 and level-2
    - Additive joint model (level_1 + level_2) and residual share
    - Structural distance summary (same-level vs baseline means)

    Set ``include_crossed=True`` only for experimental full interaction
    decomposition (not recommended when each cell has one sequence).
    """
    level_1_result = sequence_discrepancy_by_level(
        distance_matrix,
        distance_matrix.level_1_ids,
        grouping_variable=level_1_name,
        weights=weights,
        R=R,
        squared=squared,
    )
    level_2_result = sequence_discrepancy_by_level(
        distance_matrix,
        distance_matrix.level_2_ids,
        grouping_variable=level_2_name,
        weights=weights,
        R=R,
        squared=squared,
    )

    structural = summarize_distance_by_structure(distance_matrix)

    joint_r2 = np.nan
    mf_table = None
    if include_multifactor:
        factors = pd.DataFrame(
            {
                level_1_name: distance_matrix.level_1_ids,
                level_2_name: distance_matrix.level_2_ids,
            }
        )
        design, term_ids, term_labels = build_multifactor_design(factors)
        mf = distance_multifactor_anova(
            distance_matrix.matrix,
            design,
            term_ids,
            term_labels=term_labels[1:],
            weights=weights,
            squared=squared,
            R=0,
        )
        mf_table = mf.get("summary")
        if mf_table is not None and len(mf_table):
            term_rows = mf_table[mf_table["Variable"] != "Total"]
            joint_r2 = float(term_rows["PseudoR2"].sum())

    additive = None
    if include_additive:
        additive = additive_sequence_discrepancy(
            distance_matrix,
            level_1_name=level_1_name,
            level_2_name=level_2_name,
            weights=weights,
            squared=squared,
        )

    crossed = None
    if include_crossed:
        crossed = crossed_sequence_discrepancy(
            distance_matrix,
            level_1_name=level_1_name,
            level_2_name=level_2_name,
            weights=weights,
            squared=squared,
            R=R,
        )

    r1 = level_1_result.pseudo_r2
    r2 = level_2_result.pseudo_r2
    if additive is not None:
        joint_r2 = additive.joint_share
        residual_share = additive.residual_share
    elif crossed is not None:
        joint_r2 = (
            crossed.level_1_share + crossed.level_2_share + crossed.interaction_share
        )
        residual_share = crossed.residual_share
    elif np.isfinite(joint_r2):
        residual_share = max(0.0, 1.0 - joint_r2)
    else:
        residual_share = max(0.0, 1.0 - max(r1, r2))

    return HierarchicalDecompositionResult(
        level_1=level_1_result,
        level_2=level_2_result,
        joint_pseudo_r2=joint_r2,
        residual_share=residual_share,
        structural_summary=structural,
        multifactor_table=mf_table,
        additive=additive,
        crossed=crossed,
    )


def decompose_sequence_dissimilarity(
    distance_matrix: RelationalDistanceMatrix,
    level_1_ids: Optional[np.ndarray] = None,
    level_2_ids: Optional[np.ndarray] = None,
    method: str = "pseudo_r2",
    **kwargs: Any,
) -> Dict[str, float]:
    """
    User-facing alias returning a flat dictionary of component shares.

    See :func:`hierarchical_sequence_discrepancy` for the full result object.
    """
    if level_1_ids is not None or level_2_ids is not None:
        pass  # ids taken from RelationalDistanceMatrix; kept for API compatibility

    result = hierarchical_sequence_discrepancy(distance_matrix, **kwargs)
    total = result.level_1.total_discrepancy

    if result.additive is not None:
        a = result.additive
        out = {
            "total_discrepancy": total,
            "level_1_component": a.level_1_ss,
            "level_2_component": a.level_2_ss,
            "interaction_component": np.nan,
            "residual_component": a.residual_ss,
            "level_1_share": a.level_1_share,
            "level_2_share": a.level_2_share,
            "interaction_share": np.nan,
            "residual_share": a.residual_share,
            "joint_pseudo_r2": a.joint_share,
        }
        if result.crossed is not None:
            c = result.crossed
            out["interaction_component"] = c.interaction_ss
            out["interaction_share"] = c.interaction_share
        return out

    if result.crossed is not None:
        c = result.crossed
        return {
            "total_discrepancy": total,
            "level_1_component": c.level_1_ss,
            "level_2_component": c.level_2_ss,
            "interaction_component": c.interaction_ss,
            "residual_component": c.residual_ss,
            "level_1_share": c.level_1_share,
            "level_2_share": c.level_2_share,
            "interaction_share": c.interaction_share,
            "residual_share": c.residual_share,
            "joint_pseudo_r2": (
                c.level_1_share + c.level_2_share + c.interaction_share
            ),
        }

    return {
        "total_discrepancy": total,
        "level_1_component": result.level_1.between_group_discrepancy,
        "level_2_component": result.level_2.between_group_discrepancy,
        "interaction_component": np.nan,
        "residual_component": result.level_1.within_group_discrepancy,
        "level_1_share": result.level_1.pseudo_r2,
        "level_2_share": result.level_2.pseudo_r2,
        "interaction_share": np.nan,
        "residual_share": result.residual_share,
        "joint_pseudo_r2": result.joint_pseudo_r2,
    }


def permutation_test_level_effect(
    distance_matrix: Union[RelationalDistanceMatrix, np.ndarray],
    group_labels: np.ndarray,
    *,
    n_perm: int = 999,
    random_state: Optional[int] = None,
    squared: bool = False,
) -> Dict[str, Any]:
    """
    Permutation test for a single grouping level's pseudo-R².

    Parameters
    ----------
    distance_matrix : RelationalDistanceMatrix or ndarray
    group_labels : array-like
        Level-1 or level-2 identifiers.
    n_perm : int
        Number of permutations (passed as ``R`` to discrepancy analysis).
    random_state : int, optional
        Seed for the permutation test. The global NumPy RNG state is saved and
        restored so other code is not affected.
    squared : bool
        Use squared dissimilarities in the statistic.

    Returns
    -------
    dict
        observed_pseudo_r2, mean_permuted_pseudo_r2, p_value, n_perm, and details.
    """
    rng_state = np.random.get_state() if random_state is not None else None
    try:
        if random_state is not None:
            np.random.seed(random_state)

        if isinstance(distance_matrix, RelationalDistanceMatrix):
            matrix = distance_matrix.matrix
        else:
            matrix = np.asarray(distance_matrix, dtype=float)

        assoc = single_factor_association(
            distance_matrix=matrix,
            group=np.asarray(group_labels),
            R=n_perm,
            squared=squared,
        )
    finally:
        if rng_state is not None:
            np.random.set_state(rng_state)

    return {
        "observed_pseudo_r2": float(assoc["pseudo_r2"]),
        "mean_permuted_pseudo_r2": np.nan,
        "p_value": float(assoc["pseudo_f_pval"]),
        "n_perm": n_perm,
        "details": assoc,
    }
