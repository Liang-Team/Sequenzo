"""
@Author  : 梁彧祺 Yuqi Liang
@File    : crossed.py
@Time    : 14/04/2026 09:25
@Desc    :
    Crossed distance-based ANOVA: Dissimilarity ~ level_1 + level_2 + level_1:level_2.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sequenzo.discrepancy_analysis.stats.multifactor_association import (
    _weighted_hat_matrix_qr,
    distance_multifactor_anova,
    gower_matrix,
)

from ..distances import RelationalDistanceMatrix


@dataclass
class AdditiveDecompositionResult:
    """
    Additive decomposition: dissimilarity ~ level_1 + level_2 (+ residual).

    Recommended default for relational data with one sequence per (level_1, level_2) cell.

    Notes on interpretation (Type-III SS)
    ------------------------------------
    ``level_1_share`` and ``level_2_share`` are partial / marginal contributions;
    they are **not** mutually exclusive components and need not sum to 1 with
    ``residual_share``. The pair ``joint_share`` and ``residual_share`` **do** sum
    to 1 (joint explained vs pair-specific residual).
    """

    total_discrepancy: float
    level_1_share: float
    level_2_share: float
    joint_share: float
    residual_share: float
    level_1_ss: float
    level_2_ss: float
    joint_ss: float
    residual_ss: float
    summary_table: pd.DataFrame
    level_1_name: str = "level_1"
    level_2_name: str = "level_2"
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossedDecompositionResult:
    """
    Experimental crossed decomposition with interaction (advanced).

    Model: dissimilarity ~ level_1 + level_2 + level_1 × level_2 (+ residual).

    Not recommended when each (level_1, level_2) cell has only one sequence:
    the interaction is saturated and confounded with pair identity.
    """

    total_discrepancy: float
    level_1_share: float
    level_2_share: float
    interaction_share: float
    residual_share: float
    level_1_ss: float
    level_2_ss: float
    interaction_ss: float
    residual_ss: float
    summary_table: pd.DataFrame
    level_1_name: str = "level_1"
    level_2_name: str = "level_2"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        return {
            "total_discrepancy": self.total_discrepancy,
            "level_1_share": self.level_1_share,
            "level_2_share": self.level_2_share,
            "interaction_share": self.interaction_share,
            "residual_share": self.residual_share,
            "level_1_component": self.level_1_ss,
            "level_2_component": self.level_2_ss,
            "interaction_component": self.interaction_ss,
            "residual_component": self.residual_ss,
        }


def build_crossed_hierarchical_design(
    level_1_ids: Sequence[Any],
    level_2_ids: Sequence[Any],
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Design matrix for ``level_1 * level_2`` (main effects + interaction).

    Column blocks are tagged with term ids: 0 = intercept, 1 = level_1,
    2 = level_2, 3 = interaction (products of main-effect dummies).
    """
    l1 = pd.Categorical(level_1_ids)
    l2 = pd.Categorical(level_2_ids)
    n = len(l1)

    x1 = pd.get_dummies(l1, drop_first=True, dtype=float)
    x2 = pd.get_dummies(l2, drop_first=True, dtype=float)

    columns = [np.ones(n, dtype=float)]
    term_ids = [0]

    for col in x1.columns:
        columns.append(x1[col].to_numpy())
        term_ids.append(1)

    for col in x2.columns:
        columns.append(x2[col].to_numpy())
        term_ids.append(2)

    if len(x1.columns) and len(x2.columns):
        for c1 in x1.columns:
            for c2 in x2.columns:
                columns.append((x1[c1] * x2[c2]).to_numpy())
                term_ids.append(3)

    design = np.column_stack(columns)
    term_labels = ["level_1", "level_2", "interaction"]
    return design, np.asarray(term_ids, dtype=int), term_labels


def build_additive_hierarchical_design(
    level_1_ids: Sequence[Any],
    level_2_ids: Sequence[Any],
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """Design matrix for additive model level_1 + level_2 (no interaction)."""
    design, term_ids, term_labels = build_crossed_hierarchical_design(
        level_1_ids, level_2_ids
    )
    keep = term_ids <= 2
    return design[:, keep], term_ids[keep], term_labels[:2]


def check_interaction_identifiability(
    level_1_ids: Sequence[Any],
    level_2_ids: Sequence[Any],
) -> Dict[str, Any]:
    """
    Check whether a full interaction model is saturated (one sequence per cell).
    """
    cells = pd.DataFrame({"l1": level_1_ids, "l2": level_2_ids})
    cell_counts = cells.groupby(["l1", "l2"], observed=True).size()
    saturated = bool((cell_counts <= 1).all())
    return {
        "saturated": saturated,
        "n_cells": int(len(cell_counts)),
        "max_cell_count": int(cell_counts.max()) if len(cell_counts) else 0,
        "cell_counts": cell_counts,
    }


def _warn_if_interaction_saturated(level_1_ids, level_2_ids) -> None:
    info = check_interaction_identifiability(level_1_ids, level_2_ids)
    if info["saturated"]:
        warnings.warn(
            "Each level_1–level_2 cell has only one sequence. A full interaction model "
            "is saturated, so interaction and residual variation cannot be separately "
            "identified. Use additive decomposition (additive_sequence_discrepancy) "
            "and pair-specific residual diagnostics instead.",
            UserWarning,
            stacklevel=3,
        )


def _explained_ss(
    design: np.ndarray,
    g_matrix: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Explained sum of squares from a Gower matrix and design matrix."""
    if design.shape[1] == 0:
        return 0.0
    hat = _weighted_hat_matrix_qr(design, weights=weights)
    return float((hat * g_matrix).sum())


def _type3_crossed_ss(
    design: np.ndarray,
    term_ids: np.ndarray,
    g_matrix: np.ndarray,
    weights: np.ndarray,
) -> Dict[str, float]:
    """
    Type-III marginal SS for level_1, level_2, and interaction.

    Uses nested model comparisons so saturated small samples remain stable.
    """
    intercept = design[:, term_ids == 0]
    l1_block = design[:, term_ids == 1]
    l2_block = design[:, term_ids == 2]
    int_block = design[:, term_ids == 3]

    def _stack(*parts):
        arrays = [p for p in parts if p.size > 0]
        if not arrays:
            return np.zeros((design.shape[0], 0))
        return np.column_stack(arrays) if len(arrays) > 1 else arrays[0]

    m0 = intercept
    m1 = _stack(intercept, l1_block)
    m2 = _stack(intercept, l2_block)
    m12 = _stack(intercept, l1_block, l2_block)
    mfull = design

    sc_tot = float((weights * np.diag(g_matrix)).sum())
    ss0 = _explained_ss(m0, g_matrix, weights)
    ss1 = _explained_ss(m1, g_matrix, weights)
    ss2 = _explained_ss(m2, g_matrix, weights)
    ss12 = _explained_ss(m12, g_matrix, weights)
    ss_full = _explained_ss(mfull, g_matrix, weights)

    ss_l1 = max(0.0, ss12 - ss2)
    ss_l2 = max(0.0, ss12 - ss1)
    ss_int = max(0.0, ss_full - ss12)
    ss_res = max(0.0, sc_tot - ss_full)

    return {
        "total": sc_tot,
        "level_1": ss_l1,
        "level_2": ss_l2,
        "interaction": ss_int,
        "residual": ss_res,
        "ss0": ss0,
        "ss_full": ss_full,
    }


def _type3_additive_ss(
    design: np.ndarray,
    term_ids: np.ndarray,
    g_matrix: np.ndarray,
    weights: np.ndarray,
) -> Dict[str, float]:
    """Type-III SS for additive model (level_1 + level_2 only)."""
    intercept = design[:, term_ids == 0]
    l1_block = design[:, term_ids == 1]
    l2_block = design[:, term_ids == 2]

    def _stack(*parts):
        arrays = [p for p in parts if p.size > 0]
        if not arrays:
            return np.zeros((design.shape[0], 0))
        return np.column_stack(arrays) if len(arrays) > 1 else arrays[0]

    m0 = intercept
    m1 = _stack(intercept, l1_block)
    m2 = _stack(intercept, l2_block)
    m12 = design

    sc_tot = float((weights * np.diag(g_matrix)).sum())
    ss1 = _explained_ss(m1, g_matrix, weights)
    ss2 = _explained_ss(m2, g_matrix, weights)
    ss12 = _explained_ss(m12, g_matrix, weights)

    ss_l1 = max(0.0, ss12 - ss2)
    ss_l2 = max(0.0, ss12 - ss1)
    ss_res = max(0.0, sc_tot - ss12)

    return {
        "total": sc_tot,
        "level_1": ss_l1,
        "level_2": ss_l2,
        "joint": ss12,
        "residual": ss_res,
    }


def additive_sequence_discrepancy(
    distance_matrix: RelationalDistanceMatrix,
    *,
    level_1_name: str = "level_1",
    level_2_name: str = "level_2",
    weights: Optional[np.ndarray] = None,
    squared: bool = False,
) -> AdditiveDecompositionResult:
    """
    Additive joint decomposition (recommended default).

    Model: dissimilarity ~ level_1 + level_2. Residual captures pair-specific
    deviation beyond additive region and technology structure.
    """
    n = distance_matrix.n_pairs
    if weights is None:
        weights = np.ones(n, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    design, term_ids, term_labels = build_additive_hierarchical_design(
        distance_matrix.level_1_ids,
        distance_matrix.level_2_ids,
    )
    g = gower_matrix(distance_matrix.matrix, squared=squared, weights=weights)
    ss = _type3_additive_ss(design, term_ids, g, weights)
    sc_tot = ss["total"]

    if sc_tot <= 0:
        shares = {k: 0.0 for k in ("level_1", "level_2", "joint", "residual")}
    else:
        shares = {
            "level_1": ss["level_1"] / sc_tot,
            "level_2": ss["level_2"] / sc_tot,
            "joint": ss["joint"] / sc_tot,
            "residual": ss["residual"] / sc_tot,
        }

    summary = pd.DataFrame(
        {
            "Variable": term_labels + ["Joint", "Residual", "Total"],
            "SS": [
                ss["level_1"],
                ss["level_2"],
                ss["joint"],
                ss["residual"],
                sc_tot,
            ],
            "PseudoR2": [
                shares["level_1"],
                shares["level_2"],
                shares["joint"],
                shares["residual"],
                shares["joint"],
            ],
        }
    )

    return AdditiveDecompositionResult(
        total_discrepancy=sc_tot,
        level_1_share=shares["level_1"],
        level_2_share=shares["level_2"],
        joint_share=shares["joint"],
        residual_share=shares["residual"],
        level_1_ss=ss["level_1"],
        level_2_ss=ss["level_2"],
        joint_ss=ss["joint"],
        residual_ss=ss["residual"],
        summary_table=summary,
        level_1_name=level_1_name,
        level_2_name=level_2_name,
        details={"method": "type3_additive_gower"},
    )


def crossed_sequence_discrepancy(
    distance_matrix: RelationalDistanceMatrix,
    *,
    level_1_name: str = "level_1",
    level_2_name: str = "level_2",
    weights: Optional[np.ndarray] = None,
    squared: bool = False,
    R: int = 0,
    random_state: Optional[int] = None,
) -> CrossedDecompositionResult:
    """
    Experimental crossed decomposition including level_1 × level_2 interaction.

    See :func:`additive_sequence_discrepancy` for the recommended default.
    Emits a warning when the interaction model is saturated (one sequence per cell).
    """
    _warn_if_interaction_saturated(
        distance_matrix.level_1_ids,
        distance_matrix.level_2_ids,
    )
    n = distance_matrix.n_pairs
    if weights is None:
        weights = np.ones(n, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    design, term_ids, term_labels = build_crossed_hierarchical_design(
        distance_matrix.level_1_ids,
        distance_matrix.level_2_ids,
    )

    g = gower_matrix(distance_matrix.matrix, squared=squared, weights=weights)
    ss = _type3_crossed_ss(design, term_ids, g, weights)
    sc_tot = ss["total"]

    if sc_tot <= 0:
        shares = {k: 0.0 for k in ("level_1", "level_2", "interaction", "residual")}
    else:
        shares = {
            "level_1": ss["level_1"] / sc_tot,
            "level_2": ss["level_2"] / sc_tot,
            "interaction": ss["interaction"] / sc_tot,
            "residual": ss["residual"] / sc_tot,
        }

    summary = pd.DataFrame(
        {
            "Variable": term_labels + ["Residual", "Total"],
            "SS": [
                ss["level_1"],
                ss["level_2"],
                ss["interaction"],
                ss["residual"],
                sc_tot,
            ],
            "PseudoR2": [
                shares["level_1"],
                shares["level_2"],
                shares["interaction"],
                shares["residual"],
                shares["level_1"] + shares["level_2"] + shares["interaction"],
            ],
            "p_value": np.nan,
        }
    )

    mf_details: Dict[str, Any] = {"method": "type3_gower"}
    if R > 1 and n >= 5:
        try:
            mf = distance_multifactor_anova(
                distance_matrix.matrix,
                design,
                term_ids,
                term_labels=term_labels,
                weights=weights,
                squared=squared,
                R=R,
                random_state=random_state,
            )
            summary_perm = mf["summary"]
            for term in term_labels:
                row = summary_perm[summary_perm["Variable"] == term]
                if len(row):
                    summary.loc[summary["Variable"] == term, "p_value"] = float(
                        row["p_value"].iloc[0]
                    )
            mf_details["permutation"] = mf
        except (ZeroDivisionError, ValueError):
            pass

    return CrossedDecompositionResult(
        total_discrepancy=sc_tot,
        level_1_share=shares["level_1"],
        level_2_share=shares["level_2"],
        interaction_share=shares["interaction"],
        residual_share=shares["residual"],
        level_1_ss=ss["level_1"],
        level_2_ss=ss["level_2"],
        interaction_ss=ss["interaction"],
        residual_ss=ss["residual"],
        summary_table=summary,
        level_1_name=level_1_name,
        level_2_name=level_2_name,
        details=mf_details,
    )


def permutation_test_crossed_effect(
    distance_matrix: RelationalDistanceMatrix,
    *,
    n_perm: int = 999,
    random_state: Optional[int] = None,
    squared: bool = False,
    permute: str = "level_1",
) -> Dict[str, Any]:
    """
    Permutation test for crossed-model terms by shuffling one level's labels.

    Parameters
    ----------
    permute : str
        ``"level_1"`` or ``"level_2"``: which factor to permute.
    """
    if random_state is not None:
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random.default_rng()

    observed = crossed_sequence_discrepancy(
        distance_matrix, squared=squared, R=0
    )
    obs_r2 = {
        "level_1": observed.level_1_share,
        "level_2": observed.level_2_share,
        "interaction": observed.interaction_share,
    }

    n = distance_matrix.n_pairs
    weights = np.ones(n, dtype=float)
    g = gower_matrix(distance_matrix.matrix, squared=squared, weights=weights)

    perm_r2 = {k: [] for k in obs_r2}

    labels = (
        distance_matrix.level_1_ids.copy()
        if permute == "level_1"
        else distance_matrix.level_2_ids.copy()
    )
    other = (
        distance_matrix.level_2_ids
        if permute == "level_1"
        else distance_matrix.level_1_ids
    )

    for _ in range(n_perm):
        shuffled = labels.copy()
        rng.shuffle(shuffled)
        if permute == "level_1":
            design, term_ids, _ = build_crossed_hierarchical_design(shuffled, other)
        else:
            design, term_ids, _ = build_crossed_hierarchical_design(other, shuffled)

        ss = _type3_crossed_ss(design, term_ids, g, weights)
        sc_tot = ss["total"]
        if sc_tot > 0:
            perm_r2["level_1"].append(ss["level_1"] / sc_tot)
            perm_r2["level_2"].append(ss["level_2"] / sc_tot)
            perm_r2["interaction"].append(ss["interaction"] / sc_tot)

    p_values = {}
    for term, obs in obs_r2.items():
        arr = np.array(perm_r2[term])
        p_values[term] = float(np.mean(arr >= obs)) if len(arr) else np.nan

    return {
        "observed": obs_r2,
        "p_values": p_values,
        "n_perm": n_perm,
        "permute": permute,
    }
