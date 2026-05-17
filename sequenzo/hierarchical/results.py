"""
@Author  : 梁彧祺 Yuqi Liang
@File    : results.py
@Time    : 11/05/2026 08:55
@Desc    :
    High-level pipeline: decomposition, typology, profiles, and outlier diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, TYPE_CHECKING

AnalysisMode = Literal["exact", "sampled", "typology_only"]

if TYPE_CHECKING:
    from .clustering import HierarchicalClusterResult, PairTypologyResult
    from .decomposition.sampling import SampledPairwiseDistances

import numpy as np
import pandas as pd

from .data import (
    RelationalSequenceData,
    validate_relational_sequence_data,
    make_relational_sequences,
)
from .decomposition import (
    CrossedDecompositionResult,
    HierarchicalDecompositionResult,
    crossed_sequence_discrepancy,
    decompose_sequence_dissimilarity,
    hierarchical_sequence_discrepancy,
    hierarchical_sequence_discrepancy_from_sample,
    permutation_test_crossed_effect,
    permutation_test_level_effect,
    sample_structural_pairwise_distances,
)
from .distances import (
    RelationalDistanceMatrix,
    compute_relational_distance_matrix,
)
from .clustering import (
    HierarchicalClusterResult,
    cluster_level_1_profiles,
    cluster_level_2_profiles,
    cluster_pair_sequences,
    cluster_pair_trajectories,
)
from .profiles import (
    detect_pair_specific_outliers,
    summarize_level_1_profiles,
    summarize_level_2_profiles,
)
from .residuals import compute_pair_residuals
from .visualization import (
    plot_decomposition_bar,
    plot_distance_heatmap,
    plot_hierarchical_distance_heatmap,
    plot_relational_sequence_grid,
    plot_level_1_sequence_panels,
    plot_level_2_sequence_panels,
    plot_pair_outlier_sequences,
    plot_sequence_index_by_level,
    plot_level_similarity_matrix,
    plot_pair_outliers,
)


@dataclass
class HierarchicalSequenceResult:
    """
    Container for a full hierarchical sequence analysis run.

    Attributes
    ----------
    sequences : RelationalSequenceData
    distance_matrix : RelationalDistanceMatrix
    decomposition : HierarchicalDecompositionResult
    level_1_profiles, level_2_profiles : DataFrame
    pair_outliers : DataFrame
    permutation_tests : dict
    validation_summary : dict
    config : dict
        Parameters used in the analysis (method, representation, etc.).
    """

    sequences: RelationalSequenceData
    level_1_profiles: pd.DataFrame
    level_2_profiles: pd.DataFrame
    pair_outliers: pd.DataFrame
    distance_matrix: Optional[RelationalDistanceMatrix] = None
    decomposition: Optional[HierarchicalDecompositionResult] = None
    sampled_distances: Optional["SampledPairwiseDistances"] = None
    pair_residuals: Optional[pd.DataFrame] = None
    pair_clusters: Optional["HierarchicalClusterResult"] = None
    pair_typology: Optional["PairTypologyResult"] = None
    level_1_clusters: Optional["HierarchicalClusterResult"] = None
    level_2_clusters: Optional["HierarchicalClusterResult"] = None
    permutation_tests: Dict[str, Any] = field(default_factory=dict)
    validation_summary: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def _require_distance_matrix(self, plot_name: str = "This plot") -> RelationalDistanceMatrix:
        if self.distance_matrix is None:
            raise ValueError(
                f"{plot_name} requires a full distance matrix. "
                "Run with analysis_mode='exact', or set compute_full_distance=True."
            )
        return self.distance_matrix

    def _require_decomposition(self, plot_name: str = "Decomposition plot") -> HierarchicalDecompositionResult:
        if self.decomposition is None:
            raise ValueError(
                f"{plot_name} requires a decomposition result. "
                "Run with analysis_mode='exact' or 'sampled', or set run_decomposition=True."
            )
        return self.decomposition

    def summary(self) -> str:
        """Return a beginner-friendly text summary."""
        v = self.validation_summary
        d = self.decomposition
        mode = self.config.get("analysis_mode", "exact")
        lines = [
            "Hierarchical Sequence Analysis Summary",
            "====================================",
            "",
            f"Analysis mode: {mode}",
            f"Number of level-1 units: {v.get('n_level_1', '—')}",
            f"Number of level-2 units: {v.get('n_level_2', '—')}",
            f"Number of pair-level sequences: {v.get('n_pairs', '—')}",
            f"Sequence length: {v.get('n_time_points', '—')} time points",
            f"Distance method: {self.config.get('distance_method', '—')}",
            f"Representation: {self.config.get('representation', '—')}",
            f"Full distance matrix stored: {self.distance_matrix is not None}",
        ]
        if mode == "sampled":
            lines.append(
                "Sampled mode: structural decomposition only "
                "(profiles and pair outliers require a full distance matrix)."
            )
        if self.pair_typology is not None:
            pt = self.pair_typology
            lines.extend(
                [
                    "",
                    "Pair-level typology",
                    "-------------------",
                    f"Algorithm: {pt.method}",
                    f"Number of trajectory types (k): {pt.k}",
                    f"CLARA sample size: {pt.details.get('sample_size', '—')}",
                    f"CLARA iterations: {pt.stability.get('n_iterations', '—')}",
                ]
            )
            if pt.quality.get("avg_dist") is not None:
                lines.append(f"Average distance to medoids: {pt.quality.get('avg_dist'):.4f}")
            if pt.stability.get("ari_above_0.8") == pt.stability.get("ari_above_0.8"):
                lines.append(
                    f"Stability (ARI ≥ 0.8 across iterations): "
                    f"{pt.stability.get('ari_above_0.8', '—')}"
                )
            if pt.representativeness is not None:
                rep = np.asarray(pt.representativeness)
                max_rep = rep.max(axis=1) if rep.ndim == 2 else rep
                lines.append(
                    f"Mean max representativeness: {float(np.nanmean(max_rep)):.3f}"
                )
        if d is None:
            lines.extend(
                [
                    "",
                    "Decomposition",
                    "-------------",
                    "Not computed in this run (typology-only or configure "
                    "analysis_mode='exact' / 'sampled').",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "Marginal pseudo-R² (non-exclusive)",
                    "--------------------------------",
                ]
            )
            if getattr(d, "method", None) == "structural_sample":
                lines.append(
                    "Note: contrast-based structural approximation "
                    "(not full-matrix Gower decomposition)."
                )
            lines.extend(
                [
                    f"Level-1 ({self.config.get('level_1_col', 'level_1')}): "
                    f"{100 * d.level_1.pseudo_r2:.1f}%",
                    f"Level-2 ({self.config.get('level_2_col', 'level_2')}): "
                    f"{100 * d.level_2.pseudo_r2:.1f}%",
                ]
            )
        if d is not None and d.additive is not None:
            a = d.additive
            lines.extend(
                [
                    "",
                    "Additive decomposition (level_1 + level_2)",
                    "------------------------------------------",
                    f"Level-1 share: {100 * a.level_1_share:.1f}%",
                    f"Level-2 share: {100 * a.level_2_share:.1f}%",
                    f"Joint explained: {100 * a.joint_share:.1f}%",
                    f"Pair-specific residual: {100 * a.residual_share:.1f}%",
                ]
            )
        elif d is not None and pd.notna(d.joint_pseudo_r2):
            lines.append(
                f"Joint level-1 + level-2: {100 * d.joint_pseudo_r2:.1f}%"
            )
        if d is not None and d.crossed is not None:
            c = d.crossed
            lines.extend(
                [
                    "",
                    "Experimental crossed model (includes interaction)",
                    "------------------------------------------------",
                    f"Interaction share: {100 * c.interaction_share:.1f}% "
                    "(may be saturated if one sequence per cell)",
                ]
            )
        if mode == "sampled" and self.distance_matrix is None:
            lines.extend(
                [
                    "",
                    "Pair outliers / profiles",
                    "-----------------------",
                    "Not computed: sampled mode has no full distance matrix.",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "Pair outliers",
                    "-------------",
                    "Based on additive distance residuals "
                    "(unusually similar / distant vs level structure).",
                ]
            )
        lines.extend(
            [
                "",
                "Permutation tests",
                "-----------------",
            ]
        )
        for key, test in self.permutation_tests.items():
            p = test.get("p_value", float("nan"))
            lines.append(f"{key}: p = {p:.4f}" if p == p else f"{key}: p = —")

        lines.extend(
            [
                "",
                "Interpretation",
                "--------------",
                "Marginal pseudo-R² values are not additive. The additive joint model",
                "and pair residuals describe region + technology structure and",
                "pair-specific deviations from that additive expectation.",
            ]
        )
        return "\n".join(lines)

    def plot_decomposition(self, **kwargs: Any):
        return plot_decomposition_bar(
            self._require_decomposition("Decomposition plot"), **kwargs
        )

    def plot_relational_grid(self, **kwargs: Any):
        """Relational sequence grid (level-1 × level-2 cells)."""
        return plot_relational_sequence_grid(
            self.sequences,
            pair_residuals=self.pair_residuals,
            **kwargs,
        )

    def plot_region_profiles(self, **kwargs: Any):
        """Level-1 relational portfolios across level-2 counterparts."""
        return plot_level_1_sequence_panels(
            self.sequences,
            pair_residuals=self.pair_residuals,
            **kwargs,
        )

    def plot_cpc_profiles(self, **kwargs: Any):
        """Level-2 relational portfolios across level-1 counterparts."""
        return plot_level_2_sequence_panels(
            self.sequences,
            pair_residuals=self.pair_residuals,
            **kwargs,
        )

    def plot_outliers(self, **kwargs: Any):
        return plot_pair_outliers(self.pair_outliers, **kwargs)

    def plot_outlier_sequences(self, **kwargs: Any):
        """Sequence barcodes for top pair-specific residual outliers."""
        residuals = self.pair_residuals if self.pair_residuals is not None else self.pair_outliers
        return plot_pair_outlier_sequences(
            self.sequences,
            residuals,
            distance_matrix=self.distance_matrix,
            **kwargs,
        )

    def plot_distance_heatmap(self, **kwargs: Any):
        return plot_distance_heatmap(
            self._require_distance_matrix("Distance heatmap"), **kwargs
        )

    def plot_hierarchical_distance_heatmap(self, **kwargs: Any):
        return plot_hierarchical_distance_heatmap(
            self._require_distance_matrix("Hierarchical distance heatmap"), **kwargs
        )

    def plot_level_1_similarity(self, **kwargs: Any):
        return plot_level_similarity_matrix(
            self._require_distance_matrix("Level-1 similarity matrix"),
            level=1,
            **kwargs,
        )

    def plot_level_2_similarity(self, **kwargs: Any):
        return plot_level_similarity_matrix(
            self._require_distance_matrix("Level-2 similarity matrix"),
            level=2,
            **kwargs,
        )


def run_hierarchical_sequence_analysis(
    data: pd.DataFrame,
    level_1_col: str,
    level_2_col: str,
    time_col: str,
    state_col: str,
    *,
    distance_method: str = "HAM",
    representation: str = "state",
    n_perm: int = 999,
    random_state: Optional[int] = 123,
    target_state: Any = None,
    require_balanced: bool = True,
    run_profiles: bool = True,
    run_outliers: bool = True,
    residual_method: str = "additive",
    include_crossed: bool = False,
    cluster_k: Optional[int] = None,
    cluster_level_1_k: Optional[int] = None,
    cluster_level_2_k: Optional[int] = None,
    typology_algorithm: str = "pam",
    typology_sample_size: Optional[int] = None,
    typology_n_iterations: int = 100,
    distance_params: Optional[Dict[str, Any]] = None,
    analysis_mode: AnalysisMode = "exact",
    compute_full_distance: Optional[bool] = None,
    run_decomposition: Optional[bool] = None,
    n_same_level_1: int = 200_000,
    n_same_level_2: int = 200_000,
    n_baseline_pairs: int = 400_000,
) -> HierarchicalSequenceResult:
    """
    End-to-end hierarchical sequence analysis pipeline.

    Parameters
    ----------
    data : pandas.DataFrame
        Long-format relational sequence data.
    level_1_col, level_2_col, time_col, state_col : str
        Column names.
    distance_method : str
        Sequenzo distance method name.
    representation : str
        ``"state"`` or ``"spell"``.
    n_perm : int
        Permutations for level-1 and level-2 tests (0 skips tests).
    random_state : int, optional
        RNG seed for permutation tests.
    target_state : optional
        State label for profile metrics (e.g. RCA state).
    require_balanced : bool
        Require equal sequence length across pairs.
    run_profiles, run_outliers : bool
        Whether to compute profile and outlier tables.
    residual_method : str
        ``"additive"`` (recommended), ``"simple"``, or ``"crossed_anova"`` (experimental).
    include_crossed : bool
        If True, also fit experimental level_1 × level_2 interaction decomposition.
    cluster_k, cluster_level_1_k, cluster_level_2_k : int, optional
        If set, run clustering at pair / level-1 / level-2 resolution.
    typology_algorithm : str
        For pair-level clustering: ``"pam"`` (full matrix) or ``"clara"`` (scalable).
    typology_sample_size, typology_n_iterations
        CLARA subsample size and number of iterations when ``typology_algorithm="clara"``.
    analysis_mode : str
        ``"exact"`` — full distance matrix + exact decomposition;
        ``"sampled"`` — stratified pairwise sampling for decomposition (no full matrix);
        ``"typology_only"`` — pair-level typology only (CLARA/PAM without full matrix).
    compute_full_distance, run_decomposition : bool, optional
        Override mode defaults. When ``compute_full_distance=False``, no ``n×n`` matrix
        is stored (required for scalable CLARA typology at large ``n_pairs``).
    n_same_level_1, n_same_level_2, n_baseline_pairs : int
        Quotas for ``analysis_mode="sampled"`` structural sampling.
    distance_params : dict, optional
        Extra kwargs for distance computation.

    Returns
    -------
    HierarchicalSequenceResult
    """
    distance_params = distance_params or {}

    validation = validate_relational_sequence_data(
        data,
        level_1_col,
        level_2_col,
        time_col,
        state_col,
        require_balanced=require_balanced,
    )

    sequences = make_relational_sequences(
        data,
        level_1_col=level_1_col,
        level_2_col=level_2_col,
        time_col=time_col,
        state_col=state_col,
        validate=False,
        require_balanced=require_balanced,
    )

    mode = analysis_mode.lower().strip()
    if mode not in {"exact", "sampled", "typology_only"}:
        raise ValueError(
            "analysis_mode must be 'exact', 'sampled', or 'typology_only'."
        )

    if compute_full_distance is None:
        need_full_matrix = mode == "exact"
    else:
        need_full_matrix = bool(compute_full_distance)

    if run_decomposition is None:
        do_decomposition = mode in {"exact", "sampled"}
    else:
        do_decomposition = bool(run_decomposition)

    if mode == "typology_only" and cluster_k is None:
        raise ValueError(
            "analysis_mode='typology_only' requires cluster_k for pair-level typology."
        )

    dist: Optional[RelationalDistanceMatrix] = None
    if need_full_matrix:
        dist = compute_relational_distance_matrix(
            sequences,
            method=distance_method,
            representation=representation,
            **distance_params,
        )

    sampled_distances = None
    decomposition: Optional[HierarchicalDecompositionResult] = None
    if do_decomposition:
        if mode == "sampled":
            sampled_distances = sample_structural_pairwise_distances(
                sequences,
                n_same_level_1=n_same_level_1,
                n_same_level_2=n_same_level_2,
                n_baseline=n_baseline_pairs,
                method=distance_method,
                random_state=random_state,
            )
            decomposition = hierarchical_sequence_discrepancy_from_sample(
                sampled_distances,
                level_1_name=level_1_col,
                level_2_name=level_2_col,
            )
        elif dist is not None:
            decomposition = hierarchical_sequence_discrepancy(
                dist,
                level_1_name=level_1_col,
                level_2_name=level_2_col,
                R=0,
                include_additive=True,
                include_crossed=include_crossed,
            )

    perm_tests: Dict[str, Any] = {}
    if dist is not None and n_perm > 0:
        perm_tests[level_1_col] = permutation_test_level_effect(
            dist,
            dist.level_1_ids,
            n_perm=n_perm,
            random_state=random_state,
        )
        perm_tests[level_2_col] = permutation_test_level_effect(
            dist,
            dist.level_2_ids,
            n_perm=n_perm,
            random_state=random_state,
        )

    if run_profiles and dist is not None:
        l1_profiles = summarize_level_1_profiles(
            sequences, dist, target_state=target_state
        )
        l2_profiles = summarize_level_2_profiles(
            sequences, dist, target_state=target_state
        )
    else:
        l1_profiles = pd.DataFrame()
        l2_profiles = pd.DataFrame()

    pair_residuals = None
    if run_outliers and dist is not None:
        pair_residuals = compute_pair_residuals(sequences, dist, method=residual_method)
        outliers = detect_pair_specific_outliers(
            sequences, dist, method=residual_method
        )
    else:
        outliers = pd.DataFrame()

    pair_clusters = None
    pair_typology = None
    level_1_clusters = None
    level_2_clusters = None
    if cluster_k is not None and cluster_k > 0:
        algo = typology_algorithm.lower().strip()
        if algo == "clara":
            pair_typology = cluster_pair_trajectories(
                sequences,
                cluster_k,
                algorithm="clara",
                distance_method=distance_method,
                representation=representation,
                sample_size=typology_sample_size,
                n_iterations=typology_n_iterations,
                distance_params=distance_params,
                random_state=random_state,
                verbose=False,
            )
        elif dist is not None:
            pair_typology = cluster_pair_trajectories(
                sequences,
                cluster_k,
                algorithm="pam",
                distance_matrix=dist,
                random_state=random_state,
                verbose=False,
            )
            pair_clusters = cluster_pair_sequences(
                dist, cluster_k, random_state=random_state, verbose=False
            )
        else:
            raise ValueError(
                "PAM typology requires a full distance matrix. Use typology_algorithm='clara' "
                "or analysis_mode='exact' with compute_full_distance=True."
            )
    if cluster_level_1_k is not None and cluster_level_1_k > 0:
        if dist is None:
            raise ValueError("Level-1 clustering requires a full distance matrix.")
        level_1_clusters = cluster_level_1_profiles(
            sequences, dist, cluster_level_1_k, random_state=random_state, verbose=False
        )
    if cluster_level_2_k is not None and cluster_level_2_k > 0:
        if dist is None:
            raise ValueError("Level-2 clustering requires a full distance matrix.")
        level_2_clusters = cluster_level_2_profiles(
            sequences, dist, cluster_level_2_k, random_state=random_state, verbose=False
        )

    config = {
        "level_1_col": level_1_col,
        "level_2_col": level_2_col,
        "time_col": time_col,
        "state_col": state_col,
        "distance_method": distance_method,
        "representation": representation,
        "analysis_mode": mode,
        "n_perm": n_perm,
        "random_state": random_state,
        "residual_method": residual_method,
        "include_crossed": include_crossed,
        "typology_algorithm": typology_algorithm,
    }

    return HierarchicalSequenceResult(
        sequences=sequences,
        distance_matrix=dist,
        decomposition=decomposition,
        sampled_distances=sampled_distances,
        level_1_profiles=l1_profiles,
        level_2_profiles=l2_profiles,
        pair_outliers=outliers,
        pair_residuals=pair_residuals,
        pair_clusters=pair_clusters,
        pair_typology=pair_typology,
        level_1_clusters=level_1_clusters,
        level_2_clusters=level_2_clusters,
        permutation_tests=perm_tests,
        validation_summary=validation,
        config=config,
    )
