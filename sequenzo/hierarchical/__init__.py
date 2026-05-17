"""
@Author  : 梁彧祺 Yuqi Liang
@File    : __init__.py
@Time    : 12/05/2026 10:16
@Desc    :
    Hierarchical sequence analysis for relational trajectories.

    Two scalable lines: (1) decomposition of pair dissimilarity by level-1, level-2,
    and pair-specific residuals; (2) pair-level trajectory typology via PAM or CLARA.
    See ``sequenzo.hierarchical.decomposition`` and ``sequenzo.hierarchical.clustering``.
"""

from .data import (
    RelationalSequenceData,
    RelationalSequenceRecord,
    validate_relational_sequence_data,
    make_relational_sequences,
    check_balanced_panel,
    make_pair_id,
    DEFAULT_PAIR_SEPARATOR,
    DEFAULT_MAX_FULL_MATRIX_PAIRS,
)
from .representation import (
    state_sequence_to_spells,
    to_spell_sequences,
    encode_states,
)
from .distances import (
    RelationalDistanceMatrix,
    compute_relational_distance_matrix,
    relational_sequences_to_sequence_data,
)
from .decomposition import (
    AdditiveDecompositionResult,
    CrossedDecompositionResult,
    HierarchicalDecompositionResult,
    LevelDiscrepancyResult,
    SampledPairwiseDistances,
    StructuralDistanceSummary,
    additive_sequence_discrepancy,
    build_additive_hierarchical_design,
    build_crossed_hierarchical_design,
    check_interaction_identifiability,
    crossed_sequence_discrepancy,
    decompose_sequence_dissimilarity,
    describe_sampling_scheme,
    sampling_scheme_description,
    hierarchical_sequence_discrepancy,
    hierarchical_sequence_discrepancy_from_sample,
    sample_structural_pairwise_distances,
    permutation_test_crossed_effect,
    permutation_test_level_effect,
    permutation_test_level_effect_sampled,
    sample_pairwise_distances,
    sequence_discrepancy_by_level,
    sequence_discrepancy_by_level_sampled,
    summarize_distance_by_structure,
    summarize_distance_by_structure_sampled,
)
from .residuals import compute_pair_residuals, detect_pair_specific_outliers
from .profiles import (
    summarize_level_1_profiles,
    summarize_level_2_profiles,
)
from .clustering import (
    HierarchicalClusterResult,
    PairTypologyResult,
    aggregate_distance_matrix_by_level,
    cluster_level_1_profiles,
    cluster_level_2_profiles,
    cluster_pair_sequences,
    cluster_pair_trajectories,
    cluster_pair_typology_clara,
)
from .visualization import (
    plot_decomposition_bar,
    plot_marginal_pseudo_r2,
    plot_additive_marginal_shares,
    plot_joint_residual_shares,
    plot_additive_component_shares,
    plot_crossed_component_shares,
    plot_distance_heatmap,
    plot_hierarchical_distance_heatmap,
    plot_relational_sequence_grid,
    plot_level_portfolio_sequences,
    plot_level_1_sequence_panels,
    plot_level_2_sequence_panels,
    plot_pair_outlier_sequences,
    plot_sequence_index_by_level,
    plot_level_similarity_matrix,
    plot_pair_outliers,
)
from .compression import (
    CompressedRelationalSequences,
    compress_identical_relational_sequences,
)
from .results import (
    AnalysisMode,
    HierarchicalSequenceResult,
    run_hierarchical_sequence_analysis,
)

try:
    from .simulation import (
        RelationalSimulationConfig,
        SimulatedRelationalData,
        SCENARIO_NAMES,
        generate_relational_trajectories,
        run_simulation_experiment,
        evaluate_hierarchical_recovery,
        summarize_simulation_results,
        plot_simulation_recovery_layers,
        plot_simulation_robustness_layers,
        plot_single_simulation_layers,
        plot_pair_residual_diagnostics,
        plot_pooled_vs_hierarchical_comparison,
        plot_scalability_benchmark,
        plot_hierarchical_framework_diagram,
        plot_empirical_hierarchical_layers,
        compare_pooled_vs_hierarchical,
        HierarchicalSimulationSpec,
        outlier_pair_ids,
        simulate_hierarchical_relational_data,
    )
except ImportError:  # pragma: no cover - optional subpackage
    RelationalSimulationConfig = None  # type: ignore[misc, assignment]
    SimulatedRelationalData = None  # type: ignore[misc, assignment]
    SCENARIO_NAMES = ()  # type: ignore[misc, assignment]
    generate_relational_trajectories = None  # type: ignore[misc, assignment]
    run_simulation_experiment = None  # type: ignore[misc, assignment]
    evaluate_hierarchical_recovery = None  # type: ignore[misc, assignment]
    summarize_simulation_results = None  # type: ignore[misc, assignment]
    plot_simulation_recovery_layers = None  # type: ignore[misc, assignment]
    plot_simulation_robustness_layers = None  # type: ignore[misc, assignment]
    plot_single_simulation_layers = None  # type: ignore[misc, assignment]
    plot_pair_residual_diagnostics = None  # type: ignore[misc, assignment]
    plot_pooled_vs_hierarchical_comparison = None  # type: ignore[misc, assignment]
    plot_scalability_benchmark = None  # type: ignore[misc, assignment]
    plot_hierarchical_framework_diagram = None  # type: ignore[misc, assignment]
    plot_empirical_hierarchical_layers = None  # type: ignore[misc, assignment]
    compare_pooled_vs_hierarchical = None  # type: ignore[misc, assignment]
    HierarchicalSimulationSpec = None  # type: ignore[misc, assignment]
    outlier_pair_ids = None  # type: ignore[misc, assignment]
    simulate_hierarchical_relational_data = None  # type: ignore[misc, assignment]

__all__ = [
    "RelationalSequenceData",
    "RelationalSequenceRecord",
    "validate_relational_sequence_data",
    "make_relational_sequences",
    "check_balanced_panel",
    "make_pair_id",
    "DEFAULT_PAIR_SEPARATOR",
    "DEFAULT_MAX_FULL_MATRIX_PAIRS",
    "state_sequence_to_spells",
    "to_spell_sequences",
    "encode_states",
    "RelationalDistanceMatrix",
    "compute_relational_distance_matrix",
    "relational_sequences_to_sequence_data",
    "StructuralDistanceSummary",
    "LevelDiscrepancyResult",
    "HierarchicalDecompositionResult",
    "summarize_distance_by_structure",
    "sequence_discrepancy_by_level",
    "hierarchical_sequence_discrepancy",
    "hierarchical_sequence_discrepancy_from_sample",
    "sample_structural_pairwise_distances",
    "CompressedRelationalSequences",
    "compress_identical_relational_sequences",
    "AnalysisMode",
    "decompose_sequence_dissimilarity",
    "permutation_test_level_effect",
    "AdditiveDecompositionResult",
    "CrossedDecompositionResult",
    "additive_sequence_discrepancy",
    "crossed_sequence_discrepancy",
    "build_crossed_hierarchical_design",
    "build_additive_hierarchical_design",
    "check_interaction_identifiability",
    "permutation_test_crossed_effect",
    "SampledPairwiseDistances",
    "describe_sampling_scheme",
    "sampling_scheme_description",
    "sample_pairwise_distances",
    "summarize_distance_by_structure_sampled",
    "sequence_discrepancy_by_level_sampled",
    "permutation_test_level_effect_sampled",
    "compute_pair_residuals",
    "detect_pair_specific_outliers",
    "summarize_level_1_profiles",
    "summarize_level_2_profiles",
    "HierarchicalClusterResult",
    "PairTypologyResult",
    "aggregate_distance_matrix_by_level",
    "cluster_pair_sequences",
    "cluster_pair_trajectories",
    "cluster_pair_typology_clara",
    "cluster_level_1_profiles",
    "cluster_level_2_profiles",
    "plot_decomposition_bar",
    "plot_marginal_pseudo_r2",
    "plot_additive_marginal_shares",
    "plot_joint_residual_shares",
    "plot_additive_component_shares",
    "plot_crossed_component_shares",
    "plot_distance_heatmap",
    "plot_hierarchical_distance_heatmap",
    "plot_relational_sequence_grid",
    "plot_level_portfolio_sequences",
    "plot_level_1_sequence_panels",
    "plot_level_2_sequence_panels",
    "plot_pair_outlier_sequences",
    "plot_sequence_index_by_level",
    "plot_level_similarity_matrix",
    "plot_pair_outliers",
    "HierarchicalSequenceResult",
    "run_hierarchical_sequence_analysis",
    "RelationalSimulationConfig",
    "SimulatedRelationalData",
    "SCENARIO_NAMES",
    "generate_relational_trajectories",
    "run_simulation_experiment",
    "evaluate_hierarchical_recovery",
    "summarize_simulation_results",
    "plot_simulation_recovery_layers",
    "plot_simulation_robustness_layers",
    "plot_single_simulation_layers",
    "plot_pair_residual_diagnostics",
    "plot_pooled_vs_hierarchical_comparison",
    "plot_scalability_benchmark",
    "plot_hierarchical_framework_diagram",
    "plot_empirical_hierarchical_layers",
    "compare_pooled_vs_hierarchical",
    "HierarchicalSimulationSpec",
    "simulate_hierarchical_relational_data",
    "outlier_pair_ids",
]
