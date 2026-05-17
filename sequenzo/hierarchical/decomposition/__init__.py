"""
@Author  : 梁彧祺 Yuqi Liang
@File    : __init__.py
@Time    : 16/04/2026 10:22
@Desc    :
    Decomposition and scalable discrepancy analysis for hierarchical sequences.

    Submodules: ``marginal``, ``crossed``, ``sampling``.
"""

from .crossed import (
    AdditiveDecompositionResult,
    CrossedDecompositionResult,
    additive_sequence_discrepancy,
    build_additive_hierarchical_design,
    build_crossed_hierarchical_design,
    check_interaction_identifiability,
    crossed_sequence_discrepancy,
    permutation_test_crossed_effect,
)
from .marginal import (
    HierarchicalDecompositionResult,
    LevelDiscrepancyResult,
    StructuralDistanceSummary,
    decompose_sequence_dissimilarity,
    hierarchical_sequence_discrepancy,
    permutation_test_level_effect,
    sequence_discrepancy_by_level,
    summarize_distance_by_structure,
)
from .sampling import (
    SampledPairwiseDistances,
    describe_sampling_scheme,
    hierarchical_sequence_discrepancy_from_sample,
    sample_pairwise_distances,
    sample_structural_pairwise_distances,
    sampling_scheme_description,
    sequence_discrepancy_by_level_sampled,
    summarize_distance_by_structure_sampled,
    permutation_test_level_effect_sampled,
)

__all__ = [
    "AdditiveDecompositionResult",
    "CrossedDecompositionResult",
    "HierarchicalDecompositionResult",
    "LevelDiscrepancyResult",
    "SampledPairwiseDistances",
    "StructuralDistanceSummary",
    "additive_sequence_discrepancy",
    "build_additive_hierarchical_design",
    "build_crossed_hierarchical_design",
    "check_interaction_identifiability",
    "crossed_sequence_discrepancy",
    "decompose_sequence_dissimilarity",
    "describe_sampling_scheme",
    "hierarchical_sequence_discrepancy",
    "hierarchical_sequence_discrepancy_from_sample",
    "sample_structural_pairwise_distances",
    "permutation_test_crossed_effect",
    "permutation_test_level_effect",
    "permutation_test_level_effect_sampled",
    "sample_pairwise_distances",
    "sampling_scheme_description",
    "sequence_discrepancy_by_level",
    "sequence_discrepancy_by_level_sampled",
    "summarize_distance_by_structure",
    "summarize_distance_by_structure_sampled",
]
