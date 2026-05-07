"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 2026-05-07 14:59
@Desc    : Unified group-comparison entrypoint for sequence analysis.

This module consolidates methods for comparing groups of sequences into one
clear namespace. It covers:

- Discrepancy / pseudo-ANOVA statistics (pseudo-F, pseudo-R2)
- Permutation-based significance testing
- Global two-group comparison metrics (LRT/BIC)
- Kitagawa-Oaxaca-Blinder decomposition for outcome gaps
- Tree-based subgroup discovery (distance/sequence trees)
"""

from ..tree_analysis.tree_utils import (
    compute_pseudo_variance,
    compute_distance_association,
    dissmfacw,
    dissmergegroups,
)
from ..tree_analysis.permutation import (
    permutation_test,
    test_tree_split_significance,
)
from ..tree_analysis.dissassoc_permutation import (
    dissassoc_permutation_test,
    distance_multifactor_anova,
    compute_distance_indicators,
)
from ..tree_analysis.disstree import build_distance_tree
from ..tree_analysis.seqtree import build_sequence_tree

from .seqdiff import (
    compare_groups_across_positions,
    plot_group_differences_across_positions,
    print_group_differences_across_positions,
)
from .seqcompare import (
    compare_groups_overall,
    compute_likelihood_ratio_test,
    compute_bayesian_information_criterion_test,
)
from .kob_decomposition import (
    oaxaca_blinder_decomposition,
    kob_decomposition,
    KOBDecompositionResult,
)

# Public, intuitive API names
get_discrepancy = compute_pseudo_variance
get_group_distance_association = compute_distance_association
get_multifactor_discrepancy_anova = distance_multifactor_anova
get_discrepancy_indicators = compute_distance_indicators

get_permutation_test = permutation_test
get_discrepancy_permutation_test = dissassoc_permutation_test
test_tree_split = test_tree_split_significance

get_group_differences_by_position = compare_groups_across_positions
plot_group_differences_by_position = plot_group_differences_across_positions
get_group_differences_report_by_position = print_group_differences_across_positions
get_group_differences_overall = compare_groups_overall
get_lrt_test = compute_likelihood_ratio_test
get_bic_test = compute_bayesian_information_criterion_test

get_kob_decomposition = kob_decomposition
get_oaxaca_blinder_decomposition = oaxaca_blinder_decomposition

__all__ = [
    # Discrepancy and pseudo-ANOVA core
    "get_discrepancy",
    "get_group_distance_association",
    "dissmfacw",
    "dissmergegroups",
    "get_multifactor_discrepancy_anova",
    "get_discrepancy_indicators",
    # Permutation tests
    "get_permutation_test",
    "get_discrepancy_permutation_test",
    "test_tree_split",
    # Group-comparison workflows
    "get_group_differences_by_position",
    "plot_group_differences_by_position",
    "get_group_differences_report_by_position",
    "get_group_differences_overall",
    "get_lrt_test",
    "get_bic_test",
    # KOB decomposition
    "get_oaxaca_blinder_decomposition",
    "get_kob_decomposition",
    "KOBDecompositionResult",
    # Tree-based subgroup discovery
    "build_distance_tree",
    "build_sequence_tree",
]
