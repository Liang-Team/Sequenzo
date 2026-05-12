"""
Discrepancy analysis for sequence distances and covariates.

Import from this package root only. Subpackages under stats/, trees/,
positionwise/, and internal/ are for maintainers.
"""

from .stats import (
    overall_discrepancy,
    single_factor_association,
    marginal_factor_association,
    merge_cluster_groups,
    distance_multifactor_anova,
    multifactor_association,
    individual_indicators,
)
from .internal import permutation_test, association_permutation_test
from .trees import (
    distance_tree,
    sequence_tree,
    test_tree_split,
    get_leaf_membership,
    get_classification_rules,
    assign_to_leaves,
    plot_tree,
    print_tree,
    export_tree_to_dot,
    DissTreeNode,
    DissTreeSplit,
)
from .positionwise import (
    compare_groups_across_positions,
    plot_group_differences_across_positions,
    print_group_differences_across_positions,
)

__all__ = [
    "overall_discrepancy",
    "single_factor_association",
    "marginal_factor_association",
    "multifactor_association",
    "distance_multifactor_anova",
    "individual_indicators",
    "merge_cluster_groups",
    "permutation_test",
    "association_permutation_test",
    "distance_tree",
    "sequence_tree",
    "test_tree_split",
    "compare_groups_across_positions",
    "plot_group_differences_across_positions",
    "print_group_differences_across_positions",
    "get_leaf_membership",
    "get_classification_rules",
    "assign_to_leaves",
    "plot_tree",
    "print_tree",
    "export_tree_to_dot",
    "DissTreeNode",
    "DissTreeSplit",
]
