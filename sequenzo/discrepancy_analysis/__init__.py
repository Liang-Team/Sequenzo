"""
@Desc: Discrepancy analysis API for sequence group decomposition and trees.
"""

from .core import (
    get_discrepancy,
    get_group_distance_association,
    get_multifactor_discrepancy_anova,
    get_discrepancy_indicators,
    dissmfacw,
    dissmergegroups,
)
from .permutation import permutation_test as get_permutation_test
from .dissassoc_permutation import dissassoc_permutation_test as get_discrepancy_permutation_test
from .tree import (
    build_distance_tree,
    build_sequence_tree,
    test_tree_split,
)
from .positionwise import (
    get_group_differences_by_position,
    plot_group_differences_by_position,
    get_group_differences_report_by_position,
)
from .tree_helpers import get_leaf_membership, get_classification_rules, assign_to_leaves
from .tree_visualization import plot_tree, print_tree, export_tree_to_dot
from .tree_node import DissTreeNode, DissTreeSplit

__all__ = [
    "get_discrepancy",
    "get_group_distance_association",
    "get_multifactor_discrepancy_anova",
    "get_discrepancy_indicators",
    "get_permutation_test",
    "get_discrepancy_permutation_test",
    "build_distance_tree",
    "build_sequence_tree",
    "test_tree_split",
    "get_group_differences_by_position",
    "plot_group_differences_by_position",
    "get_group_differences_report_by_position",
    "dissmfacw",
    "dissmergegroups",
    "get_leaf_membership",
    "get_classification_rules",
    "assign_to_leaves",
    "plot_tree",
    "print_tree",
    "export_tree_to_dot",
    "DissTreeNode",
    "DissTreeSplit",
]
