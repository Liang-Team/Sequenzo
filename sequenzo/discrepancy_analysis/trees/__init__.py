"""Distance- and sequence-based discrepancy trees (TraMineR disstree / seqtree)."""

from .distance_tree import distance_tree
from .sequence_tree import sequence_tree
from .tree_node import DissTreeNode, DissTreeSplit
from .tree_leaf_helpers import get_leaf_membership, get_classification_rules, assign_to_leaves
from .tree_visualization import plot_tree, print_tree, export_tree_to_dot
from ..internal.permutation_engine import test_tree_split_significance

test_tree_split = test_tree_split_significance

__all__ = [
    "distance_tree",
    "sequence_tree",
    "test_tree_split",
    "DissTreeNode",
    "DissTreeSplit",
    "get_leaf_membership",
    "get_classification_rules",
    "assign_to_leaves",
    "plot_tree",
    "print_tree",
    "export_tree_to_dot",
]
