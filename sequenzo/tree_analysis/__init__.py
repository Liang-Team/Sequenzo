"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 2026-02-09 7:56
@Desc    : Tree-structured analysis for sequence data based on distance matrices.
           This module provides regression tree analysis for sequences, allowing you to
           understand how covariates explain differences in sequence patterns.

           Corresponds to TraMineR functions: seqtree(), disstree(), dissvar(), dissassoc()
"""

from .tree_utils import compute_pseudo_variance, compute_distance_association
from .disstree import build_distance_tree
from .seqtree import build_sequence_tree
from .tree_helpers import (
    get_leaf_membership,
    get_classification_rules,
    assign_to_leaves
)
from .tree_visualization import (
    plot_tree,
    print_tree,
    export_tree_to_dot,
)
from .tree_node import DissTreeNode, DissTreeSplit

__all__ = [
    # Utility functions
    'compute_pseudo_variance',
    'compute_distance_association',
    # Tree building
    'build_distance_tree',
    'build_sequence_tree',
    # Helper functions
    'get_leaf_membership',
    'get_classification_rules',
    'assign_to_leaves',
    # Visualization functions
    'plot_tree',
    'print_tree',
    'export_tree_to_dot',
    # Data structures
    'DissTreeNode',
    'DissTreeSplit',
]
