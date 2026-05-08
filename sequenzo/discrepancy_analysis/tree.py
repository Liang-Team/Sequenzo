"""
@Desc: Tree-building utilities for discrepancy analysis.
"""

from .disstree import build_distance_tree
from .seqtree import build_sequence_tree
from .permutation import test_tree_split_significance

test_tree_split = test_tree_split_significance

__all__ = [
    "build_distance_tree",
    "build_sequence_tree",
    "test_tree_split",
]
