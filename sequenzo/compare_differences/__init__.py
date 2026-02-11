"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 2026-02-11 08:32
@Desc    : Sequence difference analysis functions for comparing groups of sequences.
           This module provides tools to analyze how differences between groups of sequences
           evolve across positions and to perform likelihood ratio tests for
           comparing sets of sequences overall.

           Corresponds to TraMineR functions: seqdiff() and TraMineRextras functions: 
           seqCompare(), seqLRT(), seqBIC()
"""

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

__all__ = [
    'compare_groups_across_positions',
    'plot_group_differences_across_positions',
    'print_group_differences_across_positions',
    'compare_groups_overall',
    'compute_likelihood_ratio_test',
    'compute_bayesian_information_criterion_test',
]
