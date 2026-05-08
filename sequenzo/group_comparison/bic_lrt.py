"""
@Author  : Yuqi Liang 梁彧祺
@File    : bic_lrt.py
@Time    : 2026-02-16 15:01
@Desc    : 
Overall two-group comparison metrics (LRT/BIC).
"""

from .seqcompare import (
    compare_groups_overall,
    compute_likelihood_ratio_test,
    compute_bayesian_information_criterion_test,
)

get_group_differences_overall = compare_groups_overall
get_lrt_test = compute_likelihood_ratio_test
get_bic_test = compute_bayesian_information_criterion_test

__all__ = [
    "get_group_differences_overall",
    "get_lrt_test",
    "get_bic_test",
]
