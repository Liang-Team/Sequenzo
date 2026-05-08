"""
@Desc: Core discrepancy analysis statistics.
"""

from .tree_utils import (
    compute_pseudo_variance,
    compute_distance_association,
    dissmfacw,
    dissmergegroups,
)
from .dissassoc_permutation import (
    distance_multifactor_anova,
    compute_distance_indicators,
)

get_discrepancy = compute_pseudo_variance
get_group_distance_association = compute_distance_association
get_multifactor_discrepancy_anova = distance_multifactor_anova
get_discrepancy_indicators = compute_distance_indicators

__all__ = [
    "get_discrepancy",
    "get_group_distance_association",
    "get_multifactor_discrepancy_anova",
    "get_discrepancy_indicators",
    "dissmfacw",
    "dissmergegroups",
]
