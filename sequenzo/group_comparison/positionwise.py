"""
@Author  : Yuqi Liang 梁彧祺
@File    : positionwise.py
@Time    : 2026-02-16 07:35
@Desc    : Position-wise discrepancy comparison between groups.
"""

from .seqdiff import (
    compare_groups_across_positions,
    plot_group_differences_across_positions,
    print_group_differences_across_positions,
)

get_group_differences_by_position = compare_groups_across_positions
plot_group_differences_by_position = plot_group_differences_across_positions
get_group_differences_report_by_position = print_group_differences_across_positions

__all__ = [
    "get_group_differences_by_position",
    "plot_group_differences_by_position",
    "get_group_differences_report_by_position",
]
