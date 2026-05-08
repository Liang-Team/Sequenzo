"""
@Desc: Position-wise/local-window discrepancy analysis wrappers.
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
