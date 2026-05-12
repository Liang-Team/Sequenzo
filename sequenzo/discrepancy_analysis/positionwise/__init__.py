"""Position-wise discrepancy comparisons (TraMineR seqdiff-style workflows)."""

from .compare_by_position import (
    compare_groups_across_positions,
    plot_group_differences_across_positions,
    print_group_differences_across_positions,
)

__all__ = [
    "compare_groups_across_positions",
    "plot_group_differences_across_positions",
    "print_group_differences_across_positions",
]
