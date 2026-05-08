"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 2026-02-14 20:16
@Desc    : 
Group comparison API (position-wise and overall tests).
"""

from .positionwise import (
    get_group_differences_by_position,
    plot_group_differences_by_position,
    get_group_differences_report_by_position,
)
from .bic_lrt import (
    get_group_differences_overall,
    get_lrt_test,
    get_bic_test,
)

__all__ = [
    "get_group_differences_by_position",
    "plot_group_differences_by_position",
    "get_group_differences_report_by_position",
    "get_group_differences_overall",
    "get_lrt_test",
    "get_bic_test",
]
