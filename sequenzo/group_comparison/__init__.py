"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 2026-02-14 20:16
@Desc    : 
Group comparison API (overall LRT/BIC tests).
"""

from .group_differences import (
    get_group_differences,
    get_lrt_test,
    get_bic_test,
)

__all__ = [
    "get_group_differences",
    "get_lrt_test",
    "get_bic_test",
]
