"""
@Author  : Yuqi Liang 梁彧祺
@File    : md_clara.py
@Time    : 18/05/2026 21:37
@Desc    : 
Compatibility re-export for multidomain CLARA.

Implementation lives in :mod:`sequenzo.multidomain.clara`.
"""

from sequenzo.multidomain.clara import MDClaraResult, md_clara

__all__ = ["md_clara", "MDClaraResult"]
