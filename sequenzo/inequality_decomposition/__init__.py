"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 2026-03-01 10:51
@Desc    : 
Inequality decomposition APIs (KOB / Oaxaca-Blinder).
"""

from .kob import KOBDecompositionResult, get_kob_decomposition
from .oaxaca import get_oaxaca_blinder_decomposition

__all__ = [
    "get_kob_decomposition",
    "get_oaxaca_blinder_decomposition",
    "KOBDecompositionResult",
]
