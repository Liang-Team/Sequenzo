"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 15/04/2026 12:07
@Desc    : 

Sequence operations compatible with TraMineR-style workflows.

This module implements utility operations often used to reshape or recode
sequence representations:
    - seqconc
    - seqdecomp
    - seqsep
    - seqrecode
    - seqshift
    - seqasnum
"""

from .operations import seqconc, seqdecomp, seqsep, seqrecode, seqshift, seqasnum

__all__ = [
    "seqconc",
    "seqdecomp",
    "seqsep",
    "seqrecode",
    "seqshift",
    "seqasnum",
]
