"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 2026/02/05 9:20
@Desc    : 
Python-implemented distance measures.

This package contains distance measures that are implemented in Python
rather than C++ for easier maintenance and flexibility.
"""

from .omstran import create_transition_sequences, build_omstran_substitution_matrix
from .chi2_euclid import chi2_euclid_distances, build_chi2_allmat_pdotj

__all__ = [
    'create_transition_sequences',
    'build_omstran_substitution_matrix',
    'chi2_euclid_distances',
    'build_chi2_allmat_pdotj',
]
