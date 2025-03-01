"""
@Author  : 李欣怡
@File    : __init__.py
@Time    : 2025/2/26 23:19
@Desc    : 
"""
from .utils import get_sm_trate_substitution_cost_matrix, seqconc, seqdss, seqdur, seqlength
from .get_distance_matrix import get_distance_matrix
from .get_substitution_cost_matrix import get_substitution_cost_matrix


def _import_c_code():
    """Lazily import the c_code module to avoid circular dependencies during installation"""
    try:
        from sequenzo.dissimilarity_measures import c_code
        return c_code
    except ImportError:
        # If the C extension cannot be imported, return None
        print(
            "Warning: The C++ extension (c_code) could not be imported. Please ensure the extension module is compiled correctly.")
        return None


__all__ = [
    "get_distance_matrix",
    "get_substitution_cost_matrix"
    # Add other functions as needed
]

