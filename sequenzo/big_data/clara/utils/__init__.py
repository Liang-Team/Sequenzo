"""
@Author  : 李欣怡
@File    : __init__.py.py
@Time    : 2025/2/28 00:30
@Desc    : 
"""
from .aggregatecases import *
from .davies_bouldin import *
from .wfcmdd import *
from .k_medoids_once import k_medoids_once


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
    'k_medoids_once'
]
