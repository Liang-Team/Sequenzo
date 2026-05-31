"""
@Author  : 李欣怡
@File    : __init__.py
@Time    : 2025/2/28 00:38
@Desc    : 
"""
from .clara import clara, seqclara_range
from .results import MDClaraResult
from .visualization import plot_scores_from_dataframe

# md_clara is a thin re-export of sequenzo.multidomain.clara; loading it here
# eagerly creates a circular import (multidomain.clara -> clara_engine -> big_data.clara).
def __getattr__(name: str):
    if name == "md_clara":
        from sequenzo.multidomain.clara import md_clara as _md_clara

        return _md_clara
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _import_c_code():
    """Lazily import the c_code module to avoid circular dependencies during installation"""
    try:
        from sequenzo.clustering import clustering_c_code
        return clustering_c_code
    except ImportError:
        # If the C extension cannot be imported, return None
        print(
            "Warning: The C++ extension (c_code) could not be imported. Please ensure the extension module is compiled correctly.")
        return None


__all__ = [
    'clara',
    'seqclara_range',
    'md_clara',
    'MDClaraResult',
    'plot_scores_from_dataframe',
]