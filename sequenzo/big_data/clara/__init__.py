"""
@Author  : 李欣怡
@File    : __init__.py
@Time    : 2025/2/28 00:38
@Desc    : 
"""
from .clara import clara
from .visualization import plot_scores_from_dataframe


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
    'clara',
    'plot_scores_from_dataframe'
]