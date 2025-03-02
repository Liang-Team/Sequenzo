"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 27/02/2025 09:58
@Desc    : 
"""
from .hierarchical_clustering import Cluster, ClusterResults, ClusterQuality


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
    "Cluster",
    "ClusterResults",
    "ClusterQuality"
    "k_medoids"
    # Add other functions as needed
]
