"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 27/02/2025 09:58
@Desc    : 
"""
from .hierarchical_clustering import Cluster, ClusterResults, ClusterQuality
from .k_medoids import k_medoids


# Function or method where c_code is needed
def local_import_c_code():
    from sequenzo.dissimilarity_measures import c_code