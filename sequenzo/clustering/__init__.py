"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 27/02/2025 09:58
@Desc    : 
"""
from .hierarchical_clustering import Cluster, ClusterResults, ClusterQuality
from .k_medoids import k_medoids


def _import_c_code():
    """延迟导入 c_code 模块，避免在安装时的循环依赖"""
    try:
        from . import c_code
        return c_code
    except ImportError:
        # 如果 C 扩展无法导入，返回 None
        print("警告: C++ 扩展 (c_code) 无法导入。请确保正确编译了扩展模块。")
        return None


__all__ = [
    "Cluster",
    "ClusterResults",
    "ClusterQuality"
    "k_medoids"
    # Add other functions as needed
]
