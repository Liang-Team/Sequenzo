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
    """延迟导入 c_code 模块，避免在安装时的循环依赖"""
    try:
        from . import c_code
        return c_code
    except ImportError:
        # 如果 C 扩展无法导入，返回 None
        print("警告: C++ 扩展 (c_code) 无法导入。请确保正确编译了扩展模块。")
        return None


__all__ = [
    "get_distance_matrix",
    "get_substitution_cost_matrix"
    # Add other functions as needed
]

