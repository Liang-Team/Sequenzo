"""
@Author  : 梁彧祺
@File    : __init__.py
@Time    : 11/02/2025 16:42
@Desc    : Sequenzo Package Initialization
"""

# 定义版本号 - 这必须在所有导入之前
__version__ = "0.1.0"


# 使用惰性导入来避免安装时的循环依赖问题
def __getattr__(name):
    """
    延迟导入主要组件，以解决安装过程中的循环依赖问题。
    这允许 setuptools 获取 __version__ 而不需要先安装所有依赖项。
    """
    if name == "datasets":
        from . import datasets
        return datasets
    elif name == "visualization":
        from . import visualization
        return visualization
    elif name == "clustering":
        from . import clustering
        return clustering
    elif name == "dissimilarity_measures":
        from . import dissimilarity_measures
        return dissimilarity_measures
    elif name == "SequenceData":
        from .define_sequence_data import SequenceData
        return SequenceData
    elif name == "big_data":
        from .big_data import clara
        return clara

    raise AttributeError(f"module 'sequenzo' has no attribute '{name}'")


# 这些是包的公共 API，但使用 __getattr__ 延迟导入
__all__ = [
    'datasets',
    'visualization',
    'clustering',
    'dissimilarity_measures',
    'SequenceData',
    'big_data',
]
