"""
@Author  : Yuqi Liang 梁彧祺
@File    : setup.py
@Time    : 27/02/2025 12:13
@Desc    : Sequenzo Package Setup Configuration
"""
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
import os
import sys

def read_requirements(filename):
    """
    读取依赖文件
    """
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

def get_extra_compile_args():
    """
    获取平台特定的编译参数
    """
    if sys.platform == 'win32':
        return ['/std:c++11', '/EHsc', '/W3']  # 添加警告级别
    elif sys.platform == 'darwin':
        os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'
        return ['-std=c++11', '-Wall', '-Wextra']  # 添加更多警告
    else:
        return ['-std=c++11', '-Wall', '-Wextra']

def configure_cpp_extension():
    """
    配置 C++ 扩展，并处理可能的编译错误
    """
    try:
        ext_module = Pybind11Extension(
            'sequenzo.dissimilarity_measures.c_code',
            sources=['sequenzo/dissimilarity_measures/src/module.cpp'],
            include_dirs=[
                pybind11.get_include(),
                pybind11.get_include(user=True),
                'sequenzo/dissimilarity_measures/src/'
            ],
            extra_compile_args=get_extra_compile_args(),
            language='c++',
        )
        print("C++ extension configured successfully")
        return [ext_module]
    except Exception as e:
        print(f"Warning: Unable to configure C++ extension: {e}")
        print("The package will be installed with a Python fallback implementation")
        return []

# 动态选择对应 Python 版本的依赖文件
def get_requirements_file():
    """
    根据当前 Python 版本选择合适的依赖文件
    """
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if python_version.startswith('3.8'):
        return 'requirements-3.8.txt'
    elif python_version.startswith('3.9'):
        return 'requirements-3.9.txt'
    elif python_version.startswith('3.10'):
        return 'requirements-3.10.txt'
    else:
        return 'requirements-3.8.txt'  # 默认使用 3.8 的依赖

# 主 setup 配置
setup(
    name="sequenzo",
    version="0.1.0",
    description="A fast, scalable and intuitive Python package for social sequence analysis",
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',

    # 包信息
    packages=find_packages(exclude=['tests*']),
    package_data={
        "sequenzo.datasets": ["country_CO2_emissions.csv", "country_income.csv"],
    },

    # C++ 扩展
    ext_modules=configure_cpp_extension(),
    cmdclass={"build_ext": build_ext},

    # 依赖管理
    python_requires='>=3.8,<3.11',
    install_requires=read_requirements(get_requirements_file()),
    extras_require={
        'dev': read_requirements('requirements-dev.txt')
    },

    # 额外的元数据
    author="Yuqi Liang",
    author_email="yuqi.liang.1900@gmail.com",
    url="https://github.com/Liang-Team/Sequenzo",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)

