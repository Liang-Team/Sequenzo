"""
@Author  : Yuqi Liang 梁彧祺
@File    : setup.py
@Time    : 27/02/2025 12:13
@Desc    : Sequenzo Package Setup Configuration

This file is maintained for backward compatibility and to handle C++ & Cython extension compilation.
Most configuration is now in pyproject.toml.

Suggested command lines for developers:
    # 编译所有 Cython + C++
    python setup.py build_ext --inplace

    # 开发者模式安装
    pip install -e .
"""
from pathlib import Path
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from Cython.Build import cythonize
import pybind11
import numpy
import os
import sys
from glob import glob


def get_extra_compile_args():
    if sys.platform == 'win32':
        return ['/std:c++14', '/EHsc', '/bigobj', '/O2', '/Gy']
    elif sys.platform == 'darwin':
        os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'
        return ['-std=c++11', '-Wall', '-Wextra']
    else:
        return ['-std=c++11', '-Wall', '-Wextra']


def get_cython_compile_args():
    extra_compile_args = []
    if sys.platform == "win32":
        extra_compile_args.append("/O2")
    return extra_compile_args


def configure_cpp_extension():
    try:
        ext_module = Pybind11Extension(
            'sequenzo.dissimilarity_measures.c_code',
            sources=glob('sequenzo/dissimilarity_measures/src/*.cpp'),
            include_dirs=[
                pybind11.get_include(),
                pybind11.get_include(user=True),
                'sequenzo/dissimilarity_measures/src/',
            ],
            extra_compile_args=get_extra_compile_args(),
            language='c++',
        )
        print("C++ extension configured successfully")
        return [ext_module]
    except Exception as e:
        print(f"Warning: Unable to configure C++ extension: {e}")
        print("The package will be installed with a Python fallback implementation.")
        return []


def configure_cython_extensions():
    """
    Currently, there are two places that use cython:
    clustering/utils and dissimilarity_measures/utils.
    To avoid calling many cython files manually, I set up this function here.
    """
    # Search all the pyx files.
    try:
        pyx_paths = [
            Path("sequenzo/clustering/utils/point_biserial.pyx").as_posix(),
            Path("sequenzo/dissimilarity_measures/utils/get_sm_trate_substitution_cost_matrix.pyx").as_posix(),
            Path("sequenzo/dissimilarity_measures/utils/seqconc.pyx").as_posix(),
            Path("sequenzo/dissimilarity_measures/utils/seqdss.pyx").as_posix(),
            Path("sequenzo/dissimilarity_measures/utils/seqdur.pyx").as_posix(),
            Path("sequenzo/dissimilarity_measures/utils/seqlength.pyx").as_posix(),
            Path("sequenzo/big_data/clara/utils/get_weighted_diss.pyx").as_posix(),
        ]

        extensions = [
            Extension(
                name=path.replace("/", ".").replace(".pyx", ""),
                sources=[path],
                include_dirs=[numpy.get_include()],
                extra_compile_args=get_cython_compile_args(),
            )
            for path in pyx_paths
        ]
        print(f"Found {len(extensions)} Cython modules.")
        return cythonize(extensions, compiler_directives={"language_level": "3"})
    except Exception as e:
        print(f"Warning: Unable to configure Cython extensions: {e}")
        return []


# 防止路径缺失导致安装报错
os.makedirs("sequenzo/dissimilarity_measures/src", exist_ok=True)
os.makedirs("sequenzo/clustering/utils", exist_ok=True)

# 正式调用
setup(
    ext_modules=configure_cpp_extension() + configure_cython_extensions(),
    cmdclass={"build_ext": build_ext},
)
