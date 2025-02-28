"""
@Author  : Yuqi Liang 梁彧祺
@File    : setup.py
@Time    : 27/02/2025 12:13
@Desc    : Sequenzo Package Setup Configuration

This file is maintained for backward compatibility and to handle C++ extension compilation.
Most configuration is now in pyproject.toml.
"""

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
import os
import sys

def get_extra_compile_args():
    """
    Get platform-specific compilation arguments
    """
    if sys.platform == 'win32':
        return ['/std:c++11', '/EHsc', '/W3']  # Windows
    elif sys.platform == 'darwin':
        os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'
        return ['-std=c++11', '-Wall', '-Wextra']  # macOS
    else:
        return ['-std=c++11', '-Wall', '-Wextra']  # Linux

def configure_cpp_extension():
    """
    Configure C++ extension and handle potential compilation errors
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
        print("✅ C++ extension configured successfully")
        return [ext_module]
    except Exception as e:
        print(f"⚠️ Warning: Unable to configure C++ extension: {e}")
        print("The package will be installed with a Python fallback implementation.")
        return []

# Minimal setup configuration to handle C++ extensions
# Most configuration is now in pyproject.toml
setup(
    ext_modules=configure_cpp_extension(),
    cmdclass={"build_ext": build_ext},
)

