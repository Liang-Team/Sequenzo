"""
@Author  : Yuqi Liang 梁彧祺
@File    : setup.py
@Time    : 27/02/2025 12:13
@Desc    : 
"""
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
import os
import sys

# Force macOS deployment target if on macOS
if sys.platform == 'darwin':
    os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'

# Try to configure the C++ extension, but allow installation to proceed even if it fails
try:
    ext_module = Pybind11Extension(
        'sequenzo.dissimilarity_measures.c_code',
        sources=['sequenzo/dissimilarity_measures/src/module.cpp'],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
            'sequenzo/dissimilarity_measures/src/'  # Include directory for other .cpp files
        ],
        extra_compile_args=['-std=c++11'] if sys.platform != 'win32' else ['/std:c++11'],
        language='c++'
    )
    ext_modules = [ext_module]
    print("C++ extension configured successfully")
except Exception as e:
    print(f"Warning: Unable to configure C++ extension: {e}")
    print("The package will be installed with a Python fallback implementation")
    ext_modules = []

setup(
    name="sequenzo",
    version="0.1.0",
    packages=find_packages(),
    package_data={
        "sequenzo.datasets": ["country_CO2_emissions.csv", "country_income.csv"],
    },
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "Pillow",
        "pybind11>=2.6.0",
        "scipy",
        "scikit-learn",
    ],
)


