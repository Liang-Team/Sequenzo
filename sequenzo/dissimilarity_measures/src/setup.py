"""
@Author  : 李欣怡
@File    : setup.py
@Time    : 2025/3/30 18:55
@Desc    : 
"""
from setuptools import setup, Extension
import pybind11

pybind11_include = pybind11.get_include()

ext_modules = [
    Extension(
        'c_code',
        [r'/home/xinyi_test/test/module.cpp'],
        include_dirs=[
            pybind11_include,
            r'/home/xinyi_test/test/xsimd/include'
        ],
        language='c++',
        extra_compile_args=['-O2', '-std=c++17'],
    ),
]

setup(
    name='c_code',
    version='0.1',
    ext_modules=ext_modules,
    zip_safe=False,
)
