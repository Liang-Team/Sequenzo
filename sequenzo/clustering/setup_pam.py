from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'pam_cpp',
        ['src/PAM.cpp'],  # 注意路径：用相对路径更通用
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['/O2', '/std:c++17', '/openmp:llvm']
    ),
]

setup(
    name='pam_cpp',
    version='0.1',
    author='邓诚',
    ext_modules=ext_modules,
    zip_safe=False,
)
