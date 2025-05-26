from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        name='kmedoid_cpp',
        sources=['src/KMedoid.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['/O2', '/std:c++17'],  # Windows编译器参数
    )
]

setup(
    name='kmedoid_cpp',
    version='0.1.0',
    author='邓诚',
    ext_modules=ext_modules,
    zip_safe=False,
)
