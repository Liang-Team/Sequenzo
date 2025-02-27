from setuptools import setup, Extension
import pybind11

pybind11_include = pybind11.get_include()

ext_modules = [
    Extension(
        'example',
        [r'D:\college\research\QiQi\sequenzo\CLion\test\module.cpp'],
        include_dirs=[pybind11_include],
        language='c++',
    ),
]

setup(
    name='example',
    version='0.1',
    ext_modules=ext_modules,
    zip_safe=False,
)
