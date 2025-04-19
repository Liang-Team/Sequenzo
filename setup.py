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
import subprocess
from glob import glob
import tempfile

def ensure_xsimd_exists():
    xsimd_dir = Path(__file__).parent / "sequenzo" / "dissimilarity_measures" / "src" / "xsimd"

    # 如果目录不存在或为空，则 clone
    if xsimd_dir.exists() and any(xsimd_dir.iterdir()):
        print(f"[INFO] xsimd already exists at {xsimd_dir}, skipping clone.")
        return

    print(f"[INFO] xsimd not found or empty at {xsimd_dir}, attempting to clone...")
    try:
        # 如果目录存在但为空，也要确保上级目录存在
        xsimd_dir.parent.mkdir(parents=True, exist_ok=True)

        if xsimd_dir.exists():
            xsimd_dir.rmdir()

        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/xtensor-stack/xsimd.git",
            str(xsimd_dir)
        ], check=True)
        print("[INFO] xsimd cloned successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to clone xsimd automatically: {e}")


ensure_xsimd_exists()


def get_mac_arch():
    """
    Detects the current macOS architecture.
    Returns:
        str: 'x86_64' or 'arm64' depending on the Mac hardware.
    """
    try:
        return subprocess.check_output(['uname', '-m']).decode().strip()
    except Exception:
        return None


def has_openmp_support():
    """
    Check if the current compiler supports OpenMP.
    Returns:
        bool
    """
    if getattr(has_openmp_support, "_checked", False):
        return has_openmp_support._result

    try:
        test_code = '#include <omp.h>\nint main() { return 0; }'
        temp_dir = tempfile.gettempdir()
        source_path = os.path.join(temp_dir, 'test_openmp.cpp')
        binary_path = os.path.join(temp_dir, 'test_openmp')

        with open(source_path, 'w') as f:
            f.write(test_code)

        result = subprocess.run(
            ['clang++', '-fopenmp', source_path, '-o', binary_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Clean up
        os.remove(source_path)
        if os.path.exists(binary_path):
            os.remove(binary_path)

        supported = result.returncode == 0
        if supported:
            print("OpenMP is supported. Enabling -fopenmp.")
        else:
            print("OpenMP is NOT supported by the current compiler. Skipping -fopenmp.")

        has_openmp_support._result = supported
        has_openmp_support._checked = True
        return supported

    except Exception:
        has_openmp_support._result = False
        has_openmp_support._checked = True
        return False


def get_compile_args_for_file(filename):
    if sys.platform == 'win32':
        base_cflags = ['/W4', '/bigobj']
        base_cppflags = ['/std:c++17'] + base_cflags
        openmp_flag = ['/openmp']
        # openmp_flag = []
    else:
        base_cflags = ['-Wall', '-Wextra']
        base_cppflags = ['-std=c++17'] + base_cflags
        openmp_flag = ['-fopenmp'] if has_openmp_support() else []
        # openmp_flag = []

    if sys.platform == 'darwin':
        os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'
        arch = get_mac_arch()
        if arch in ('x86_64', 'arm64'):
            arch_flags = ['-arch', arch]
        else:
            arch_flags = []
    else:
        arch_flags = []

    if filename.endswith(".cpp"):
        return base_cppflags + arch_flags + openmp_flag
    else:
        return base_cflags + arch_flags + openmp_flag


def get_include_dirs():
    """
    Collects all required include directories for compiling C++ and Cython code.
    Returns:
        list: Paths to include directories.
    """
    base_dir = Path(__file__).parent.resolve()
    return [
        pybind11.get_include(),
        pybind11.get_include(user=True),
        numpy.get_include(),
        'sequenzo/dissimilarity_measures/src/',
        str(base_dir / 'sequenzo' / 'dissimilarity_measures' / 'src' / 'xsimd' / 'include'),
    ]


def configure_cpp_extension():
    """
    Configures the Pybind11 C++ extension module.
    Returns:
        list: A list with one or zero configured Pybind11Extension.
    """
    try:
        ext_module = Pybind11Extension(
            'sequenzo.dissimilarity_measures.c_code',
            sources=glob('sequenzo/dissimilarity_measures/src/*.cpp'),
            include_dirs=get_include_dirs(),
            extra_compile_args=get_compile_args_for_file("dummy.cpp"),
            language='c++',
            define_macros=[('VERSION_INFO', '"0.0.1"')],
        )
        print("C++ extension configured successfully")
        return [ext_module]
    except Exception as e:
        print(f"Failed to configure C++ extension: {e}")
        print("Fallback: Python-only functionality will be used.")
        return []


def configure_cython_extensions():
    """
    Configures and compiles all .pyx files via Cython.
    Returns:
        list: Compiled Cython extensions (or empty list if failed).
    """
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

        extensions = []
        for path in pyx_paths:
            extra_args = get_compile_args_for_file(path)
            extension = Extension(
                name=str(Path(path).with_suffix("")).replace("/", ".").replace("\\", "."),
                sources=[path],
                include_dirs=get_include_dirs(),
                extra_compile_args=extra_args,
            )
            extensions.append(extension)
        print(f"Found {len(extensions)} Cython modules.")
        return cythonize(extensions, compiler_directives={"language_level": "3"})
    except Exception as e:
        print(f"Failed to configure Cython extensions: {e}")
        return []


class BuildExt(build_ext):
    """
    Custom build_ext class that prints architecture info on macOS.
    """
    def build_extensions(self):
        if sys.platform == 'darwin':
            arch = get_mac_arch()
            print(f"Compiling extensions for macOS [{arch}]...")
        super().build_extensions()


# Ensure necessary folders exist to prevent file not found errors
os.makedirs("sequenzo/dissimilarity_measures/src", exist_ok=True)
os.makedirs("sequenzo/clustering/utils", exist_ok=True)

# Run the actual setup process
setup(
    ext_modules=configure_cpp_extension() + configure_cython_extensions(),
    cmdclass={"build_ext": BuildExt},
)