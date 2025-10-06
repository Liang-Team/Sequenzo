"""
@Author  : Yuqi Liang 梁彧祺
@File    : setup.py
@Time    : 27/02/2025 12:13
@Desc    : Sequenzo Package Setup Configuration

This file is maintained for backward compatibility and to handle C++ & Cython extension compilation.
Most configuration is now in pyproject.toml.

Architecture Control (macOS):
    # Intel Mac only (faster compilation, smaller files)
    export SEQUENZO_ARCH=x86_64
    pip install -e .
    
    # Apple Silicon only
    export SEQUENZO_ARCH=arm64
    pip install -e .
    
    # Universal Binary (default, works on all Macs)
    export ARCHFLAGS="-arch x86_64 -arch arm64"
    pip install -e .
    
    # Let system auto-detect (recommended for most users)
    pip install -e .

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
    Intelligently detects the target macOS architecture for compilation.
    
    Priority:
    1. ARCHFLAGS environment variable (user override)
    2. SEQUENZO_ARCH environment variable (project-specific)
    3. Current hardware architecture
    
    Returns:
        str or list: Architecture string(s) for compilation
    """
    # Check for user-specified architecture flags
    archflags = os.environ.get('ARCHFLAGS', '').strip()
    if archflags:
        # Parse ARCHFLAGS like "-arch x86_64 -arch arm64"
        archs = []
        parts = archflags.split()
        for i, part in enumerate(parts):
            if part == '-arch' and i + 1 < len(parts):
                archs.append(parts[i + 1])
        if archs:
            print(f"[SETUP] Using ARCHFLAGS architectures: {archs}")
            return archs
    
    # Check for project-specific override
    project_arch = os.environ.get('SEQUENZO_ARCH', '').strip()
    if project_arch:
        print(f"[SETUP] Using SEQUENZO_ARCH: {project_arch}")
        return project_arch
    
    # Default: detect current hardware
    try:
        hardware_arch = subprocess.check_output(['uname', '-m']).decode().strip()
        print(f"[SETUP] Using hardware architecture: {hardware_arch}")
        return hardware_arch
    except Exception:
        print("[SETUP] Warning: Could not detect architecture, defaulting to x86_64")
        return 'x86_64'


def has_openmp_support():
    """
    Check if the current compiler supports OpenMP.
    Can be forced via SEQUENZO_ENABLE_OPENMP environment variable.
    Returns:
        bool
    """
    # Check for forced OpenMP enable (for CI/CD)
    if os.environ.get('SEQUENZO_ENABLE_OPENMP', '').strip().lower() in ('1', 'true', 'on', 'yes'):
        print("[SETUP] OpenMP force-enabled via SEQUENZO_ENABLE_OPENMP")
        return True
    
    if getattr(has_openmp_support, "_checked", False):
        return has_openmp_support._result

    try:
        test_code = '#include <omp.h>\nint main() { return 0; }'
        temp_dir = tempfile.gettempdir()
        source_path = os.path.join(temp_dir, 'test_openmp.cpp')
        
        with open(source_path, 'w') as f:
            f.write(test_code)

        if sys.platform == 'win32':
            # Windows: 尝试MSVC编译器，如果失败则假设支持OpenMP
            binary_path = os.path.join(temp_dir, 'test_openmp.exe')
            try:
                result = subprocess.run(
                    ['cl', '/openmp', source_path, '/Fe:' + binary_path],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                # If cl is not available or times out, assume OpenMP is supported
                # This happens in some CI environments
                print("[SETUP] Could not test OpenMP with cl compiler, assuming supported")
                has_openmp_support._result = True
                has_openmp_support._checked = True
                return True
        else:
            # macOS/Linux: 使用clang++/g++
            binary_path = os.path.join(temp_dir, 'test_openmp')
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

def get_homebrew_libomp_paths():
    """检测 libomp 的 Homebrew 路径（兼容 Intel / Apple Silicon）"""
    candidates = [
        "/opt/homebrew/opt/libomp",  # Apple Silicon (ARM64)
        "/usr/local/opt/libomp",     # Intel macOS
    ]
    for path in candidates:
        if os.path.exists(path):
            include_dir = os.path.join(path, "include")
            lib_dir = os.path.join(path, "lib")
            return include_dir, lib_dir
    return None, None

def get_macos_openmp_config():
    """
    检测 macOS 上的 libomp 安装路径并返回 (compile_flags, link_flags)
    """
    try:
        # 允许用户通过环境变量指定 libomp 安装位置
        brew_prefix = os.environ.get("LIBOMP_PREFIX")
        if not brew_prefix:
            # 尝试通过 Homebrew 获取路径
            brew_prefix = subprocess.check_output(
                ["brew", "--prefix", "libomp"],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()
    except Exception:
        brew_prefix = "/usr/local"  # fallback

    include_dir = os.path.join(brew_prefix, "include")
    lib_dir = os.path.join(brew_prefix, "lib")

    compile_flags = [
        "-Xpreprocessor", "-fopenmp",
        f"-I{include_dir}"
    ]
    link_flags = [
        f"-L{lib_dir}",
        "-lomp"
    ]

    print(f"[SETUP] macOS OpenMP from: {brew_prefix}")
    print(f"[SETUP] OpenMP compile flags: {' '.join(compile_flags)}")
    print(f"[SETUP] OpenMP link flags: {' '.join(link_flags)}")

    return compile_flags, link_flags


def get_compile_args_for_file(filename):
    if sys.platform == 'win32':
        base_cflags = ['/W1', '/bigobj']
        base_cppflags = ['/std:c++17'] + base_cflags
        openmp_flag = ['/openmp:experimental'] if has_openmp_support() else []
        compile_args = ["/O2"]
        arch_flags = []
    else:
        base_cflags = ['-Wall', '-Wextra']
        base_cppflags = ['-std=c++17'] + base_cflags
        arch_flags = []
        openmp_flag = []
        compile_args = []

        if sys.platform == 'darwin':
            os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'
            compile_args = ["-O3", "-ffast-math"]

            if has_openmp_support():
                omp_compile_flags, omp_link_flags = get_macos_openmp_config()
                openmp_flag.extend(omp_compile_flags)
                os.environ["LDFLAGS"] = " ".join(omp_link_flags)
                # ✅ 在这里设置 LDFLAGS，供 setuptools/distutils 自动拾取
                print(f"[SETUP] LDFLAGS set: {os.environ['LDFLAGS']}")
        else:
            if has_openmp_support():
                openmp_flag = ['-fopenmp']
            compile_args = ["-O3", "-march=native", "-ffast-math"]

    if filename.endswith(".cpp"):
        return base_cppflags + arch_flags + openmp_flag + compile_args
    else:
        return base_cflags + arch_flags + openmp_flag + compile_args


def get_dissimilarity_measures_include_dirs():
    """
    Collects all required include directories for compiling C++ and Cython code in dissimilarity measures.
    Returns:
        list: Paths to include directories in dissimilarity measures.
    """
    base_dir = Path(__file__).parent.resolve()
    return [
        pybind11.get_include(),
        pybind11.get_include(user=True),
        numpy.get_include(),
        'sequenzo/dissimilarity_measures/src/',
        str(base_dir / 'sequenzo' / 'dissimilarity_measures' / 'src' / 'xsimd' / 'include'),
        'sequenzo/clustering/src/',
    ]

def get_clustering_include_dirs():
    """
    Collects all required include directories for compiling C++ and Cython code in clustering measures.
    Returns:
        list: Paths to include directories in clustering measures.
    """
    return [
        pybind11.get_include(),
        pybind11.get_include(user=True),
        numpy.get_include(),
        'sequenzo/clustering/src/',
    ]


def get_link_args():
    """获取平台特定的链接参数"""
    if has_openmp_support():
        if sys.platform == 'darwin':
            _, omp_link_flags = get_macos_openmp_config()
            return omp_link_flags
        elif sys.platform == 'win32':
            return []  # Windows MSVC自动链接
        else:
            return ['-lgomp']
    return []

def configure_cpp_extension():
    """
    Configures the Pybind11 C++ extension module.
    Returns:
        list: A list with one or zero configured Pybind11Extension.
    """
    try:
        link_args = get_link_args()
        
        # Compile only the binding translation unit to avoid duplicate symbols.
        # The binding TU `module.cpp` includes the other implementation .cpp files.
        diss_ext_module = Pybind11Extension(
            'sequenzo.dissimilarity_measures.c_code',
            sources=['sequenzo/dissimilarity_measures/src/module.cpp'],
            include_dirs=get_dissimilarity_measures_include_dirs(),
            extra_compile_args=get_compile_args_for_file("dummy.cpp"),
            extra_link_args=link_args,
            language='c++',
            define_macros=[('VERSION_INFO', '"0.0.1"'),
                           ('NPY_NO_DEPRECATED_API', 'NPY_1_23_API_VERSION')],
        )
        print("  - Dissimilarity measures C++ extension configured successfully.")

        # Same for clustering: compile only the binding TU.
        clustering_ext_module = Pybind11Extension(
            'sequenzo.clustering.clustering_c_code',
            sources=['sequenzo/clustering/src/module.cpp'],
            include_dirs=get_clustering_include_dirs(),
            extra_compile_args=get_compile_args_for_file("dummy.cpp"),
            extra_link_args=link_args,
            language='c++',
            define_macros=[('VERSION_INFO', '"0.0.1"'),
                           ('NPY_NO_DEPRECATED_API', 'NPY_1_23_API_VERSION')],
        )
        print("  - Clustering C++ extension configured successfully.")

        print("C++ extension configured successfully.")
        return [diss_ext_module, clustering_ext_module]
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
            # point_biserial.pyx removed - using C++ implementation instead
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
                include_dirs=get_dissimilarity_measures_include_dirs(),
                extra_compile_args=extra_args,
                define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_23_API_VERSION')]
            )
            extensions.append(extension)
        print(f"Found {len(extensions)} Cython modules.")
        return cythonize(extensions, compiler_directives={"language_level": "3"}, force=True)
    except Exception as e:
        print(f"Failed to configure Cython extensions: {e}")
        return []


class BuildExt(build_ext):
    """
    Custom build_ext class with enhanced architecture and OpenMP reporting.
    """
    def build_extensions(self):
        if sys.platform == 'darwin':
            arch = get_mac_arch()
            if isinstance(arch, list):
                print(f"[SETUP] Compiling Universal Binary for macOS: {arch}")
            else:
                print(f"[SETUP] Compiling for macOS [{arch}]")
            
            # Show OpenMP status
            if has_openmp_support():
                print("[SETUP] OpenMP support detected - parallel compilation enabled")
            else:
                print("[SETUP] OpenMP not available - using serial compilation")
        
        print(f"[SETUP] Building {len(self.extensions)} extension(s)...")
        super().build_extensions()
        print("[SETUP] Extension compilation completed!")


# Ensure necessary folders exist to prevent file not found errors
os.makedirs("sequenzo/dissimilarity_measures/src", exist_ok=True)
os.makedirs("sequenzo/clustering/src", exist_ok=True)
os.makedirs("sequenzo/clustering/utils", exist_ok=True)

# Run the actual setup process
setup(
    ext_modules=configure_cpp_extension() + configure_cython_extensions(),
    cmdclass={"build_ext": BuildExt},
)