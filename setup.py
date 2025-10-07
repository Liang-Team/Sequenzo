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
from setuptools.command.install import install
from setuptools.command.bdist_wheel import bdist_wheel
from pybind11.setup_helpers import Pybind11Extension, build_ext
from Cython.Build import cythonize
import pybind11
import numpy
import os
import sys
import subprocess
from glob import glob
import tempfile
import importlib.util

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
    # Cache the result to avoid repeated detection
    if hasattr(get_mac_arch, '_cached_result'):
        return get_mac_arch._cached_result
    
    print("[SETUP] === Detecting macOS architecture ===")
    
    # Check for user-specified architecture flags
    archflags = os.environ.get('ARCHFLAGS', '').strip()
    print(f"[SETUP] ARCHFLAGS environment variable: '{archflags}'")
    
    if archflags:
        # Parse ARCHFLAGS like "-arch x86_64 -arch arm64"
        archs = []
        parts = archflags.split()
        for i, part in enumerate(parts):
            if part == '-arch' and i + 1 < len(parts):
                archs.append(parts[i + 1])
        if archs:
            print(f"[SETUP] OK - Using ARCHFLAGS architectures: {archs}")
            get_mac_arch._cached_result = archs
            return archs
        else:
            print(f"[SETUP] Warning: ARCHFLAGS set but no architectures found")
    
    # Check for project-specific override
    project_arch = os.environ.get('SEQUENZO_ARCH', '').strip()
    if project_arch:
        print(f"[SETUP] OK - Using SEQUENZO_ARCH: {project_arch}")
        get_mac_arch._cached_result = project_arch
        return project_arch
    
    # Default: detect current hardware
    try:
        hardware_arch = subprocess.check_output(['uname', '-m']).decode().strip()
        print(f"[SETUP] OK - Using hardware architecture: {hardware_arch}")
        get_mac_arch._cached_result = hardware_arch
        return hardware_arch
    except Exception:
        print("[SETUP] Warning: Could not detect architecture, defaulting to x86_64")
        get_mac_arch._cached_result = 'x86_64'
        return 'x86_64'


def install_libomp_on_apple_silicon():
    """
    Automatically install libomp on Apple Silicon Macs if needed.
    This function is called during setup to ensure OpenMP is available.
    """
    try:
        # Import the OpenMP setup module without importing sequenzo __init__
        project_root = Path(__file__).parent.resolve()
        openmp_setup_path = project_root / 'sequenzo' / 'openmp_setup.py'
        spec = importlib.util.spec_from_file_location('sequenzo_openmp_setup', str(openmp_setup_path))
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, 'ensure_openmp_support')()
        # Fallback to basic check if the module cannot be loaded
        import platform
        
        # Only run on macOS Apple Silicon
        if sys.platform != 'darwin' or platform.machine() != 'arm64':
            return True
        
        # Check if we're in a conda environment (don't interfere)
        if os.environ.get('CONDA_DEFAULT_ENV'):
            return True
        
        # Check if Homebrew is available
        try:
            subprocess.run(['brew', '--version'], 
                          stdout=subprocess.DEVNULL, 
                          stderr=subprocess.DEVNULL, 
                          check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("""
OpenMP Dependency Detection

On Apple Silicon Mac, Sequenzo requires OpenMP support for parallel computation.

Please run the following command to install OpenMP support:
    brew install libomp

If you don't have Homebrew installed, please visit https://brew.sh to install Homebrew first.
            """)
            return False
        
        # Check if libomp is already installed
        try:
            subprocess.run(['brew', 'list', 'libomp'], 
                          stdout=subprocess.DEVNULL, 
                          stderr=subprocess.DEVNULL, 
                          check=True)
            print("[SETUP] libomp is already installed")
        except subprocess.CalledProcessError:
            # Attempt to install libomp automatically
            print("[SETUP] Detected Apple Silicon Mac, auto-installing OpenMP support...")
            try:
                result = subprocess.run(['brew', 'install', 'libomp'], 
                                      check=True, 
                                      stdout=subprocess.PIPE, 
                                      stderr=subprocess.PIPE)
                print("[SETUP] OpenMP support installed successfully!")
            except subprocess.CalledProcessError as e:
                print(f"""
[SETUP] Automatic OpenMP installation failed: {e}

Please manually run the following command:
    brew install libomp

After installation, please re-run the installation command.
                """)
                return False
        
        # 关键修复：自动设置编译器环境变量
        print("[SETUP] Setting up compiler environment for OpenMP...")
        try:
            # 获取 Homebrew 路径
            brew_prefix = subprocess.check_output(['brew', '--prefix'], text=True).strip()
            llvm_path = f"{brew_prefix}/opt/llvm"
            
            # 检查 LLVM 是否已安装
            if os.path.exists(f"{llvm_path}/bin/clang"):
                # 设置编译器环境变量
                os.environ['CC'] = f"{llvm_path}/bin/clang"
                os.environ['CXX'] = f"{llvm_path}/bin/clang++"
                os.environ['LDFLAGS'] = f"-L{llvm_path}/lib -L{brew_prefix}/lib {os.environ.get('LDFLAGS', '')}"
                os.environ['CPPFLAGS'] = f"-I{llvm_path}/include -I{brew_prefix}/include {os.environ.get('CPPFLAGS', '')}"
                os.environ['DYLD_LIBRARY_PATH'] = f"{llvm_path}/lib:{brew_prefix}/lib:{os.environ.get('DYLD_LIBRARY_PATH', '')}"
                
                print(f"[SETUP] Compiler environment configured:")
                print(f"   - CC: {os.environ['CC']}")
                print(f"   - CXX: {os.environ['CXX']}")
                return True
            else:
                print("[SETUP] LLVM not found via Homebrew, trying to install...")
                try:
                    subprocess.run(['brew', 'install', 'llvm'], check=True)
                    # 重新尝试设置环境变量
                    if os.path.exists(f"{llvm_path}/bin/clang"):
                        os.environ['CC'] = f"{llvm_path}/bin/clang"
                        os.environ['CXX'] = f"{llvm_path}/bin/clang++"
                        os.environ['LDFLAGS'] = f"-L{llvm_path}/lib -L{brew_prefix}/lib {os.environ.get('LDFLAGS', '')}"
                        os.environ['CPPFLAGS'] = f"-I{llvm_path}/include -I{brew_prefix}/include {os.environ.get('CPPFLAGS', '')}"
                        os.environ['DYLD_LIBRARY_PATH'] = f"{llvm_path}/lib:{brew_prefix}/lib:{os.environ.get('DYLD_LIBRARY_PATH', '')}"
                        print("[SETUP] LLVM installed and environment configured!")
                        return True
                except subprocess.CalledProcessError:
                    print("[SETUP] Could not install LLVM, will use system clang (may not support OpenMP)")
                    return True
        except Exception as e:
            print(f"[SETUP] Could not configure compiler environment: {e}")
            print("[SETUP] Will attempt to build with system compiler (may not support OpenMP)")
            return True
        
        return True
    except Exception:
        # As a last resort, don't block configuration here
        return True


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
    
    # Auto-install libomp on Apple Silicon if needed
    import platform
    if sys.platform == 'darwin' and platform.machine() == 'arm64':
        if not install_libomp_on_apple_silicon():
            print("[SETUP] OpenMP dependency installation failed, will use serial version")
            return False
    
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


def get_compile_args_for_file(filename):
    if sys.platform == 'win32':
        base_cflags = ['/W1', '/bigobj']  # Reduced warning level for faster compilation
        base_cppflags = ['/std:c++17'] + base_cflags
        
        # Windows OpenMP support
        if has_openmp_support():
            openmp_flag = ['/openmp:experimental']
            print("[SETUP] Windows OpenMP flags: /openmp")
        else:
            openmp_flag = []
    else:
        base_cflags = ['-Wall', '-Wextra']
        base_cppflags = ['-std=c++17'] + base_cflags
        
        # OpenMP flags with platform-specific optimization
        if has_openmp_support():
            if sys.platform == 'darwin':
                # macOS: 使用libomp，分离编译和链接标志
                openmp_flag = ['-Xpreprocessor', '-fopenmp']
                print("[SETUP] macOS OpenMP flags: -Xpreprocessor -fopenmp")
            else:
                # Linux/Other: 使用libgomp
                openmp_flag = ['-fopenmp']  
                print("[SETUP] Linux OpenMP flags: -fopenmp")
        else:
            openmp_flag = []

    if sys.platform == 'win32':
        # More conservative Windows flags for better compatibility
        compile_args = ["/O2"]
    else:
        # Use -mcpu=native for Apple Silicon, avoid -march=native which is not supported by clang
        if sys.platform == 'darwin':
            compile_args = ["-O3", "-ffast-math"]
        else:
            compile_args = ["-O3", "-march=native", "-ffast-math"]

    if sys.platform == 'darwin':
        os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'
        arch = get_mac_arch()
        
        # Handle both single architecture and multiple architectures
        if isinstance(arch, list):
            # Multiple architectures (Universal Binary)
            arch_flags = []
            for a in arch:
                if a in ('x86_64', 'arm64'):
                    arch_flags.extend(['-arch', a])
            print(f"[SETUP] Compiling for universal2: {arch}")
            print(f"[SETUP] Architecture flags: {arch_flags}")
        elif isinstance(arch, str) and arch in ('x86_64', 'arm64'):
            # Single architecture
            arch_flags = ['-arch', arch]
            print(f"[SETUP] Compiling for single architecture: {arch}")
        else:
            # Unknown or unsupported architecture
            arch_flags = []
            if arch:
                print(f"[SETUP] Warning: Unsupported architecture '{arch}', skipping arch flags")
    else:
        arch_flags = []

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
    link_args = []
    
    if has_openmp_support():
        if sys.platform == 'darwin':
            # macOS: Link against libomp and set rpath
            link_args.append('-lomp')
            
            # Add library path from environment or Homebrew default
            ldflags = os.environ.get('LDFLAGS', '')
            if ldflags:
                # Parse LDFLAGS for -L and -Wl,-rpath options
                for flag in ldflags.split():
                    if flag.startswith('-L') or flag.startswith('-Wl,'):
                        link_args.append(flag)
            else:
                # Fallback: try to detect Homebrew libomp location
                try:
                    brew_prefix = subprocess.check_output(['brew', '--prefix', 'libomp'], 
                                                         text=True).strip()
                    lib_path = f"{brew_prefix}/lib"
                    link_args.append(f'-L{lib_path}')
                    link_args.append(f'-Wl,-rpath,{lib_path}')
                    print(f"[SETUP] Auto-detected libomp at: {lib_path}")
                except (subprocess.CalledProcessError, FileNotFoundError):
                    print("[SETUP] Warning: Could not auto-detect libomp location")
        elif sys.platform == 'win32':
            pass  # Windows MSVC自动链接
        else:
            link_args.append('-lgomp')
    
    # Add architecture flags for macOS universal2 builds
    if sys.platform == 'darwin':
        arch = get_mac_arch()
        if isinstance(arch, list):
            # Multiple architectures (Universal Binary)
            for a in arch:
                if a in ('x86_64', 'arm64'):
                    link_args.extend(['-arch', a])
            print(f"[SETUP] Link args for universal2: {link_args}")
        elif isinstance(arch, str) and arch in ('x86_64', 'arm64'):
            # Single architecture
            link_args.extend(['-arch', arch])
            print(f"[SETUP] Link args for {arch}: {link_args}")
    
    return link_args

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
            define_macros=[('VERSION_INFO', '"0.1.21"'),
                           ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
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
            define_macros=[('VERSION_INFO', '"0.1.21"'),
                           ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
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

        # Get link args for architecture support
        link_args = get_link_args()

        extensions = []
        for path in pyx_paths:
            extra_args = get_compile_args_for_file(path)
            extension = Extension(
                name=str(Path(path).with_suffix("")).replace("/", ".").replace("\\", "."),
                sources=[path],
                include_dirs=get_dissimilarity_measures_include_dirs(),
                extra_compile_args=extra_args,
                extra_link_args=link_args,
                define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
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

# Custom install command to run post-installation setup
class InstallCommand(install):
    """Custom install command that runs post-installation setup."""
    
    def run(self):
        # Run the standard install
        super().run()
        
        # Run post-installation setup
        try:
            import subprocess
            import sys
            from pathlib import Path
            
            # Get the post-install script path
            script_path = Path(__file__).parent / "scripts" / "post_install.py"
            
            if script_path.exists():
                print("\nRunning post-installation setup...")
                subprocess.run([sys.executable, str(script_path)], check=True)
            else:
                print("Post-installation script not found, skipping...")
                
        except Exception as e:
            print(f"Post-installation setup failed: {e}")
            print("You can manually run: python -m sequenzo.openmp_setup")


class BdistWheel(bdist_wheel):
    """Custom bdist_wheel command to exclude source files from wheel."""
    
    def run(self):
        # Run the standard bdist_wheel
        super().run()
        
        # Clean up .c and .pyx files from the wheel
        print("\n[WHEEL] Cleaning up source files from wheel distribution...")
        try:
            import zipfile
            import tempfile
            import shutil
            from pathlib import Path
            
            # Find the generated wheel file
            dist_dir = Path(self.dist_dir)
            wheel_files = list(dist_dir.glob('*.whl'))
            
            if not wheel_files:
                print("[WHEEL] No wheel files found to clean")
                return
            
            for wheel_path in wheel_files:
                print(f"[WHEEL] Processing: {wheel_path.name}")
                
                # Create a temporary directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    
                    # Extract the wheel
                    with zipfile.ZipFile(wheel_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_path)
                    
                    # Find and remove .c and .pyx files
                    removed_count = 0
                    removed_files = []
                    for pattern in ['**/*.c', '**/*.pyx', '**/*.pxd']:
                        for file_to_remove in temp_path.glob(pattern):
                            # Only remove if it's a Cython-generated file
                            if '/sequenzo/' in str(file_to_remove):
                                removed_files.append(str(file_to_remove.relative_to(temp_path)))
                                file_to_remove.unlink()
                                removed_count += 1
                                print(f"[WHEEL]   Removed: {file_to_remove.relative_to(temp_path)}")
                    
                    if removed_count > 0:
                        # Update RECORD file to remove deleted entries
                        record_files = list(temp_path.glob('**/*.dist-info/RECORD'))
                        if record_files:
                            record_path = record_files[0]
                            print(f"[WHEEL] Updating RECORD file")
                            
                            # Read existing RECORD
                            with open(record_path, 'r') as f:
                                record_lines = f.readlines()
                            
                            # Normalize removed file paths for comparison (handle both / and \)
                            removed_set = set()
                            for rf in removed_files:
                                removed_set.add(rf.replace('\\', '/'))
                            
                            # Filter out removed files
                            new_record_lines = []
                            for line in record_lines:
                                if line.strip():  # Skip empty lines
                                    file_path = line.split(',')[0].replace('\\', '/')
                                    # Keep the line if it's not in removed files
                                    if not any(file_path.endswith(rf) or rf in file_path for rf in removed_set):
                                        new_record_lines.append(line)
                            
                            # Write updated RECORD
                            with open(record_path, 'w') as f:
                                f.writelines(new_record_lines)
                            
                            print(f"[WHEEL] Updated RECORD file: removed {removed_count} entries")
                        
                        # Repack the wheel
                        wheel_path.unlink()
                        with zipfile.ZipFile(wheel_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
                            for file_path in temp_path.rglob('*'):
                                if file_path.is_file():
                                    arcname = file_path.relative_to(temp_path)
                                    zip_ref.write(file_path, arcname)
                        
                        print(f"[WHEEL] Cleaned wheel: removed {removed_count} source files")
                    else:
                        print("[WHEEL] No source files found to remove")
                        
        except Exception as e:
            print(f"[WHEEL] Warning: Failed to clean source files from wheel: {e}")
            print("[WHEEL] The wheel is still usable, but may contain unnecessary source files")


# Run the actual setup process
setup(
    ext_modules=configure_cpp_extension() + configure_cython_extensions(),
    cmdclass={
        "build_ext": BuildExt,
        "install": InstallCommand,
        "bdist_wheel": BdistWheel,
    },
)