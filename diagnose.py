#!/usr/bin/env python
"""
Sequenzo Installation Diagnostic Tool

This script helps diagnose common installation and import issues.
Run this before reporting a bug to collect system information.

Usage:
    python diagnose.py
"""

import sys
import os
import platform
import subprocess


def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_section(title):
    """Print a section title"""
    print(f"\n📋 {title}")
    print("-" * 70)


def run_command(cmd, silent_fail=False):
    """Run a command and return output"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None if silent_fail else "Error running command"


def check_python_environment():
    """Check Python version and environment"""
    print_section("Python Environment")
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    
    # Check if in virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    print(f"Virtual environment: {'Yes (OK)' if in_venv else 'No (X) - Warning: You should use a virtual environment!'}")
    
    # Check for conda
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env:
        print(f"Conda environment: {conda_env}")
    
    # Check pip version
    pip_version = run_command(f"{sys.executable} -m pip --version")
    if pip_version:
        print(f"Pip version: {pip_version}")


def check_sequenzo_installation():
    """Check if Sequenzo is installed and importable"""
    print_section("Sequenzo Installation")
    
    # Check if sequenzo is installed
    pip_list = run_command(f"{sys.executable} -m pip list")
    if pip_list and 'sequenzo' in pip_list.lower():
        # Extract version
        for line in pip_list.split('\n'):
            if 'sequenzo' in line.lower():
                print(f"Installed package: {line}")
                break
    else:
        print("[X] Sequenzo is NOT installed")
        print("\nTo install: pip install sequenzo")
        return False
    
    # Try to import sequenzo
    print("\nTrying to import sequenzo...")
    try:
        import sequenzo
        print(f"OK - Import successful!")
        
        # Try to get version
        try:
            version = sequenzo.__version__
            print(f"  Version: {version}")
        except AttributeError:
            # Version might not be available in development installs
            print(f"  Version: (development install)")
        
        print(f"  Location: {sequenzo.__file__}")
        return True
    except ImportError as e:
        print(f"[X] Import FAILED with error:")
        print(f"  {type(e).__name__}: {e}")
        return False
    except Exception as e:
        print(f"[X] Unexpected error during import:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_dependencies():
    """Check critical dependencies"""
    print_section("Critical Dependencies")
    
    dependencies = ['numpy', 'pandas', 'pybind11', 'cython']
    numpy_version_issue = False
    
    for dep in dependencies:
        try:
            module = __import__(dep)
            version = getattr(module, '__version__', 'unknown')
            
            # Special check for numpy version
            if dep == 'numpy' and version != 'unknown':
                major_version = int(version.split('.')[0])
                if major_version < 2:
                    print(f"[X] {dep}: {version} (TOO OLD - Need 2.x)")
                    numpy_version_issue = True
                else:
                    print(f"OK - {dep}: {version}")
            else:
                print(f"OK - {dep}: {version}")
        except ImportError:
            print(f"[X] {dep}: NOT FOUND")
        except Exception as e:
            print(f"[!] {dep}: Error checking ({e})")
    
    if numpy_version_issue:
        print("\n[!] WARNING: NumPy 1.x detected!")
        print("   Sequenzo requires NumPy 2.x for compatibility.")
        print("   This is likely the cause of your import errors.")
    
    return not numpy_version_issue


def check_compilers():
    """Check for available compilers (informational)"""
    print_section("Compiler Information (for reference)")
    
    if sys.platform == 'darwin':
        clang = run_command('clang --version', silent_fail=True)
        if clang:
            print(f"Clang: {clang.split()[0] if clang else 'Not found'}")
        
        # Check for Homebrew
        brew = run_command('brew --version', silent_fail=True)
        if brew:
            print(f"Homebrew: Installed (OK)")
            
            # Check for libomp on Apple Silicon
            if platform.machine() == 'arm64':
                libomp = run_command('brew list libomp', silent_fail=True)
                print(f"libomp (OpenMP): {'Installed (OK)' if libomp else 'Not installed'}")
    
    elif sys.platform == 'win32':
        msvc = run_command('cl', silent_fail=True)
        print(f"MSVC: {'Available (OK)' if msvc else 'Not detected'}")
    
    else:  # Linux
        gcc = run_command('gcc --version', silent_fail=True)
        if gcc:
            print(f"GCC: {gcc.split()[0] if gcc else 'Not found'}")


def provide_recommendations(sequenzo_works, numpy_ok):
    """Provide recommendations based on findings"""
    print_header("Recommendations")
    
    if sequenzo_works:
        print("\n✅ Sequenzo is working correctly!")
        print("\nIf you're experiencing issues in specific scenarios,")
        print("please report them at: https://github.com/Liang-Team/Sequenzo/issues")
    else:
        print("\n[X] Sequenzo is not working. Try these fixes:\n")
        
        if not numpy_ok:
            print("Fix 1: Upgrade NumPy to 2.x (RECOMMENDED - likely fixes your issue)")
            print("  " + sys.executable + " -m pip install --upgrade 'numpy>=2.0.0'")
            print("  " + sys.executable + " -m pip uninstall sequenzo -y")
            print("  " + sys.executable + " -m pip install --no-cache-dir sequenzo")
            print("\n  Why? Sequenzo wheels are compiled with NumPy 2.x.")
            print("  NumPy 1.x is incompatible due to ABI changes.")
        
        print("\nFix 2: Clean reinstall" + (" (if NumPy upgrade doesn't work)" if not numpy_ok else ""))
        print("  " + sys.executable + " -m pip uninstall sequenzo -y")
        print("  " + sys.executable + " -m pip cache purge")
        print("  " + sys.executable + " -m pip install --no-cache-dir sequenzo")
        
        print("\nFix 3: Create a fresh virtual environment with NumPy 2.x")
        if os.environ.get('CONDA_DEFAULT_ENV'):
            print("  conda create -n sequenzo_fresh python=3.11 -y")
            print("  conda activate sequenzo_fresh")
            print("  pip install 'numpy>=2.0.0'  # Install NumPy 2.x first")
            print("  pip install sequenzo")
        else:
            print("  python -m venv sequenzo_env")
            print("  source sequenzo_env/bin/activate  # On Windows: sequenzo_env\\Scripts\\activate")
            print("  pip install 'numpy>=2.0.0'")
            print("  pip install sequenzo")
        
        print("\nFor more help, see:")
        print("  - QUICK_FIX.md")
        print("  - TROUBLESHOOTING.md")
        print("  - docs/WHY_IMPORT_FAILS.md (explains NumPy compatibility)")
        print("  - https://github.com/Liang-Team/Sequenzo/issues")


def save_diagnostic_report():
    """Offer to save diagnostic information to a file"""
    print("\n" + "=" * 70)
    response = input("\nSave diagnostic report to file? [y/N]: ").strip().lower()
    if response in ('y', 'yes'):
        filename = "sequenzo_diagnostic.txt"
        try:
            # Re-run diagnostics and capture to file
            import io
            from contextlib import redirect_stdout
            
            with open(filename, 'w') as f:
                with redirect_stdout(f):
                    print_header("Sequenzo Diagnostic Report")
                    check_python_environment()
                    check_sequenzo_installation()
                    check_dependencies()
                    check_compilers()
            
            print(f"OK - Diagnostic report saved to: {filename}")
            print("  You can share this file when reporting issues on GitHub.")
        except Exception as e:
            print(f"[X] Failed to save report: {e}")


def main():
    """Main diagnostic routine"""
    print_header("Sequenzo Installation Diagnostics")
    print("\nThis tool will check your Sequenzo installation and help diagnose issues.")
    
    check_python_environment()
    sequenzo_works = check_sequenzo_installation()
    numpy_ok = check_dependencies()
    check_compilers()
    provide_recommendations(sequenzo_works, numpy_ok)
    save_diagnostic_report()
    
    print("\n" + "=" * 70)
    print("Diagnostic complete!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDiagnostic cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nUnexpected error during diagnostic: {e}")
        sys.exit(1)

