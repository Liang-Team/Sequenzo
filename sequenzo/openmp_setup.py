#!/usr/bin/env python3
"""
@Author  : Yuqi Liang 梁彧祺
@File    : openmp_setup.py
@Time    : 07/10/2025 10:42
@Desc    : 

OpenMP Setup for Apple Silicon Macs

This module provides automatic OpenMP dependency management for Apple Silicon Macs.
It ensures that libomp is available for parallel computation without requiring
manual user intervention.
"""

import sys
import os
import subprocess
import platform
import ctypes
from pathlib import Path


def check_libomp_availability():
    """
    Check if libomp is available on the system.
    
    Returns:
        bool: True if libomp is available, False otherwise
    """
    try:
        # Try to load libomp directly
        ctypes.CDLL('libomp.dylib')
        return True
    except OSError:
        pass
    
    # Try common Homebrew paths
    homebrew_paths = [
        '/opt/homebrew/lib/libomp.dylib',  # Apple Silicon
        '/usr/local/lib/libomp.dylib',     # Intel Mac
    ]
    
    for path in homebrew_paths:
        if os.path.exists(path):
            try:
                ctypes.CDLL(path)
                return True
            except OSError:
                continue
    
    return False


def check_homebrew_available():
    """
    Check if Homebrew is available on the system.
    
    Returns:
        bool: True if Homebrew is available, False otherwise
    """
    try:
        subprocess.run(['brew', '--version'], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL, 
                      check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_libomp_via_homebrew():
    """
    Install libomp via Homebrew.
    
    Returns:
        bool: True if installation successful, False otherwise
    """
    try:
        print("🔧 Installing libomp via Homebrew...")
        result = subprocess.run(['brew', 'install', 'libomp'], 
                              check=True, 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE)
        print("[>] libomp installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[>] libomp installation failed: {e}")
        return False
    except Exception as e:
        print(f"[>] Error during installation: {e}")
        return False


def setup_openmp_environment():
    """
    Set up OpenMP environment variables for Apple Silicon.
    
    Returns:
        bool: True if setup successful, False otherwise
    """
    try:
        # Get Homebrew prefix
        result = subprocess.run(['brew', '--prefix'], 
                              capture_output=True, text=True, check=True)
        homebrew_prefix = result.stdout.strip()
        
        # Set environment variables
        lib_path = f"{homebrew_prefix}/lib"
        include_path = f"{homebrew_prefix}/include"
        
        os.environ['DYLD_LIBRARY_PATH'] = f"{lib_path}:{os.environ.get('DYLD_LIBRARY_PATH', '')}"
        os.environ['LDFLAGS'] = f"-L{lib_path} {os.environ.get('LDFLAGS', '')}"
        os.environ['CPPFLAGS'] = f"-I{include_path} {os.environ.get('CPPFLAGS', '')}"
        
        print(f"[>] OpenMP environment variables set")
        print(f"   - Library path: {lib_path}")
        print(f"   - Include path: {include_path}")
        return True
        
    except Exception as e:
        print(f"[>] Failed to set environment variables: {e}")
        return False


def ensure_openmp_support():
    """
    Ensure OpenMP support is available on Apple Silicon Macs.
    This function handles the complete setup process.
    
    Returns:
        bool: True if OpenMP is available, False otherwise
    """
    # Only run on macOS
    if sys.platform != 'darwin':
        return True
    
    # Only run on Apple Silicon
    if platform.machine() != 'arm64':
        return True
    
    # Check if we're in a conda environment (don't interfere)
    if os.environ.get('CONDA_DEFAULT_ENV'):
        print("[>] Detected Conda environment, skipping OpenMP auto-setup")
        return True
    
    print("[>] Detected Apple Silicon Mac, checking OpenMP support...")
    
    # Check if libomp is already available
    if check_libomp_availability():
        print("[>] OpenMP support is available")
        return True
    
    # Check if Homebrew is available
    if not check_homebrew_available():
        print("""
[>] OpenMP Dependency Detection

On Apple Silicon Mac, Sequenzo requires OpenMP support for parallel computation.

Please run the following command to install OpenMP support:
    brew install libomp

If you don't have Homebrew installed, please visit https://brew.sh to install Homebrew first.
        """)
        return False
    
    # Check if libomp is already installed via Homebrew
    try:
        subprocess.run(['brew', 'list', 'libomp'], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL, 
                      check=True)
        print("[>] libomp is already installed via Homebrew")
        
        # Set up environment variables
        setup_openmp_environment()
        return True
    except subprocess.CalledProcessError:
        pass  # libomp not installed, continue with installation
    
    # Attempt to install libomp automatically
    if install_libomp_via_homebrew():
        # Set up environment variables after installation
        setup_openmp_environment()
        return True
    else:
        print("""
[>] Automatic OpenMP installation failed

Please manually run the following command:
    brew install libomp

After installation, please restart Python or re-import sequenzo.
        """)
        return False


def get_openmp_status():
    """
    Get the current OpenMP status and provide helpful information.
    
    Returns:
        dict: Status information about OpenMP support
    """
    status = {
        'platform': sys.platform,
        'architecture': platform.machine(),
        'is_apple_silicon': sys.platform == 'darwin' and platform.machine() == 'arm64',
        'libomp_available': check_libomp_availability(),
        'homebrew_available': check_homebrew_available(),
        'conda_environment': bool(os.environ.get('CONDA_DEFAULT_ENV')),
    }
    
    return status


if __name__ == "__main__":
    # Run the setup when called directly
    success = ensure_openmp_support()
    if success:
        print("[>] OpenMP support is ready!")
    else:
        print("[>] OpenMP support unavailable, will use serial computation")
    
    # Print status information
    status = get_openmp_status()
    print(f"\n[>] System Status:")
    print(f"   - Platform: {status['platform']}")
    print(f"   - Architecture: {status['architecture']}")
    print(f"   - Apple Silicon: {status['is_apple_silicon']}")
    print(f"   - libomp available: {status['libomp_available']}")
    print(f"   - Homebrew available: {status['homebrew_available']}")
    print(f"   - Conda environment: {status['conda_environment']}")
