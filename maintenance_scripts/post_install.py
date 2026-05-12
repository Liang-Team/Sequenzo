""" 
@Author  : Yuqi Liang 梁彧祺
@File    : post_install.py
@Time    : 07/10/2025 10:35
@Desc    : 
Post-installation script for Sequenzo

This script runs after installation to ensure all dependencies are properly set up,
especially OpenMP support on Apple Silicon Macs.
"""

import sys
import os
import subprocess
import platform
from pathlib import Path


def main():
    """Main post-installation setup function."""
    print("[>] Sequenzo post-installation setup...")
    
    # Check if we're on Apple Silicon macOS
    if sys.platform == 'darwin' and platform.machine() == 'arm64':
        print("[>] Detected Apple Silicon Mac, checking OpenMP support...")
        
        # Try to import and run the OpenMP setup
        try:
            from sequenzo.openmp_setup import ensure_openmp_support, get_openmp_status
            
            # Ensure OpenMP support
            success = ensure_openmp_support()
            
            if success:
                print("[>] OpenMP support is ready!")
            else:
                print("[>] OpenMP support unavailable, will use serial computation")
            
            # Print detailed status
            status = get_openmp_status()
            print(f"\n[>] System Status:")
            print(f"   - Platform: {status['platform']}")
            print(f"   - Architecture: {status['architecture']}")
            print(f"   - Apple Silicon: {status['is_apple_silicon']}")
            print(f"   - libomp available: {status['libomp_available']}")
            print(f"   - Homebrew available: {status['homebrew_available']}")
            print(f"   - Conda environment: {status['conda_environment']}")
            
        except ImportError as e:
            print(f"[>] Cannot import OpenMP setup module: {e}")
            print("💡 Please ensure Sequenzo is properly installed")
        except Exception as e:
            print(f"[>] Error during OpenMP setup: {e}")
    else:
        print("[>] Non-Apple Silicon system, no special OpenMP setup needed")
    
    print("\n[>] Sequenzo installation completed!")
    print("[>] View documentation: https://sequenzo.yuqi-liang.tech")
    print("[>] Test OpenMP support: python -m sequenzo.openmp_setup")


if __name__ == "__main__":
    main()
