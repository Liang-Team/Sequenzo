#!/usr/bin/env python
"""
Clean Build Artifacts Script for Sequenzo

This script removes all build artifacts that might cause import errors
after upgrading Python, numpy, or other dependencies.

Usage:
    python scripts/clean_build.py
    
Or from anywhere:
    python -m scripts.clean_build
"""

import os
import shutil
import sys
from pathlib import Path


def clean_build_artifacts(project_root=None):
    """
    Remove all build artifacts from the Sequenzo project.
    
    Args:
        project_root: Path to the project root. If None, auto-detects.
    """
    if project_root is None:
        # Auto-detect project root (parent of scripts directory)
        project_root = Path(__file__).parent.parent.resolve()
    else:
        project_root = Path(project_root).resolve()
    
    print(f"[CLEAN] Cleaning build artifacts in: {project_root}")
    print("=" * 70)
    
    # Directories to remove completely
    dirs_to_remove = [
        'build',
        'dist',
        '.eggs',
        'sequenzo.egg-info',
    ]
    
    # File patterns to remove
    patterns_to_remove = [
        '**/*.so',      # Unix shared libraries
        '**/*.pyd',     # Windows shared libraries
        '**/*.pyc',     # Python bytecode
        '**/*.pyo',     # Python optimized bytecode
        '**/*.c',       # Cython-generated C files (in utils/)
    ]
    
    # Directories to remove recursively (by name)
    recursive_dirs = [
        '__pycache__',
        '.pytest_cache',
    ]
    
    total_removed = 0
    
    # Remove top-level directories
    for dir_name in dirs_to_remove:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"[CLEAN] Removing directory: {dir_path.relative_to(project_root)}")
            shutil.rmtree(dir_path)
            total_removed += 1
    
    # Remove files matching patterns
    for pattern in patterns_to_remove:
        for file_path in project_root.glob(pattern):
            # Special handling for .c files - only remove Cython-generated ones
            if pattern.endswith('.c'):
                # Only remove .c files in utils/ directories (Cython output)
                if 'utils' not in file_path.parts:
                    continue
                # Check if it has a corresponding .pyx file
                pyx_path = file_path.with_suffix('.pyx')
                if not pyx_path.exists():
                    continue
            
            if file_path.is_file():
                try:
                    print(f"[CLEAN] Removing file: {file_path.relative_to(project_root)}")
                    file_path.unlink()
                    total_removed += 1
                except Exception as e:
                    print(f"[WARN] Could not remove {file_path}: {e}")
    
    # Remove __pycache__ and similar directories recursively
    for dir_name in recursive_dirs:
        for dir_path in project_root.glob(f'**/{dir_name}'):
            if dir_path.is_dir():
                try:
                    print(f"[CLEAN] Removing directory: {dir_path.relative_to(project_root)}")
                    shutil.rmtree(dir_path)
                    total_removed += 1
                except Exception as e:
                    print(f"[WARN] Could not remove {dir_path}: {e}")
    
    print("=" * 70)
    print(f"[CLEAN] ✓ Cleanup complete! Removed {total_removed} items.")
    print()
    print("Next steps:")
    print("  1. Reinstall Sequenzo:")
    print("     pip install -e .  (for development)")
    print("     pip install --no-cache-dir sequenzo  (from PyPI)")
    print()
    print("  2. Verify installation:")
    print("     python -c 'import sequenzo; print(sequenzo.__version__)'")
    print()


def main():
    """Main entry point for the script."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                Sequenzo Build Artifacts Cleaner                      ║
╚══════════════════════════════════════════════════════════════════════╝

This script will remove all compiled extensions and build artifacts
that might cause import errors after upgrading dependencies.

""")
    
    # Ask for confirmation
    response = input("Do you want to proceed? [Y/n]: ").strip().lower()
    if response and response not in ('y', 'yes'):
        print("[CLEAN] Cancelled by user.")
        sys.exit(0)
    
    try:
        clean_build_artifacts()
    except Exception as e:
        print(f"\n[ERROR] Failed to clean build artifacts: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

