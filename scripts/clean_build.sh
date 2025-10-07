#!/bin/bash
# Clean Build Artifacts for Sequenzo (macOS/Linux)
# Usage: bash scripts/clean_build.sh

echo "════════════════════════════════════════════════════════════════════"
echo "           Sequenzo Build Artifacts Cleaner (Unix/macOS)            "
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "This script will remove all compiled extensions and build artifacts"
echo "that might cause import errors after upgrading dependencies."
echo ""

# Get the project root (parent directory of scripts/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "Project root: $PROJECT_ROOT"
echo ""

# Ask for confirmation
read -p "Do you want to proceed? [Y/n]: " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
    echo "[CLEAN] Cancelled by user."
    exit 0
fi

cd "$PROJECT_ROOT" || exit 1

echo ""
echo "[CLEAN] Removing build directories..."

# Remove top-level build directories
rm -rf build/
rm -rf dist/
rm -rf .eggs/
rm -rf *.egg-info

echo "[CLEAN] Removing compiled extensions..."

# Remove compiled extensions
find . -name "*.so" -type f -delete
find . -name "*.pyd" -type f -delete
find . -name "*.pyc" -type f -delete
find . -name "*.pyo" -type f -delete

# Remove Cython-generated C files (only in utils/ directories)
find ./sequenzo -path "*/utils/*.c" -type f -delete

echo "[CLEAN] Removing cache directories..."

# Remove cache directories
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "[CLEAN] ✓ Cleanup complete!"
echo ""
echo "Next steps:"
echo "  1. Reinstall Sequenzo:"
echo "     pip install -e .  (for development)"
echo "     pip install --no-cache-dir sequenzo  (from PyPI)"
echo ""
echo "  2. Verify installation:"
echo "     python -c 'import sequenzo; print(sequenzo.__version__)'"
echo ""

