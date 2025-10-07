# Sequenzo v0.1.21 Release Notes

**Release Date:** October 7, 2025

## 🐛 Critical Bug Fix

### Fixed: NumPy ABI Compatibility Issue

**Issue:** Users were experiencing `ImportError: numpy.core.multiarray failed to import` when using Sequenzo with different NumPy versions.

**Root Cause:** 
- C++ extensions were compiled with specific NumPy versions
- Binary incompatibility between compile-time and runtime NumPy versions
- Affected users on Windows, Linux, and macOS differently

**Solution:**
- Implemented `oldest-supported-numpy` build strategy
- Updated NumPy C API version to `NPY_1_7_API_VERSION` for maximum compatibility
- Modified dependency specifications to allow broader NumPy version ranges
- Updated CI/CD pipeline to ensure consistent builds

**Impact:**
- ✅ Users can now use any compatible NumPy version (1.21.0+)
- ✅ Python 3.12 users can use NumPy 2.x
- ✅ No more import errors due to NumPy version mismatches
- ✅ Improved overall package stability

## 📦 Package Changes

### Updated Dependencies

**Build-time (pyproject.toml):**
```toml
[build-system]
requires = [
    "oldest-supported-numpy; python_version < '3.12'",
    "numpy>=1.26.0; python_version >= '3.12'"
]
```

**Runtime (pyproject.toml):**
```toml
dependencies = [
    "numpy>=1.21.0",  # Broad compatibility
    ...
]
```

**Requirements Files:**
- `requirements-3.9.txt`: `numpy>=1.21.0,<2.0.0`
- `requirements-3.10.txt`: `numpy>=1.21.0,<2.0.0`
- `requirements-3.11.txt`: `numpy>=1.23.0,<2.0.0`
- `requirements-3.12.txt`: `numpy>=1.26.0` (supports 1.x and 2.x)

### Version Updates

- Package version: `0.1.20` → `0.1.21`
- Extension VERSION_INFO: `0.0.1` → `0.1.21`
- NumPy API: `NPY_1_23_API_VERSION` → `NPY_1_7_API_VERSION`

## 🔧 Technical Improvements

1. **Build System:**
   - CI/CD now uses `oldest-supported-numpy` for consistent builds
   - Updated GitHub Actions workflow for better dependency management

2. **C++ Extensions:**
   - All extensions now compiled with `NPY_1_7_API_VERSION`
   - Ensures compatibility across NumPy 1.7 - 2.x range

3. **Documentation:**
   - Added `developer/NUMPY_COMPATIBILITY.md` with technical details
   - Added `UPGRADE_TO_0.1.21.md` for user upgrade instructions

## 📊 Compatibility Matrix

| Python | NumPy Range | Tested With | Status |
|--------|-------------|-------------|--------|
| 3.9    | 1.21.0 - 1.x | 1.22.4, 1.26.4 | ✅ |
| 3.10   | 1.21.0 - 1.x | 1.22.4, 1.26.4 | ✅ |
| 3.11   | 1.23.0 - 1.x | 1.23.0, 1.26.4 | ✅ |
| 3.12   | 1.26.0 - 2.x | 1.26.0, 2.1.1  | ✅ |

## 🚀 Upgrade Instructions

### For Users with Existing Installation

If you're experiencing import errors:

```bash
pip uninstall sequenzo -y
pip cache purge
pip install --upgrade --no-cache-dir sequenzo
```

### For New Users

```bash
pip install sequenzo
```

## ✅ Verification

After installation, verify it works:

```python
import sequenzo
import numpy as np

print(f"Sequenzo: {sequenzo.__version__}")
print(f"NumPy: {np.__version__}")

# Test C++ extensions
from sequenzo.dissimilarity_measures import c_code
from sequenzo.clustering import clustering_c_code
print("✅ All extensions loaded successfully!")
```

## 📝 Migration Notes

### Breaking Changes
- **None** - This is a bug fix release with full backward compatibility

### Deprecations
- **None**

### Recommended Actions
1. Upgrade to v0.1.21 to avoid NumPy compatibility issues
2. Use flexible NumPy version ranges in your own requirements files
3. If pinning NumPy version, ensure it's within the supported range for your Python version

## 🙏 Acknowledgments

Thanks to all users who reported the NumPy import errors. This fix ensures a much better experience for everyone!

## 📚 Additional Resources

- **Upgrade Guide:** [UPGRADE_TO_0.1.21.md](UPGRADE_TO_0.1.21.md)
- **Technical Details:** [developer/NUMPY_COMPATIBILITY.md](developer/NUMPY_COMPATIBILITY.md)
- **GitHub Issues:** https://github.com/Liang-Team/Sequenzo/issues

## 🔮 Next Steps

We're committed to improving Sequenzo's stability and performance. Future updates will focus on:
- Enhanced error messages
- Better dependency management
- Performance optimizations
- Extended platform support

---

**Full Changelog:** https://github.com/Liang-Team/Sequenzo/compare/v0.1.20...v0.1.21

