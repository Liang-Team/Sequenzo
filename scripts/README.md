# Sequenzo Maintenance Scripts

This directory contains utility scripts for maintaining and troubleshooting Sequenzo installations.

## Available Scripts

### 1. `clean_build.py` (Recommended)
**Cross-platform Python script for cleaning build artifacts**

**Usage:**
```bash
python scripts/clean_build.py
```

**What it does:**
- Removes `build/`, `dist/`, `.eggs/` directories
- Removes all compiled extensions (`.so`, `.pyd` files)
- Removes Python bytecode files (`.pyc`, `.pyo`)
- Removes cache directories (`__pycache__`, `.pytest_cache`)
- Removes Cython-generated C files in `utils/` directories

**When to use:**
- After upgrading Python versions
- After upgrading numpy or other dependencies
- When you see import errors like `numpy.core.multiarray failed to import`
- Before switching between development branches
- When GitHub Actions tests pass but local imports fail

---

### 2. `clean_build.sh` (Unix/macOS)
**Shell script for Unix-based systems**

**Usage:**
```bash
bash scripts/clean_build.sh
# Or make it executable and run directly:
chmod +x scripts/clean_build.sh
./scripts/clean_build.sh
```

Does the same cleanup as `clean_build.py` but optimized for Unix shells.

---

### 3. `clean_build.bat` (Windows)
**Batch script for Windows**

**Usage:**
```cmd
scripts\clean_build.bat
```

Does the same cleanup as `clean_build.py` but optimized for Windows command prompt.

---

### 4. `post_install.py`
**Automatic post-installation configuration**

This script runs automatically after `pip install sequenzo` to:
- Check for OpenMP support on macOS
- Configure compiler paths if needed
- Display helpful setup messages

You normally don't need to run this manually, but if needed:
```bash
python scripts/post_install.py
```

---

## Common Scenarios

### Scenario 1: Import errors after upgrading packages

**Problem:**
```
ImportError: numpy.core.multiarray failed to import
```

**Solution:**
```bash
python scripts/clean_build.py
pip uninstall sequenzo -y
pip install sequenzo
```

---

### Scenario 2: Different behavior between local and CI

**Problem:**
- GitHub Actions tests pass ✅
- Local imports fail ❌

**Reason:**  
CI runs in a clean environment every time, but your local environment has cached build artifacts compiled with old dependencies.

**Solution:**
```bash
python scripts/clean_build.py
pip install -e . --no-build-isolation
```

---

### Scenario 3: Switching Python versions

**Before switching:**
```bash
python scripts/clean_build.py
conda deactivate  # or deactivate your current env
```

**After switching:**
```bash
conda activate new_env  # activate new Python environment
pip install sequenzo
```

---

### Scenario 4: Development workflow

When working on Sequenzo source code:

```bash
# Make changes to code
git pull origin main  # or switch branches

# Clean old builds
python scripts/clean_build.py

# Rebuild and reinstall
pip install -e . --no-build-isolation

# Test changes
python -c "import sequenzo; print(sequenzo.__version__)"
```

---

## Why do build artifacts cause problems?

Python C extensions (`.so` on Unix, `.pyd` on Windows) are compiled binary files that:

1. **Link against specific library versions** (numpy, Python runtime)
2. **Use architecture-specific CPU instructions**
3. **Expect specific ABI (Application Binary Interface) versions**

When you upgrade Python or dependencies, these binaries become incompatible but remain in your `build/` directory. Python tries to use them, resulting in import errors.

**The fix:** Delete old binaries and rebuild them with current dependencies.

---

## For More Help

See the main [TROUBLESHOOTING.md](../TROUBLESHOOTING.md) guide for detailed troubleshooting steps and solutions to common problems.

If you still have issues:
1. Clean your environment using the scripts above
2. Collect diagnostic information:
   ```bash
   python --version
   pip list | grep sequenzo
   python -c "import sequenzo; print(sequenzo.__file__)"
   ```
3. Report the issue: https://github.com/Liang-Team/Sequenzo/issues

---

## Quick Reference

| Task | Command |
|------|---------|
| Clean build artifacts | `python scripts/clean_build.py` |
| Full reinstall from PyPI | `pip install --no-cache-dir --force-reinstall sequenzo` |
| Development install | `pip install -e . --no-build-isolation` |
| Check installation | `python -c "import sequenzo; print(sequenzo.__version__)"` |
| See full path | `python -c "import sequenzo; print(sequenzo.__file__)"` |

