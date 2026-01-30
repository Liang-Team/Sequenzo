# Troubleshooting Guide for Sequenzo Installation Issues

## Common Installation Problems and Solutions

### Problem 1: Import Errors After Installation (numpy.core.multiarray failed)

**Symptoms:**
```python
ImportError: numpy.core.multiarray failed to import
ImportError: dynamic module does not define module export function (PyInit_...)
```

**Root Cause:**  
This usually happens when there's a mismatch between the compiled binary extensions in the installed package and your current Python/numpy version. This can occur when:
- You upgraded Python (e.g., 3.10 → 3.11)
- You upgraded numpy or other dependencies
- You have conflicting packages in your environment
- Pip's cache has corrupted files

**Solution - For Regular Users (installed via `pip install sequenzo`):**

You **don't need** the Sequenzo source code. Just clean your pip cache and reinstall:

#### Step 1: Completely remove old installation

```bash
# Uninstall sequenzo
pip uninstall sequenzo -y

# Clear pip cache to remove any corrupted cached files
pip cache purge
```

#### Step 2: Reinstall from PyPI

```bash
# Reinstall with fresh download (no cache)
pip install --no-cache-dir sequenzo
```

#### Step 3: Verify installation
```python
python -c "import sequenzo; print(sequenzo.__version__)"
```

If you still see errors, try creating a fresh virtual environment:

```bash
# Create new environment
conda create -n sequenzo_fresh python=3.11 -y
conda activate sequenzo_fresh

# Install sequenzo
pip install sequenzo

# Test
python -c "import sequenzo; print('Success!')"
```

---

**Solution - For Developers (working with source code):**

If you're developing Sequenzo from the GitHub repository:

#### Step 1: Clean build artifacts

**Quick method:**
```bash
# Navigate to your Sequenzo source directory
cd /path/to/Sequenzo-main

# Use our cleaning script
python scripts/clean_build.py
```

**Manual method:**
```bash
rm -rf build/ dist/ *.egg-info
find . -name "*.so" -delete
find . -name "*.pyd" -delete
find . -name "__pycache__" -delete
pip uninstall sequenzo -y
```

#### Step 2: Rebuild and reinstall

```bash
pip install -e . --no-build-isolation
```

#### Step 3: Verify installation
```python
import sequenzo
print(sequenzo.__version__)
```

---

### Problem 2: Conda Environment Issues

**Symptoms:**
- Installation succeeds but imports fail
- Different behavior between terminal and IDE

**Solution:**

Make sure you're using the same environment everywhere:

```bash
# Check which Python you're using
which python  # macOS/Linux
where python  # Windows

# Check which pip you're using
which pip     # macOS/Linux
where pip     # Windows

# They should both point to your conda environment
# Example: /Users/yourname/anaconda3/envs/dphil/bin/python
```

If they don't match, activate your environment:
```bash
conda activate dphil  # or your environment name
```

Then reinstall:
```bash
pip uninstall sequenzo -y
pip install --no-cache-dir sequenzo
```

---

### Problem 3: Multiple Python Installations Conflict

**Symptoms:**
- `pip install` succeeds but `import sequenzo` fails
- Package shows up in `pip list` but not importable

**Solution:**

Use Python module syntax to ensure consistency:

```bash
# Instead of: pip install sequenzo
python -m pip install sequenzo

# Instead of: pip uninstall sequenzo
python -m pip uninstall sequenzo
```

This ensures you're using the pip associated with the exact Python interpreter you're running.

---

### Problem 4: OpenMP-Related Errors on macOS

**Symptoms:**
```
Library not loaded: @rpath/libomp.dylib
```

**Solution:**

Install OpenMP support via Homebrew:

```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install OpenMP
brew install libomp

# If using Apple Silicon (M1/M2/M3), also install LLVM
brew install llvm

# Then reinstall Sequenzo
pip uninstall sequenzo -y
pip install --no-cache-dir sequenzo
```

---

### Problem 5: Windows Compiler Issues

**Symptoms:**
```
error: Microsoft Visual C++ 14.0 or greater is required
```

**Solution:**

Install Microsoft C++ Build Tools:

1. Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Install "Desktop development with C++"
3. Restart your terminal/IDE
4. Reinstall Sequenzo:
   ```bash
   pip install --no-cache-dir sequenzo
   ```

**Note:** If you're using pre-built wheels from PyPI, you should NOT need a compiler. This error only appears when building from source.

---

## Prevention Tips

### For Regular Users

1. **Always use virtual environments**  
   Never install packages system-wide (don't use `sudo pip install`)

2. **Use pre-built wheels when possible**  
   ```bash
   pip install sequenzo  # This downloads pre-built wheels from PyPI
   ```

3. **Keep dependencies updated**  
   ```bash
   pip install --upgrade sequenzo
   ```

### For Developers

1. **Clean build artifacts before switching branches or Python versions**
   ```bash
   # Quick clean script (save as clean.sh or clean.bat)
   rm -rf build/ dist/ *.egg-info
   find . -name "*.so" -delete
   find . -name "__pycache__" -delete
   ```

2. **Use `pip install -e .` for development**  
   This creates an "editable install" that reflects your code changes immediately

3. **After updating numpy or other dependencies, rebuild extensions**
   ```bash
   pip uninstall sequenzo -y
   rm -rf build/
   pip install -e . --no-build-isolation
   ```

---

## Still Having Issues?

### Quick Diagnostic Commands

Run these and share the output when reporting issues:

```bash
# System information
python --version
pip --version
uname -a  # macOS/Linux
systeminfo  # Windows

# Environment check
pip list | grep -i sequenzo
pip list | grep -i numpy
python -c "import sequenzo; print(sequenzo.__file__)"

# Import test
python -c "import sequenzo; print('Success!')"
```

### Reporting Bugs

If the above solutions don't work, please:

1. **Clean your environment** (follow Step 1 above)
2. **Collect diagnostic information** (run commands above)
3. **Create an issue** on GitHub: https://github.com/Liang-Team/Sequenzo/issues
4. **Include**:
   - Your operating system and version
   - Python version (`python --version`)
   - How you installed Sequenzo (PyPI or from source)
   - Full error traceback
   - Output of diagnostic commands

We typically respond within 24-48 hours and aim to fix issues within one week.

---

## Developer Reference: Manual Build

If you need to build from source (most users don't need this):

```bash
# 1. Clone the repository
git clone https://github.com/Liang-Team/Sequenzo.git
cd Sequenzo

# 2. Install build dependencies
pip install -r requirements/requirements-3.10.txt  # Match your Python version

# 3. Clean previous builds
rm -rf build/ dist/ *.egg-info
find . -name "*.so" -delete

# 4. Build extensions in-place
python setup.py build_ext --inplace

# 5. Install in editable mode
pip install -e .
```

**Why GitHub Actions tests pass but local installation fails:**

- GitHub Actions runs in a **clean environment** every time
- Your local environment may have **cached build artifacts** from previous installations
- Solution: Clean your local environment as shown above

---

## Quick Fix Checklist

- [ ] Removed `build/` directory
- [ ] Removed `*.egg-info` directories
- [ ] Removed all `.so` or `.pyd` files
- [ ] Uninstalled sequenzo (`pip uninstall sequenzo`)
- [ ] Checked you're using the correct Python/pip (`which python`, `which pip`)
- [ ] Reinstalled with `pip install --no-cache-dir sequenzo`
- [ ] Tested import: `python -c "import sequenzo"`

If all checkboxes are checked and it still doesn't work, please report the issue on GitHub.

