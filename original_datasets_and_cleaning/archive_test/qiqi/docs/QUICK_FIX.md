# Quick Fix Guide - Sequenzo Import Errors

## 😱 Getting Import Errors?

```python
ImportError: numpy.core.multiarray failed to import
ImportError: dynamic module does not define module export function
```

**Don't panic!** This is a common issue that's easy to fix.

---

## 🔍 First: Check Your NumPy Version

Most import errors are caused by NumPy version mismatch. Run this first:

```bash
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
```

- **If you see `1.x.x`** (like 1.22.4, 1.26.4) → This is the problem!
- **If you see `2.x.x`** (like 2.1.1, 2.3.3) → Skip to "Other Issues" below

---

## ✅ Quick Fix (Works 90% of the time)

Copy and paste these commands into your terminal:

### If you have NumPy 1.x (most common case):

```bash
# Upgrade numpy to 2.x
pip install --upgrade "numpy>=2.0.0"

# Reinstall sequenzo
pip uninstall sequenzo -y
pip cache purge
pip install --no-cache-dir sequenzo
```

### If you already have NumPy 2.x but still getting errors:

**First, verify NumPy is actually working:**

```bash
# Test if NumPy itself works
python -c "import numpy; print(f'NumPy {numpy.__version__} OK')"
```

- **If this fails** → Your NumPy installation is corrupted, see "NumPy Installation Corrupted" below
- **If this works** → Continue to reinstall sequenzo:

```bash
pip uninstall sequenzo -y
pip cache purge
pip install --no-cache-dir sequenzo
```

### If the above doesn't work, create a fresh environment:

```bash
# Create a brand new environment
conda create -n sequenzo_new python=3.11 -y
conda activate sequenzo_new

# Install sequenzo
pip install sequenzo

# Test it
python -c "import sequenzo; print('✓ Success!')"
```

---

## 🔍 Why does this happen?

**Main Reason: NumPy 2.0 ABI Incompatibility**

Sequenzo wheels are compiled with NumPy 2.x, but your environment might have NumPy 1.x. NumPy 2.0 (released 2024) introduced breaking ABI changes:
- Code compiled with NumPy 2.x **cannot** run with NumPy 1.x
- → You get `numpy.core.multiarray failed to import`

**How did you get NumPy 1.x?**
- You installed pandas/scipy first, which installed NumPy 1.x
- You're using a conda environment that came with NumPy 1.x
- You created a requirements.txt with an old NumPy version

**The fix:** Upgrade to NumPy 2.x and reinstall sequenzo.

---

## 🔧 Still Failing After Upgrading NumPy?

### Problem: NumPy Installation Corrupted

Even after upgrading to NumPy 2.x, if you still see the error, NumPy itself might be corrupted.

**Test if NumPy works:**
```bash
python -c "import numpy; print(numpy.__version__)"
```

If this **fails**, your NumPy is broken. Fix it:

```bash
# 1. Completely uninstall everything
pip uninstall numpy sequenzo -y

# 2. Clear all caches
pip cache purge

# 3. If using conda, also remove from conda
conda uninstall numpy --force -y  # Only if you're in a conda env

# 4. Reinstall NumPy fresh
pip install --no-cache-dir "numpy>=2.0.0"

# 5. Verify NumPy works
python -c "import numpy; print(f'NumPy {numpy.__version__} works!')"

# 6. Install sequenzo
pip install --no-cache-dir sequenzo
```

### Problem: Multiple NumPy Versions Conflict

You might have multiple NumPy installations confusing Python.

**Check:**
```bash
pip list | grep -i numpy
# Should see only ONE line with numpy
```

**If you see multiple lines or mixed conda/pip:**

```bash
# Nuclear option: Create a fresh environment
conda create -n sequenzo_fresh python=3.11 -y
conda activate sequenzo_fresh

# Install ONLY with pip (don't mix conda install)
pip install "numpy>=2.0.0"
pip install sequenzo

# Test
python -c "import sequenzo; print('Success!')"
```

### Problem: Conda/Pip Mixing Issues

**Check which Python you're using:**
```bash
which python  # macOS/Linux
where python  # Windows
```

Should point to your conda environment, not system Python.

**If it points to system Python or wrong env:**
```bash
# Make sure you're in the right environment
conda activate your_environment_name

# Then reinstall
pip uninstall numpy sequenzo -y
pip install "numpy>=2.0.0" sequenzo
```

---

## 🪟 On Windows?

Use these commands instead:

```cmd
pip uninstall sequenzo -y
pip cache purge
pip install --no-cache-dir sequenzo
```

Or create a fresh environment:

```cmd
conda create -n sequenzo_new python=3.11 -y
conda activate sequenzo_new
pip install sequenzo
python -c "import sequenzo; print('✓ Success!')"
```

---

## 💻 Which Python am I using?

Run this to check:

```bash
python --version
which python    # macOS/Linux
where python    # Windows
```

Make sure it points to your conda environment (should see something like `/anaconda3/envs/yourenv/bin/python`).

If it doesn't, activate your environment first:

```bash
conda activate your_environment_name
```

---

## 🧪 How to test if it's fixed?

Run this in Python:

```python
import sequenzo
print(sequenzo.__version__)
print("✓ Everything works!")
```

---

## 🚨 Still not working?

1. **Check if you're in the right environment:**
   ```bash
   conda env list
   # The active environment has a * next to it
   ```

2. **Make sure you're using the right pip:**
   ```bash
   which pip     # macOS/Linux
   where pip     # Windows
   # Should match your Python path
   ```

3. **Try the nuclear option (fresh environment):**
   ```bash
   conda create -n sequenzo_clean python=3.11 -y
   conda activate sequenzo_clean
   pip install sequenzo
   ```

4. **Still stuck?** See our detailed [TROUBLESHOOTING.md](TROUBLESHOOTING.md) guide.

5. **Need help?** Report an issue: https://github.com/Liang-Team/Sequenzo/issues

   We respond within 24-48 hours!

---

## ✨ Pro Tips

**Before installing sequenzo:**
- Use a virtual environment (conda or venv)
- Use Python 3.9, 3.10, 3.11, or 3.12
- Don't use `sudo pip install` (never!)

**After upgrading Python or packages:**
- Always reinstall sequenzo with `pip install --no-cache-dir sequenzo`

**For development:**
- See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for developer-specific instructions

