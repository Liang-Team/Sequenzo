# 诊断：numpy.core.multiarray failed to import

## 用户情况
- ✅ 已升级 numpy 到 2.0+
- ✅ 已卸载重装 sequenzo
- ❌ 仍然出现错误：`numpy.core.multiarray failed to import`

## 可能的原因

### 原因 1: NumPy 安装损坏（最可能）

即使升级到 numpy 2.x，如果安装过程有问题，也会出现这个错误。

#### 检查方法：
```bash
# 1. 检查 numpy 是否能单独导入
python -c "import numpy; print(numpy.__version__)"

# 2. 如果上面失败，说明 numpy 本身有问题
# 如果成功，继续检查：
python -c "import numpy.core.multiarray; print('OK')"
```

#### 可能的情况：
- **情况 A**：连 numpy 都导不了 → NumPy 安装损坏
- **情况 B**：numpy 能导入，但 sequenzo 导入时报错 → sequenzo 的问题

---

### 原因 2: 多个 NumPy 版本冲突

用户可能同时安装了多个版本的 numpy。

#### 检查方法：
```bash
pip list | grep -i numpy
# 应该只有一行，如果有多行说明有冲突

# 检查 numpy 安装位置
python -c "import numpy; print(numpy.__file__)"

# 检查是否有多个 numpy
find $(python -c "import site; print(site.getsitepackages()[0])") -name "numpy*"
```

---

### 原因 3: Conda vs Pip 混用

如果用户在 conda 环境里混用 pip 和 conda 安装包，可能导致路径混乱。

#### 检查方法：
```bash
# 检查使用的是哪个 python
which python
which pip

# 检查是否在 conda 环境
echo $CONDA_DEFAULT_ENV

# 查看 numpy 是通过什么安装的
conda list numpy  # 如果在 conda 环境
pip show numpy
```

#### 症状：
- `conda list numpy` 显示一个版本
- `pip show numpy` 显示另一个版本
- Python 不知道该用哪个

---

### 原因 4: C 扩展编译时链接了错误的 NumPy

即使用户有 numpy 2.x，如果 sequenzo 的 wheel 是用不兼容的方式编译的，也会失败。

#### 这种情况下的症状：
```python
import numpy  # ✅ 成功
print(numpy.__version__)  # 2.3.3

import sequenzo  # ❌ 失败
# ImportError: numpy.core.multiarray failed to import
```

---

## 诊断步骤（让用户运行）

### 步骤 1: 基础检查
```bash
# 运行这个完整的诊断
python << 'EOF'
import sys
print(f"Python: {sys.version}")
print(f"Python executable: {sys.executable}")

try:
    import numpy
    print(f"✓ NumPy version: {numpy.__version__}")
    print(f"  NumPy location: {numpy.__file__}")
except Exception as e:
    print(f"✗ NumPy import failed: {e}")
    sys.exit(1)

try:
    import numpy.core.multiarray
    print(f"✓ numpy.core.multiarray OK")
except Exception as e:
    print(f"✗ numpy.core.multiarray failed: {e}")

try:
    import sequenzo
    print(f"✓ Sequenzo imported successfully")
except Exception as e:
    print(f"✗ Sequenzo import failed: {e}")
    import traceback
    traceback.print_exc()
EOF
```

### 步骤 2: 检查是否有多个 NumPy
```bash
# 查找所有 numpy 安装
python << 'EOF'
import sys
import os

site_packages = []
for path in sys.path:
    if 'site-packages' in path or 'dist-packages' in path:
        site_packages.append(path)
        numpy_path = os.path.join(path, 'numpy')
        if os.path.exists(numpy_path):
            print(f"Found numpy at: {numpy_path}")
            version_file = os.path.join(numpy_path, 'version.py')
            if os.path.exists(version_file):
                with open(version_file) as f:
                    for line in f:
                        if 'version' in line.lower():
                            print(f"  {line.strip()}")
                            break
EOF
```

### 步骤 3: 完全清理重装（核心解决方案）
```bash
# 1. 完全卸载 numpy 和 sequenzo
pip uninstall numpy sequenzo -y

# 2. 清理 pip 缓存
pip cache purge

# 3. 如果在 conda 环境，也从 conda 卸载
conda uninstall numpy --force -y

# 4. 清理可能的残留文件
python -c "import site; print(site.getsitepackages()[0])" | xargs -I {} find {} -name "*numpy*" -o -name "*sequenzo*" 2>/dev/null

# 5. 重新安装（按顺序）
pip install --no-cache-dir "numpy>=2.0.0"

# 6. 验证 numpy
python -c "import numpy; print(f'NumPy {numpy.__version__} OK')"

# 7. 安装 sequenzo
pip install --no-cache-dir sequenzo

# 8. 测试
python -c "import sequenzo; print('Success!')"
```

---

## 特殊情况处理

### 情况 A: macOS + Homebrew Python

如果用户用的是 Homebrew 安装的 Python，可能有权限问题。

```bash
# 检查
which python3
# 如果是 /usr/local/bin/python3 或 /opt/homebrew/bin/python3

# 解决方案：使用虚拟环境
python3 -m venv ~/sequenzo_env
source ~/sequenzo_env/bin/activate
pip install --upgrade pip
pip install "numpy>=2.0.0"
pip install sequenzo
```

### 情况 B: Windows + Anaconda

```cmd
# 创建全新环境
conda create -n sequenzo_clean python=3.11 -y
conda activate sequenzo_clean

# 不要用 conda install numpy！用 pip
pip install "numpy>=2.0.0"
pip install sequenzo
```

### 情况 C: 系统 Python（不推荐但可能遇到）

```bash
# 检查是否是系统 Python
which python
# 如果是 /usr/bin/python

# 不要在系统 Python 装包！创建虚拟环境
python3 -m venv ~/.venv/sequenzo
source ~/.venv/sequenzo/bin/activate
pip install "numpy>=2.0.0" sequenzo
```

---

## 终极解决方案：从源码安装

如果上述方法都不行，让用户从源码安装（这样会在他们的机器上重新编译）：

```bash
# 1. 克隆仓库
git clone https://github.com/Liang-Team/Sequenzo.git
cd Sequenzo

# 2. 创建干净环境
conda create -n sequenzo_build python=3.11 -y
conda activate sequenzo_build

# 3. 安装依赖
pip install "numpy>=2.0.0"
pip install -r requirements/requirements-3.11.txt

# 4. 从源码安装
pip install -e . --no-build-isolation

# 5. 测试
python -c "import sequenzo; print('Success!')"
```

这样会：
- 在用户机器上重新运行 Cython（生成新的 .c 文件）
- 用用户的 numpy 版本编译
- 避免所有预编译 wheel 的问题

---

## 需要收集的信息

请让用户提供以下信息：

```bash
# 一键收集诊断信息
python << 'EOF' > sequenzo_debug.txt
import sys
import platform
import subprocess

print("="*70)
print("System Information")
print("="*70)
print(f"OS: {platform.system()} {platform.release()}")
print(f"Architecture: {platform.machine()}")
print(f"Python: {sys.version}")
print(f"Python executable: {sys.executable}")

print("\n" + "="*70)
print("NumPy Information")
print("="*70)
try:
    import numpy
    print(f"Version: {numpy.__version__}")
    print(f"Location: {numpy.__file__}")
    print(f"Config: {numpy.__config__.show()}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*70)
print("Pip List")
print("="*70)
subprocess.run([sys.executable, "-m", "pip", "list"])

print("\n" + "="*70)
print("Environment Variables")
print("="*70)
import os
for key in ['PATH', 'PYTHONPATH', 'CONDA_DEFAULT_ENV', 'VIRTUAL_ENV']:
    print(f"{key}: {os.environ.get(key, 'Not set')}")

print("\n" + "="*70)
print("Import Test")
print("="*70)
try:
    import numpy
    print("✓ numpy")
except Exception as e:
    print(f"✗ numpy: {e}")

try:
    import numpy.core.multiarray
    print("✓ numpy.core.multiarray")
except Exception as e:
    print(f"✗ numpy.core.multiarray: {e}")

try:
    import sequenzo
    print("✓ sequenzo")
except Exception as e:
    print(f"✗ sequenzo: {e}")
    import traceback
    traceback.print_exc()
EOF

cat sequenzo_debug.txt
```

然后让用户把 `sequenzo_debug.txt` 发给你。

---

## 我的猜测

基于经验，最可能的原因是：

1. **NumPy 安装不完整**（70% 可能性）
   - 升级过程中断
   - 文件损坏
   - 解决：完全卸载重装

2. **Conda/Pip 混用**（20% 可能性）
   - conda 装了一个 numpy，pip 又装了一个
   - Python 找到了错误的版本
   - 解决：统一用 pip 或创建新环境

3. **PyPI wheel 与用户环境不兼容**（10% 可能性）
   - wheel 编译配置问题
   - 解决：从源码安装

---

## 快速测试命令（给用户）

```bash
# 最简单的测试
python -c "import numpy; import sequenzo; print('All OK')"

# 如果失败，运行详细测试
python -c "
import sys
try:
    import numpy as np
    print(f'NumPy {np.__version__} at {np.__file__}')
    arr = np.array([1,2,3])
    print(f'NumPy works: {arr.sum()}')
except Exception as e:
    print(f'NumPy error: {e}')
    sys.exit(1)

try:
    import sequenzo
    print('Sequenzo imported successfully')
except Exception as e:
    print(f'Sequenzo error: {e}')
    import traceback
    traceback.print_exc()
"
```

