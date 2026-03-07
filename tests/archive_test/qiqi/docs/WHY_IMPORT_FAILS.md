# 为什么 GitHub Actions 测试通过，但用户本地导入失败？

## 问题现象

- ✅ GitHub Actions 所有测试都通过
- ❌ 用户在 Mac/Windows 上安装后导入失败
- 错误信息：
  - `ImportError: numpy.core.multiarray failed to import`
  - `ImportError: dynamic module does not define module export function (PyInit_...)`

## 两个主要原因

### 原因 1：NumPy 2.0 ABI 不兼容（最常见）

### NumPy 1.x vs 2.x 的重大变化

NumPy 2.0 (2024年发布) 引入了 **ABI (Application Binary Interface) 破坏性更新**：

- 用 **numpy 1.x** 编译的 C 扩展 → 只能在 numpy 1.x 运行
- 用 **numpy 2.x** 编译的 C 扩展 → 可以在 numpy 2.x 运行

**反之不行！** 用 numpy 1.x 编译的扩展在 numpy 2.x 环境会报错。

### 我们的情况

**GitHub Actions 编译时：**
```yaml
pip install --only-binary=all "numpy<2.5"  # Line 80 in python-app.yml
```
这会装 numpy 2.x（比如 2.1.1），wheel 就用 numpy 2.x 编译。

**用户安装时：**
```python
# pyproject.toml line 34
dependencies = [
    "numpy>=1.19.5,<2.5",  # 允许 1.19.5 到 2.4.x
]
```

如果用户环境里已经有 numpy 1.x（比如从其他包装的），pip 不会升级它，因为 `numpy>=1.19.5,<2.5` 满足条件。

**结果：**
- Wheel 用 numpy 2.x 编译 ✅
- 用户运行环境是 numpy 1.x ❌
- → 导入失败！

## 解决方案

### 方案 1：限制 numpy 版本一致性（推荐）

**修改 pyproject.toml：**

```toml
# 旧的（有问题）
dependencies = [
    "numpy>=1.19.5,<2.5",
]

# 新的（修复）
dependencies = [
    "numpy>=2.0.0,<2.5",  # 强制 numpy 2.x
]
```

这样用户安装时会自动升级到 numpy 2.x，匹配我们编译时的版本。

**注意：** 这会强制用户升级 numpy，可能影响他们的其他代码。

### 方案 2：为不同 numpy 版本编译多个 wheel

使用 `oldest-supported-numpy` 编译策略：

```yaml
# In .github/workflows/python-app.yml
- name: Install dependencies
  run: |
    pip install oldest-supported-numpy  # 用最老的兼容版本编译
```

这样编译出的 wheel 可以在 numpy 1.x 和 2.x 都能用（向前兼容）。

### 方案 3：在 PyPI 上传多个版本的 wheel

为 numpy 1.x 和 2.x 分别编译：

```bash
# Wheel for numpy 1.x
pip install "numpy>=1.19,<2.0"
python -m build
mv dist/sequenzo-*.whl dist/sequenzo-*-np1.whl

# Wheel for numpy 2.x  
pip install "numpy>=2.0,<2.5"
python -m build
mv dist/sequenzo-*.whl dist/sequenzo-*-np2.whl
```

但这会让发布流程非常复杂。

## 推荐解决方案（综合考虑）

**短期修复（立即解决用户问题）：**

在 README 和 TROUBLESHOOTING 中明确说明：

```bash
# 如果遇到 numpy.core.multiarray 错误
pip install --upgrade "numpy>=2.0.0"
pip uninstall sequenzo -y
pip install sequenzo
```

**长期修复（下一个版本）：**

1. 修改 `pyproject.toml` 强制 numpy 2.x：
   ```toml
   dependencies = [
       "numpy>=2.0.0,<2.5",
   ]
   ```

2. 在 GitHub Actions 中明确使用相同的 numpy 版本策略

3. 在文档中说明 numpy 版本要求

## 为什么新虚拟环境也会有问题？

即使用户创建了新的虚拟环境，可能：

1. **conda 环境预装了 numpy 1.x：**
   ```bash
   conda create -n myenv python=3.11
   # conda 可能默认装了 numpy 1.x
   ```

2. **先装了其他依赖 numpy 1.x 的包：**
   ```bash
   pip install pandas  # 可能装了 numpy 1.x
   pip install sequenzo  # 不会升级 numpy，因为 1.x 满足条件
   ```

3. **pip 的依赖解析策略：**
   pip 不会主动升级已安装的包，除非明确要求。

## 检测方法

让用户运行：

```bash
python -c "import numpy; print(f'numpy version: {numpy.__version__}')"
```

- 如果是 `1.x.x` → 这就是问题所在
- 如果是 `2.x.x` → 可能是其他问题

## 快速修复命令

```bash
# 升级 numpy 到 2.x
pip install --upgrade "numpy>=2.0.0"

# 重装 sequenzo
pip uninstall sequenzo -y
pip cache purge
pip install --no-cache-dir sequenzo

# 验证
python -c "import sequenzo; print('Success!')"
```

## 预防措施

### 对于用户：

1. 使用新虚拟环境时，先装 numpy 2.x：
   ```bash
   conda create -n myenv python=3.11
   conda activate myenv
   pip install "numpy>=2.0.0"  # 先装 numpy
   pip install sequenzo
   ```

2. 定期更新依赖：
   ```bash
   pip install --upgrade sequenzo numpy
   ```

### 对于开发者：

1. 在 `pyproject.toml` 中明确版本要求
2. 在 CI/CD 中测试多个 numpy 版本
3. 文档中清楚说明兼容性要求

---

### 原因 2：Cython 生成的 .c 文件被 Git Tracked（开发者问题）

#### 问题描述

如果你在本地开发时运行了 `python setup.py build_ext --inplace`，Cython 会生成 `.c` 文件，例如：
```
sequenzo/dissimilarity_measures/utils/get_sm_trate_substitution_cost_matrix.c
```

这些文件**包含硬编码的路径**，指向你本地的 conda/venv 环境：

```c
// 生成的 .c 文件里
"include_dirs": [
    "/Users/yourname/Documents/project/.conda/lib/python3.12/site-packages/numpy/_core/include",
    ...
]
```

如果这些文件被 commit 到 git 并推送到 GitHub，用户安装时会使用这些文件，但路径是错的！

#### 症状

```python
ImportError: dynamic module does not define module export function (PyInit_get_sm_trate_substitution_cost_matrix)
```

#### 解决方案（开发者）

**1. 从 git 中删除这些文件：**

```bash
git rm --cached sequenzo/dissimilarity_measures/utils/*.c
git rm --cached sequenzo/big_data/clara/utils/*.c
git commit -m "Remove Cython-generated .c files from git"
```

**2. 更新 `.gitignore`：**

确保 `.gitignore` 中**不要**有以下行：
```gitignore
# ❌ 错误 - 不要这样做
!sequenzo/dissimilarity_measures/utils/*.c
```

应该让它们被忽略：
```gitignore
# ✅ 正确 - 让 Cython 生成的 .c 文件被忽略
*.c

# 只保留手写的 C++ 源文件
!sequenzo/dissimilarity_measures/src/*.cpp
!sequenzo/clustering/src/*.cpp
```

**3. 在 CI/CD 中自动生成：**

GitHub Actions 会在构建 wheel 时自动调用 Cython 生成新的 `.c` 文件，所以不需要 commit 它们。

#### 用户遇到这个问题怎么办？

等待下一个版本发布，或者从源码安装并清理：

```bash
git clone https://github.com/Liang-Team/Sequenzo.git
cd Sequenzo

# 删除旧的 .c 文件
find . -path "*/utils/*.c" -delete

# 重新构建
pip install -e .
```

---

## 相关链接

- NumPy 2.0 Migration Guide: https://numpy.org/devdocs/numpy_2_0_migration_guide.html
- NumPy ABI Changes: https://numpy.org/doc/stable/dev/api_changes.html
- Cython Documentation - Source Files and Compilation: https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html
- PEP 691 - Binary distribution format: https://peps.python.org/pep-0691/

