# NumPy Compatibility Fix (v0.1.21)

## 问题背景

在 v0.1.20 及之前的版本中，用户可能遇到以下错误：

```
ImportError: numpy.core.multiarray failed to import
```

### 根本原因

这是因为 NumPy 的 C API 在不同版本间存在二进制不兼容性（ABI incompatibility）：

1. **编译时使用的 NumPy 版本** 和 **运行时安装的 NumPy 版本** 不匹配
2. C++ 扩展（.pyd/.so 文件）在编译时绑定了特定版本的 NumPy ABI
3. 如果用户安装的 NumPy 版本与编译时不同，会导致导入失败

### 具体场景

- 开发者在 macOS 用 `numpy==1.26.4` 编译 wheel
- Windows 用户安装后使用 `numpy==2.1.1`
- → 导致 C 扩展无法加载

## 解决方案

### 核心策略：oldest-supported-numpy

采用业界最佳实践：**用最老的兼容 NumPy 版本编译，允许用户使用更新的版本运行**

这是因为 NumPy 保证了 **向前兼容性**（forward compatibility）：
- 用旧版本编译的代码可以在新版本上运行 ✅
- 用新版本编译的代码不能在旧版本上运行 ❌

### 具体修改

#### 1. pyproject.toml - 编译时依赖

```toml
[build-system]
requires = [
    "setuptools>=64",
    "wheel>=0.40.0",
    "pybind11>=2.10.0",
    "cython>=0.29.21",
    # 使用 oldest-supported-numpy 确保向前兼容
    "oldest-supported-numpy; python_version < '3.12'",
    "numpy>=1.26.0; python_version >= '3.12'"
]
```

**说明**：
- `oldest-supported-numpy` 会自动选择每个 Python 版本支持的最老 NumPy 版本
- Python 3.12 特殊处理，因为它只支持 NumPy >= 1.26

#### 2. pyproject.toml - 运行时依赖

```toml
dependencies = [
    # 运行时允许广泛的版本范围
    "numpy>=1.21.0",
    ...
]
```

**说明**：
- 用户可以安装任何 >= 1.21.0 的 NumPy 版本
- 包括 NumPy 2.x（如果用户的其他依赖支持的话）

#### 3. requirements-*.txt 文件

```txt
# Python 3.9, 3.10
numpy>=1.21.0,<2.0.0

# Python 3.11
numpy>=1.23.0,<2.0.0

# Python 3.12
numpy>=1.26.0  # 支持 1.x 和 2.x
```

**说明**：
- 为每个 Python 版本指定合适的版本范围
- 避免固定版本（`==`），使用范围（`>=`,`<`）

#### 4. setup.py - NumPy API 版本

```python
define_macros=[
    ('VERSION_INFO', '"0.1.21"'),
    ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')  # 从 NPY_1_23_API_VERSION 改为 NPY_1_7_API_VERSION
]
```

**说明**：
- `NPY_1_7_API_VERSION` 是最稳定和兼容的 API 版本
- 确保编译的扩展可以在广泛的 NumPy 版本上运行

## 如何重新编译和发布

### 1. 清理旧的构建文件

```bash
# 使用脚本清理
python scripts/clean_build.py

# 或手动清理
rm -rf build/ dist/ *.egg-info
```

### 2. 安装编译依赖

```bash
# 确保安装了 oldest-supported-numpy
pip install oldest-supported-numpy
```

### 3. 构建 wheel

```bash
# 对于每个 Python 版本，在对应的环境中执行：
python -m pip install --upgrade build
python -m build --wheel

# 或使用 pip wheel
pip wheel . --no-deps -w dist/
```

### 4. 验证 wheel

```bash
# 解压 wheel 查看内容
unzip -l dist/sequenzo-0.1.21-*.whl

# 确认：
# 1. 没有 .c/.pyx 源文件（只有编译后的 .pyd/.so）
# 2. 版本号正确
```

### 5. 测试

在不同的环境中测试：

```bash
# 创建测试环境
python -m venv test_env
source test_env/bin/activate  # Windows: test_env\Scripts\activate

# 安装不同版本的 NumPy 测试
pip install numpy==1.21.0
pip install dist/sequenzo-0.1.21-*.whl
python -c "import sequenzo; print(sequenzo.__version__)"

pip install --upgrade numpy==1.26.4
python -c "import sequenzo; print('NumPy 1.26.4 works!')"

pip install --upgrade numpy==2.1.0
python -c "import sequenzo; print('NumPy 2.1.0 works!')"
```

## 用户升级指南

### 遇到 ImportError 的用户

如果用户已经安装了 v0.1.20 或更早版本，并遇到 `numpy.core.multiarray failed to import` 错误：

```bash
# 1. 完全卸载
pip uninstall sequenzo -y

# 2. 清理缓存
pip cache purge

# 3. 重新安装
pip install --upgrade --no-cache-dir sequenzo
```

### 新用户

直接安装即可：

```bash
pip install sequenzo
```

## 技术细节

### 为什么使用 NPY_1_7_API_VERSION？

- NumPy 1.7 引入了稳定的 C API
- 这个 API 版本在 NumPy 1.7 - 2.x 之间保持兼容
- 使用更新的 API 版本（如 NPY_1_23_API_VERSION）会限制兼容性

### oldest-supported-numpy 的工作原理

这个包维护了一个 Python 版本到最老兼容 NumPy 版本的映射：

```python
# 示例映射
{
    "3.9": "numpy==1.19.3",
    "3.10": "numpy==1.21.6",
    "3.11": "numpy==1.23.2",
    "3.12": "numpy==1.26.0",
}
```

### 版本范围说明

- `numpy>=1.21.0,<2.0.0`：允许 1.21 到 1.x 的最新版本，但不包括 2.x
- `numpy>=1.26.0`：允许 1.26+ 和 2.x（Python 3.12 支持）

## 参考资源

- [NumPy C API compatibility](https://numpy.org/doc/stable/reference/c-api/array.html)
- [oldest-supported-numpy package](https://github.com/scipy/oldest-supported-numpy)
- [Python package binary compatibility](https://packaging.python.org/guides/distributing-packages-using-setuptools/#binary-extensions)

## 变更历史

- **v0.1.21** (2025-10-07): 修复 NumPy 兼容性问题
  - 使用 oldest-supported-numpy 编译
  - 允许广泛的运行时 NumPy 版本
  - 更新 NPY_NO_DEPRECATED_API 为 NPY_1_7_API_VERSION

- **v0.1.20** (之前): 存在 NumPy ABI 不兼容问题

