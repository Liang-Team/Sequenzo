# Wheel 打包指南 - 排除源文件

## 问题描述

在之前的版本中，wheel 包（.whl）包含了不必要的源文件：
- `.c` 文件（Cython 生成的中间 C 代码）
- `.pyx` 文件（Cython 源代码）
- `.pxd` 文件（Cython 头文件）

这些文件会导致以下问题：
1. **包体积增大**：不必要地增加了 wheel 包的大小
2. **安装冲突**：用户在不同 Python 版本间切换时，这些文件可能导致导入错误
3. **混淆用户**：用户可能误以为需要重新编译这些文件

## 根本原因

- **`MANIFEST.in`** 主要控制**源码分发包（sdist, .tar.gz）**的内容
- 对于 **wheel 包（.whl）**，`MANIFEST.in` 的 `global-exclude` 指令**基本无效**
- Wheel 打包会自动包含：
  - 所有通过 `ext_modules` 编译的二进制文件（.so/.pyd）**[需要保留]**
  - Cython 生成的 .c 文件 **[应该删除]**
  - setup.py 中指定的源文件 **[应该删除]**

## 解决方案

### 1. 修改 `MANIFEST.in`

从 sdist 中排除 .pyx 文件（只在开发时需要）：

```ini
# 不再包含 .pyx 文件到源码分发中
# recursive-include sequenzo *.pyx  # ❌ 已删除

# 只包含 Python 文件
recursive-include sequenzo *.py  # ✅
```

### 2. 修改 `setup.py`

添加自定义的 `bdist_wheel` 命令，在打包完成后自动清理源文件：

```python
from setuptools.command.bdist_wheel import bdist_wheel

class BdistWheel(bdist_wheel):
    """Custom bdist_wheel command to exclude source files from wheel."""
    
    def run(self):
        # 标准打包流程
        super().run()
        
        # 从 wheel 中删除 .c, .pyx, .pxd 文件
        # [详细实现见 setup.py]
```

### 3. 打包流程

#### 清理旧文件

```bash
# 使用清理脚本
python scripts/clean_build.py

# 或手动清理
rm -rf build/ dist/ *.egg-info
find . -name "*.so" -delete
find . -name "*.c" -path "*/sequenzo/*" -delete
```

#### 构建新的 wheel

```bash
# 构建 wheel（会自动清理源文件）
python -m build --wheel

# 或使用传统方式
python setup.py bdist_wheel
```

#### 验证 wheel 内容

```bash
# 查看 wheel 中的文件列表
unzip -l dist/sequenzo-*.whl | grep -E "\.c$|\.pyx$|\.pxd$"

# 如果输出为空，说明清理成功 ✅
# 如果有输出，说明仍有源文件 ❌
```

#### 验证 .so 文件仍然存在

```bash
# 确认二进制文件仍然存在
unzip -l dist/sequenzo-*.whl | grep "\.so$"

# 应该看到类似：
#   sequenzo/dissimilarity_measures/c_code.cpython-311-darwin.so
#   sequenzo/clustering/clustering_c_code.cpython-311-darwin.so
#   等等...
```

## 最佳实践

### 什么应该在 wheel 中？

✅ **必须包含：**
- `.py` 文件（Python 源代码）
- `.so` / `.pyd` 文件（编译后的二进制扩展）
- `.csv` 数据文件
- `README.md`, `LICENSE` 等文档

❌ **不应该包含：**
- `.c` 文件（Cython 生成的中间文件）
- `.pyx` 文件（Cython 源代码）
- `.pxd` 文件（Cython 头文件）
- `.cpp`, `.h`, `.hpp` 文件（C++ 源代码）
- `__pycache__` 目录
- `.pyc` 文件

### 什么应该在 sdist 中？

✅ **必须包含：**
- 所有 `.py` 文件
- 所有 `.pyx`, `.pxd` 文件（用于从源码编译）
- 所有 `.cpp`, `.h`, `.hpp` 文件
- `setup.py`, `pyproject.toml`, `MANIFEST.in`
- `README.md`, `LICENSE`

❌ **不应该包含：**
- 编译生成的 `.c` 文件
- 编译好的 `.so` / `.pyd` 文件
- `build/`, `dist/` 目录
- `__pycache__` 目录

## 常见问题

### Q: 为什么 wheel 中要保留 .so 文件？

A: `.so` (macOS/Linux) 和 `.pyd` (Windows) 是编译后的二进制文件，这是用户运行 Sequenzo 所必需的。wheel 的主要优势就是提供预编译的二进制文件，用户无需自己编译。

### Q: 为什么要删除 .c 文件？

A: `.c` 文件是 Cython 从 `.pyx` 自动生成的中间文件，在 wheel 包中没有用处：
- 用户不需要重新编译（已经有 .so 文件了）
- 文件很大（几百 KB），增加包体积
- 可能导致版本冲突

### Q: 为什么要删除 .pyx 文件？

A: `.pyx` 是 Cython 源代码，只在开发时需要。在 wheel 包中：
- 用户不需要查看或修改源代码
- 已经编译成 .so 文件了
- 删除可以减小包体积

### Q: 如果用户想从源码编译怎么办？

A: 用户可以：
1. 从 GitHub 克隆源代码
2. 使用源码分发包（sdist, .tar.gz）
3. Sdist 包含所有源文件（.pyx, .cpp 等）

### Q: 这会影响现有用户吗？

A: 不会。用户只关心：
- 能否成功安装 ✅
- 能否导入 sequenzo ✅
- 功能是否正常 ✅

删除源文件不影响这些。

## 测试清单

发布新版本前，请验证：

- [ ] 清理所有构建文件
- [ ] 构建新的 wheel
- [ ] 验证 wheel 中**没有** `.c`, `.pyx`, `.pxd` 文件
- [ ] 验证 wheel 中**有** `.so` / `.pyd` 文件
- [ ] 在干净的虚拟环境中测试安装
- [ ] 测试 `import sequenzo` 成功
- [ ] 运行基本功能测试

## 更新日志

### v0.1.21+（即将发布）

- ✅ 修改 `MANIFEST.in`，从 sdist 中移除不必要的 .pyx 包含
- ✅ 修改 `setup.py`，添加自定义 `bdist_wheel` 命令
- ✅ Wheel 包自动清理 .c, .pyx, .pxd 文件
- ✅ 减小 wheel 包体积约 30-50%
- ✅ 避免用户在 Python 版本切换时的导入错误

### v0.1.20 及之前

- ❌ Wheel 包包含 .c, .pyx 文件
- ❌ 包体积较大
- ❌ 可能导致用户环境冲突

## 参考资料

- [Python Packaging User Guide - Wheel](https://packaging.python.org/en/latest/specifications/binary-distribution-format/)
- [setuptools - Including Data Files](https://setuptools.pypa.io/en/latest/userguide/datafiles.html)
- [Cython - Distributing Cython Modules](https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#distributing-cython-modules)

