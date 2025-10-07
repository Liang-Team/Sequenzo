# Python 包分发方式详解

## 核心概念：两种分发方式

### Wheel (.whl) - 给普通用户

**目标用户：** 99% 的普通用户，只想用你的包，不想编译

**工作流程：**

```
开发者机器（你）：
.pyx 源文件 
  ↓ (Cython)
.c 中间文件
  ↓ (C 编译器)
.so/.pyd 二进制文件
  ↓ (打包成 wheel)
sequenzo-0.1.21-cp311-cp311-macosx_10_9_universal2.whl

用户机器：
pip install sequenzo
  ↓ (直接解压 wheel)
得到编译好的 .so 文件
  ↓
import sequenzo  ✅ 成功！无需编译
```

**Wheel 包含：**
```
sequenzo/
├── __init__.py                           ✅ Python 源码
├── dissimilarity_measures/
│   ├── __init__.py                       ✅ Python 源码
│   ├── c_code.cpython-311-darwin.so      ✅ 编译好的二进制（从 C++ 编译）
│   └── utils/
│       ├── get_sm_trate_substitution_cost_matrix.cpython-311-darwin.so  ✅ 编译好的（从 .pyx 编译）
│       ├── get_sm_trate_substitution_cost_matrix.pyx  ❌ 不需要！
│       └── get_sm_trate_substitution_cost_matrix.c    ❌ 不需要！
└── clustering/
    └── clustering_c_code.cpython-311-darwin.so  ✅ 编译好的二进制
```

用户**不需要**：
- Cython
- C/C++ 编译器
- 任何构建工具

用户**只需要**：
- Python 3.11（匹配 wheel 版本）
- `pip install` 即可

---

### Source Distribution (.tar.gz) - 给开发者或特殊用户

**目标用户：**
- 想修改源码的开发者
- 特殊平台没有预编译 wheel 的用户
- 想从源码安装的用户

**工作流程：**

```
开发者机器（你）：
打包源码（不包含 .c, .so）
  ↓
sequenzo-0.1.21.tar.gz

用户机器：
pip install sequenzo --no-binary :all:
  ↓ (解压 tar.gz)
得到 .pyx, .cpp 源文件
  ↓ (在用户机器上编译)
.pyx → .c → .so
  ↓
import sequenzo  ✅ 成功！（需要编译环境）
```

**Sdist 包含：**
```
sequenzo/
├── __init__.py                           ✅ Python 源码
├── dissimilarity_measures/
│   ├── src/
│   │   ├── module.cpp                    ✅ C++ 源码
│   │   ├── module.hpp                    ✅ C++ 头文件
│   │   └── ...
│   └── utils/
│       ├── get_sm_trate_substitution_cost_matrix.pyx  ✅ Cython 源码
│       ├── get_sm_trate_substitution_cost_matrix.c    ❌ 不包含！自动生成的
│       └── get_sm_trate_substitution_cost_matrix.so   ❌ 不包含！编译产物
├── setup.py                              ✅ 构建脚本
└── pyproject.toml                        ✅ 构建配置
```

用户**需要**：
- Cython
- C/C++ 编译器（gcc/clang/MSVC）
- pybind11, numpy 等构建依赖

---

## 实际例子对比

### 场景 1：普通用户安装（使用 Wheel）

```bash
# 用户的机器
$ pip install sequenzo

# pip 自动选择合适的 wheel
Downloading sequenzo-0.1.21-cp311-cp311-macosx_10_9_universal2.whl

# 直接解压，无需编译
Installing collected packages: sequenzo
Successfully installed sequenzo-0.1.21

# 立即可用
$ python -c "import sequenzo; print('成功！')"
成功！
```

**用户得到了什么？**
- `get_sm_trate_substitution_cost_matrix.cpython-311-darwin.so` ✅ 可以直接运行的二进制文件
- **不需要** .pyx 文件
- **不需要** .c 文件
- **不需要** 编译器

---

### 场景 2：开发者或源码安装（使用 Sdist）

```bash
# 特殊平台用户，没有预编译 wheel
$ pip install sequenzo --no-binary :all:

# pip 下载源码包
Downloading sequenzo-0.1.21.tar.gz

# 在本地编译
Building wheel for sequenzo (setup.py) ... 
  Running Cython on get_sm_trate_substitution_cost_matrix.pyx
  .pyx → .c
  Compiling .c → .so
  ...
Successfully built sequenzo

# 编译后可用
$ python -c "import sequenzo; print('成功！')"
成功！
```

**用户需要什么？**
- `get_sm_trate_substitution_cost_matrix.pyx` ✅ Cython 源文件
- Cython（用于 .pyx → .c）
- C 编译器（用于 .c → .so）

---

## 为什么 Wheel 不包含 .pyx 和 .c？

### 原因 1：用户不需要

用户已经有了 `.so` 文件（编译好的二进制），就像：
- 你买了编译好的软件（.exe），不需要源代码（.c）
- 你下载了 Chrome，不需要 Chromium 源码

### 原因 2：避免混淆

如果 wheel 同时包含 `.pyx` 和 `.so`：

```python
# 用户的疑惑
sequenzo/utils/
├── my_module.pyx  # 这是什么？我要编译它吗？
├── my_module.c    # 这又是什么？
└── my_module.so   # 我应该用哪个？
```

### 原因 3：避免版本冲突

旧版本的 `.c` 文件 + 新版本的 numpy = 💥 导入错误

```
ImportError: numpy.core.multiarray failed to import
```

这正是你之前遇到的问题！

### 原因 4：减小包体积

```
get_sm_trate_substitution_cost_matrix.pyx    3 KB
get_sm_trate_substitution_cost_matrix.c    578 KB   ← 删除这个！
get_sm_trate_substitution_cost_matrix.so   145 KB   ← 保留这个
```

---

## 总结

| 文件类型 | 开发时 | Sdist (.tar.gz) | Wheel (.whl) | 用户需要吗？ |
|---------|--------|----------------|--------------|-------------|
| `.pyx`  | ✅ 需要 | ✅ 包含 | ❌ 不包含 | ❌ 不需要（有 .so 了）|
| `.c`    | ⚠️ 自动生成 | ❌ 不包含 | ❌ 不包含 | ❌ 不需要 |
| `.so`   | ⚠️ 编译产物 | ❌ 不包含 | ✅ 包含 | ✅ 需要！这是关键 |
| `.cpp`  | ✅ 需要 | ✅ 包含 | ❌ 不包含 | ❌ 不需要（有 .so 了）|
| `.py`   | ✅ 需要 | ✅ 包含 | ✅ 包含 | ✅ 需要！|

**关键理解：**
- **Wheel = 编译好的产品**，只包含 `.py` 和 `.so`
- **Sdist = 源代码**，包含 `.py`, `.pyx`, `.cpp`
- 用户从 wheel 安装时，**已经有编译好的 .so**，不需要源码
- 用户从 sdist 安装时，**需要源码**，在本地编译成 .so

## PyPI 上的最佳实践

同时上传两种包：

```bash
# 1. 构建 wheel（多个平台）
python -m build --wheel  # macOS
# → sequenzo-0.1.21-cp311-cp311-macosx_10_9_universal2.whl

# 在其他平台重复...
# → sequenzo-0.1.21-cp311-cp311-win_amd64.whl
# → sequenzo-0.1.21-cp311-cp311-manylinux_2_17_x86_64.whl

# 2. 构建 sdist
python -m build --sdist
# → sequenzo-0.1.21.tar.gz

# 3. 上传到 PyPI
twine upload dist/*
```

**用户体验：**
- macOS Python 3.11 用户 → 自动下载 macOS wheel → 无需编译 ✅
- Windows Python 3.11 用户 → 自动下载 Windows wheel → 无需编译 ✅
- Linux Python 3.11 用户 → 自动下载 Linux wheel → 无需编译 ✅
- 罕见平台用户 → 下载 sdist → 本地编译 → 需要编译器 ⚠️

这就是为什么大家都喜欢 wheel：**下载即用，无需编译！**

