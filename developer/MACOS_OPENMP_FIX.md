# macOS OpenMP 链接问题修复

## 问题描述

在 macOS 上使用 cibuildwheel 构建 wheel 时，出现 OpenMP 符号找不到的错误：

```
ImportError: dlopen(...clustering_c_code.cpython-312-darwin.so, 0x0002): 
symbol not found in flat namespace '___kmpc_barrier'
```

`___kmpc_barrier` 是 OpenMP 运行时库（libomp）的符号，这个错误表明：
- ✅ 编译时启用了 OpenMP（`-fopenmp`）
- ❌ 运行时无法找到 OpenMP 动态库（libomp.dylib）

## 根本原因

1. **编译环境配置不完整**：cibuildwheel 的 macOS 环境中缺少 `LDFLAGS` 和 `CPPFLAGS`
2. **缺少 rpath 设置**：编译的 `.so` 文件没有正确设置运行时库搜索路径
3. **依赖打包缺失**：wheel 中没有包含 libomp 动态库

## 解决方案

### 1. 设置完整的编译环境变量

在 `CIBW_ENVIRONMENT_MACOS` 中添加：

```yaml
CIBW_ENVIRONMENT_MACOS: >
  SEQUENZO_ENABLE_OPENMP=1
  LDFLAGS="-L$(brew --prefix libomp)/lib -Wl,-rpath,$(brew --prefix libomp)/lib"
  CPPFLAGS="-I$(brew --prefix libomp)/include"
  DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib:$DYLD_LIBRARY_PATH"
```

**关键点**：
- `LDFLAGS`: 包含 libomp 库路径 + rpath 设置
  - `-L$(brew --prefix libomp)/lib`: 告诉链接器在哪里找 libomp
  - `-Wl,-rpath,$(brew --prefix libomp)/lib`: 设置运行时库搜索路径
- `CPPFLAGS`: 包含 libomp 头文件路径
- `DYLD_LIBRARY_PATH`: 运行时动态库搜索路径

### 2. 验证 libomp 安装

在 `CIBW_BEFORE_BUILD_MACOS` 中添加验证逻辑：

```yaml
CIBW_BEFORE_BUILD_MACOS: |
  echo "Installing build dependencies for macOS"
  python -m pip install --upgrade pip
  CURRENT_ARCH=$(uname -m)
  echo "Current architecture: $CURRENT_ARCH"
  
  # Verify libomp is installed
  if [ -d "$(brew --prefix libomp 2>/dev/null)" ]; then
    echo "✓ libomp found at: $(brew --prefix libomp)"
  else
    echo "⚠ Warning: libomp not found, attempting to install..."
    brew install libomp || echo "Failed to install libomp"
  fi
  
  echo "libomp setup complete"
```

### 3. 使用 delocate 打包依赖

添加 `CIBW_REPAIR_WHEEL_COMMAND_MACOS` 来将 libomp 打包进 wheel：

```yaml
CIBW_REPAIR_WHEEL_COMMAND_MACOS: >
  DYLD_LIBRARY_PATH=$(brew --prefix libomp)/lib:$DYLD_LIBRARY_PATH
  delocate-wheel --require-archs {delocate_archs} -w {dest_dir} -v {wheel}
```

**delocate 的作用**：
- 扫描 wheel 中的 `.so` 文件
- 查找所有依赖的动态库（包括 libomp.dylib）
- 将依赖库复制到 wheel 内部（通常在 `.dylibs/` 目录）
- 修改 `.so` 文件的 rpath，指向 wheel 内部的库

## 技术原理

### macOS 动态库链接

在 macOS 上，动态库链接分为两个阶段：

1. **编译/链接时**：
   - 使用 `-L` 标志指定库搜索路径
   - 使用 `-l` 标志指定要链接的库（如 `-lomp`）
   - 生成的 `.so` 文件包含依赖库的引用

2. **运行时**：
   - 系统根据 `.so` 文件中的 rpath 和 install_name 查找依赖库
   - 如果找不到，就会出现 "symbol not found" 错误

### rpath 的重要性

`rpath` (runtime search path) 告诉系统在运行时去哪里查找动态库：

```bash
# 查看 .so 文件的依赖和 rpath
otool -L clustering_c_code.cpython-312-darwin.so
otool -l clustering_c_code.cpython-312-darwin.so | grep -A2 LC_RPATH
```

没有正确的 rpath，即使编译时链接了 libomp，运行时也找不到。

### delocate vs rpath

有两种方式让用户能运行 wheel：

1. **设置 rpath 指向系统位置**（如 `/opt/homebrew/lib`）
   - ❌ 要求用户机器上安装了 libomp
   - ❌ 不同用户的 libomp 路径可能不同
   
2. **使用 delocate 打包库到 wheel**（推荐）
   - ✅ wheel 自包含，无需用户安装额外依赖
   - ✅ 跨平台兼容性更好
   - ⚠ wheel 体积会增加（每个架构约 200KB）

## 验证修复

构建完成后，验证 wheel 是否正确打包了 OpenMP：

```bash
# 1. 解压 wheel
unzip -l sequenzo-*.whl

# 2. 查看是否包含 libomp
# 应该在 sequenzo.libs/ 或 .dylibs/ 目录中看到 libomp.dylib

# 3. 安装并测试
pip install sequenzo-*.whl
python -c "
import sequenzo.clustering.clustering_c_code as cc
print('✓ C++ extensions loaded successfully')
"
```

## 相关文件

- `.github/workflows/python-app.yml`: CI/CD 工作流配置
- `setup.py`: 编译配置，包含 OpenMP 检测和链接参数
- `developer/OPENMP_FIX_SUMMARY.md`: 之前的 OpenMP 修复总结

## 参考资料

- [cibuildwheel 文档](https://cibuildwheel.readthedocs.io/)
- [delocate 文档](https://github.com/matthew-brett/delocate)
- [macOS 动态库链接](https://developer.apple.com/library/archive/documentation/DeveloperTools/Conceptual/DynamicLibraries/)
- [OpenMP in macOS](https://mac.r-project.org/openmp/)

## 更新日期

2025年10月7日

