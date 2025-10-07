# macOS OpenMP Symbol Not Found 修复方案

## 问题描述

在 macOS GitHub Actions 中构建 wheel 时，测试阶段出现 OpenMP 符号找不到的错误：

```
ImportError: dlopen(...clustering_c_code.cpython-312-darwin.so, 0x0002): 
symbol not found in flat namespace '___kmpc_barrier'
```

**错误分析**：
- `___kmpc_barrier` 是 OpenMP 运行时库（libomp.dylib）中的符号
- ✅ 编译时成功启用了 OpenMP（使用了 `-fopenmp` 标志）
- ❌ 运行时无法找到 OpenMP 动态库

## 根本原因

1. **硬编码的 rpath**：setup.py 和 workflow 中设置了 `-Wl,-rpath,$(brew --prefix libomp)/lib`，这会将 Homebrew 的绝对路径写入 .so 文件。用户机器上如果没有相同路径的 libomp，就会失败。

2. **delocate 未能正确打包**：即使 delocate-wheel 运行了，但由于 rpath 已经被硬编码，delocate 可能无法正确识别需要打包的库，或者打包后没有更新 rpath。

3. **测试环境缺少 OpenMP**：cibuildwheel 的测试环境是隔离的，即使构建机器上有 libomp，测试环境可能找不到。

## 解决方案

### 1. 移除硬编码的 rpath（setup.py）

**修改前**：
```python
link_args.append(f'-L{lib_path}')
link_args.append(f'-Wl,-rpath,{lib_path}')  # ❌ 硬编码路径
```

**修改后**：
```python
link_args.append(f'-L{lib_path}')
# 不设置 rpath，让 delocate-wheel 来处理
```

**原理**：
- 编译时只告诉链接器在哪里找到 libomp（`-L`）
- 运行时的路径由 delocate-wheel 设置为相对路径（`@loader_path/.dylibs/`）

### 2. 简化 workflow 中的 LDFLAGS

**修改前**：
```yaml
LDFLAGS="-L$(brew --prefix libomp)/lib -Wl,-rpath,$(brew --prefix libomp)/lib"
```

**修改后**：
```yaml
LDFLAGS="-L$(brew --prefix libomp)/lib"
# 移除 -Wl,-rpath，让 delocate 处理
```

### 3. 增强 delocate-wheel 诊断

添加了详细的诊断信息来追踪 delocate 的执行：

```yaml
CIBW_REPAIR_WHEEL_COMMAND_MACOS: |
  # 1. 检查 libomp 是否存在
  echo "=== Inspecting wheel .so files before repair ==="
  # 使用 otool -L 查看 .so 文件的依赖
  
  # 2. 设置环境变量帮助 delocate 找到 libomp
  export DYLD_LIBRARY_PATH="$LIBOMP_PATH:$DYLD_LIBRARY_PATH"
  
  # 3. 运行 delocate-wheel（带 -L 指定库路径）
  delocate-wheel --require-archs {delocate_archs} -L "$LIBOMP_PATH" -w {dest_dir} -v {wheel}
  
  # 4. 验证打包结果
  echo "=== Inspecting repaired .so files ==="
  # 检查 .dylibs 目录是否包含 libomp
  # 使用 otool -L 确认 rpath 已更新
```

**关键改进**：
- 使用 `-L "$LIBOMP_PATH"` 明确告诉 delocate 在哪里找 libomp
- 设置 `DYLD_LIBRARY_PATH` 确保 delocate 自身能找到库
- 添加前后对比诊断，便于调试

### 4. 为测试添加 fallback 支持

如果 delocate 打包失败，测试仍需能够运行（至少在 CI 环境中）：

```yaml
CIBW_TEST_COMMAND_MACOS: |
  # 设置 DYLD_LIBRARY_PATH 作为 fallback
  if [ -d "$(brew --prefix libomp 2>/dev/null)/lib" ]; then
    export DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib:$DYLD_LIBRARY_PATH"
    echo "DYLD_LIBRARY_PATH for test: $DYLD_LIBRARY_PATH"
  fi
  python - <<'PY'
  # ... 测试代码 ...
  PY
```

**注意**：这只是 fallback，正确的做法还是让 delocate 打包好库。

## 技术原理

### macOS 动态库链接机制

1. **编译时**：
   ```bash
   clang++ -L/path/to/lib -lomp mycode.cpp -o mycode.so
   ```
   - `-L`: 告诉链接器在哪里找 libomp.dylib
   - `-lomp`: 链接 libomp 库
   - 生成的 .so 文件会包含对 libomp.dylib 的引用

2. **运行时**（未使用 delocate）：
   - 系统根据 .so 文件中的 install_name 查找 libomp.dylib
   - 如果路径不存在 → 报错 "symbol not found"

3. **使用 delocate 后**：
   - delocate 扫描 .so 文件的依赖
   - 将 libomp.dylib 复制到 wheel 的 `.dylibs/` 目录
   - 修改 .so 文件的引用：`libomp.dylib` → `@loader_path/.dylibs/libomp.dylib`
   - 运行时系统会在 .so 文件同级目录下的 `.dylibs/` 找到库

### 为什么不硬编码 rpath

**硬编码的问题**：
```python
# ❌ 不推荐
link_args.append('-Wl,-rpath,/opt/homebrew/lib')
```

1. 不同用户的 Homebrew 路径不同（Intel vs Apple Silicon）
2. 用户可能没有安装 libomp
3. delocate 可能无法正确处理已经硬编码的 rpath

**正确做法**：
```python
# ✅ 推荐
link_args.append('-L/opt/homebrew/lib')  # 编译时
# 让 delocate 设置 rpath 为 @loader_path/.dylibs/
```

## 验证方法

### 本地验证

```bash
# 1. 构建 wheel
python -m build

# 2. 解压 wheel 检查
unzip -l dist/sequenzo-*.whl | grep dylib
# 应该看到 sequenzo.libs/ 或 .dylibs/ 目录中有 libomp.dylib

# 3. 检查 .so 文件的依赖
unzip dist/sequenzo-*.whl -d /tmp/wheel-check
find /tmp/wheel-check -name "*.so" -exec otool -L {} \;
# 应该看到 @loader_path/.dylibs/libomp.dylib 而不是绝对路径

# 4. 安装并测试（在没有 libomp 的环境中）
pip install dist/sequenzo-*.whl --force-reinstall
python -c "import sequenzo.clustering.clustering_c_code; print('✓ Success')"
```

### GitHub Actions 验证

查看 workflow 日志中的以下部分：

1. **Inspecting wheel .so files before repair**：
   - 确认 .so 文件引用了 libomp.dylib

2. **Running delocate-wheel**：
   - 确认没有错误，显示 "✓ delocate-wheel succeeded"

3. **Checking bundled dependencies**：
   - `delocate-listdeps` 应该列出 libomp.dylib

4. **Inspecting repaired .so files**：
   - otool 应该显示 `@loader_path/.dylibs/libomp.dylib`

5. **Test command**：
   - 应该成功 import 且没有 symbol not found 错误

## 预期结果

### 成功的标志

1. **编译阶段**：
   ```
   [SETUP] macOS OpenMP flags: -Xpreprocessor -fopenmp
   [SETUP] Auto-detected libomp at: /opt/homebrew/opt/libomp/lib
   [SETUP] Link args for x86_64: ['-lomp', '-L/opt/homebrew/opt/libomp/lib', '-arch', 'x86_64']
   ```

2. **delocate 阶段**：
   ```
   ✓ libomp.dylib found
   ✓ delocate-wheel succeeded with --require-archs
   ```

3. **测试阶段**：
   ```
   Sequenzo import successful
   ✓ OMdistance class available
   ✓ DHDdistance class available
   ...
   ```

### 如果还有问题

如果测试仍然失败，检查：

1. **delocate 是否真正打包了 libomp**：
   - 查看 "Checking bundled dependencies" 日志
   - 如果没有列出 libomp，说明 delocate 没找到它

2. **架构不匹配**：
   - 查看 libomp.dylib 的架构（lipo -info）
   - 查看 wheel 的目标架构（{delocate_archs}）
   - 确保两者兼容

3. **DYLD_LIBRARY_PATH fallback**：
   - 检查测试命令是否设置了 DYLD_LIBRARY_PATH
   - 这至少能让 CI 测试通过，虽然不是最佳解决方案

## 相关文件

- `.github/workflows/python-app.yml`：CI/CD workflow 配置
- `setup.py`：编译配置（get_link_args 函数）
- `developer/MACOS_OPENMP_FIX.md`：之前的 OpenMP 修复文档

## 参考资料

- [delocate 文档](https://github.com/matthew-brett/delocate)
- [cibuildwheel macOS 配置](https://cibuildwheel.readthedocs.io/en/stable/options/#repair-wheel-command)
- [macOS dylib 最佳实践](https://developer.apple.com/library/archive/documentation/DeveloperTools/Conceptual/DynamicLibraries/)

## 更新日期

2025年10月7日

