# 开发工具目录 (Development Tools)

此目录包含sequenzo开发和测试相关的工具脚本。

## 📋 文件说明

### OpenMP检测工具

#### `test_openmp.py` - 通用OpenMP检测
**用途**: 跨平台OpenMP支持检测和性能验证  
**适用**: macOS, Linux, Windows  
**功能**:
- 检查C++扩展是否链接了OpenMP库
- 使用平台特定工具检测动态库依赖
  - macOS: `otool -L`
  - Linux: `ldd`
  - Windows: 基础检测
- 简单的性能测试

**使用方法**:
```bash
python tools/test_openmp.py
```

#### `check_windows_openmp.py` - Windows专用检测
**用途**: Windows环境下的详细OpenMP检测  
**适用**: 仅Windows  
**功能**:
- Windows环境信息检查
- Visual Studio Build Tools检测
- MSVC编译器支持验证
- DLL依赖分析
- 提供Windows特定的安装指导

**使用方法**:
```bash
python tools/check_windows_openmp.py
```

## 🚀 快速使用指南

### 1. 开发者首次设置
```bash
# macOS/Linux
export SEQUENZO_ENABLE_OPENMP=1
pip install -e .
python tools/test_openmp.py

# Windows
set SEQUENZO_ENABLE_OPENMP=1
pip install -e .
python tools/check_windows_openmp.py
```

### 2. 验证OpenMP状态
```bash
# 任何平台
python tools/test_openmp.py

# Windows详细检测
python tools/check_windows_openmp.py
```

### 3. 性能测试前验证
在进行任何性能基准测试之前，确保运行相应的检测脚本，确认使用的是并行版本而非串行版本。

---

💡 **提示**: 这些工具专为开发者和研究人员设计，确保sequenzo的OpenMP支持正常工作。
