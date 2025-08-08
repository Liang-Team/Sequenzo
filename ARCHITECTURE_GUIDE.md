# Sequenzo Architecture Compilation Guide | Sequenzo 架构编译指南

**Author**: Yuqi Liang
**Date**: August 8, 2025  
**Version**: 1.0

---

## 🌍 English Version

### 🎯 Problems Solved
- ✅ Fixed architecture conflict warnings on macOS
- ✅ Provided users with compilation strategy choices
- ✅ Support for Intel Mac, Apple Silicon Mac, and Universal Binary
- ✅ Clear compilation feedback information

### 🚀 Usage

#### 1. Recommended Method (Auto-detection)
```bash
pip install -e .
```
The system will automatically detect your Mac type and use the optimal compilation settings.

#### 2. Intel Mac Only (Faster Compilation)
```bash
export SEQUENZO_ARCH=x86_64
pip install -e .
```

#### 3. Apple Silicon Only
```bash
export SEQUENZO_ARCH=arm64
pip install -e .
```

#### 4. Universal Binary (Compatible with All Macs)
```bash
export ARCHFLAGS="-arch x86_64 -arch arm64"
pip install -e .
```

### 🔍 Compilation Information

During compilation, you'll see output similar to:
```
[SETUP] Using hardware architecture: x86_64
[SETUP] Compiling for macOS [x86_64]
[SETUP] ⚠️  OpenMP not available - using serial compilation
[SETUP] Building 9 extension(s)...
[SETUP] ✅ Extension compilation completed!
```

### 📊 Performance Comparison

| Strategy | Compile Time | File Size | Compatibility | Recommended Use |
|----------|-------------|-----------|---------------|-----------------|
| Single Arch | Fast ⚡ | Small 📦 | Limited | Development |
| Universal | Slow 🐌 | Large 📦📦 | Perfect ✅ | Distribution |

### 🛠️ Troubleshooting

#### If you still see architecture warnings:
```bash
# Clean old compilation files
pip uninstall sequenzo
rm -rf build/ *.egg-info/
export SEQUENZO_ARCH=x86_64  # Force single architecture
pip install -e .
```

#### If compilation fails:
```bash
# Check your architecture
uname -m
# Use the simplest compilation method
export SEQUENZO_ARCH=$(uname -m)
pip install -e .
```

### 📚 Technical Details

Improved architecture detection priority:
1. `ARCHFLAGS` environment variable (pip compatible)
2. `SEQUENZO_ARCH` environment variable (project-specific)
3. Hardware architecture auto-detection

This ensures compatibility with existing tools while giving users complete control.

---

## 🇨🇳 中文版本

# Sequenzo 架构编译指南

## 🎯 解决的问题
- ✅ 修复了 macOS 上的架构冲突警告
- ✅ 给用户提供编译策略选择权
- ✅ 支持Intel Mac、Apple Silicon Mac和Universal Binary
- ✅ 提供清晰的编译反馈信息

## 🚀 使用方法

### 1. 推荐方式（自动检测）
```bash
pip install -e .
```
系统会自动检测你的Mac类型并使用最佳编译设置。

### 2. Intel Mac专用（更快编译）
```bash
export SEQUENZO_ARCH=x86_64
pip install -e .
```

### 3. Apple Silicon专用
```bash
export SEQUENZO_ARCH=arm64
pip install -e .
```

### 4. Universal Binary（兼容所有Mac）
```bash
export ARCHFLAGS="-arch x86_64 -arch arm64"
pip install -e .
```

## 🔍 编译信息说明

编译时你会看到类似的输出：
```
[SETUP] Using hardware architecture: x86_64
[SETUP] Compiling for macOS [x86_64]
[SETUP] ⚠️  OpenMP not available - using serial compilation
[SETUP] Building 9 extension(s)...
[SETUP] ✅ Extension compilation completed!
```

## 📊 性能对比

| 编译策略 | 编译时间 | 文件大小 | 兼容性 | 推荐场景 |
|---------|---------|---------|--------|----------|
| 单架构 | 快 ⚡ | 小 📦 | 限制 | 开发测试 |
| Universal | 慢 🐌 | 大 📦📦 | 完美 ✅ | 发布分发 |

## 🛠️ 问题排查

### 如果还有架构警告：
```bash
# 清理旧的编译文件
pip uninstall sequenzo
rm -rf build/ *.egg-info/
export SEQUENZO_ARCH=x86_64  # 强制单架构
pip install -e .
```

### 如果编译失败：
```bash
# 检查你的架构
uname -m
# 使用最简单的编译方式
export SEQUENZO_ARCH=$(uname -m)
pip install -e .
```

## 📚 技术细节

改进的架构检测优先级：
1. `ARCHFLAGS` 环境变量（pip兼容）
2. `SEQUENZO_ARCH` 环境变量（项目专用）
3. 硬件架构自动检测

这样既保证了与现有工具的兼容性，又给了用户完全的控制权。

---

## 📖 更新日志 | Change Log

### Version 1.0 (2025-08-08)
- **Initial Release**: Complete architecture compilation system overhaul
- **Multi-platform Support**: Intel Mac, Apple Silicon, Universal Binary
- **Smart Detection**: Intelligent architecture detection with user override options
- **Clear Feedback**: Enhanced compilation information and error reporting
- **Documentation**: Comprehensive bilingual guide (English/Chinese)

### 首次发布 (2025-08-08)
- **架构系统重构**: 完整的编译架构系统改进
- **多平台支持**: Intel Mac、Apple 芯片、Universal Binary
- **智能检测**: 带用户覆盖选项的智能架构检测
- **清晰反馈**: 增强的编译信息和错误报告
- **文档完善**: 全面的双语指南（英文/中文）


