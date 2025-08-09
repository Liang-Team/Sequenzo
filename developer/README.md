# Sequenzo 开发者工具包 (Developer Toolkit)

**🎯 目标受众**: 开发者、维护者、贡献者  
**📂 性质**: 内部开发工具和技术文档  

此目录包含sequenzo项目的所有开发者专用材料，包括测试工具、技术文档和实施指南。

---

## 🔧 开发工具 (Development Tools)

### OpenMP检测和诊断工具

#### `test_openmp.py` - 通用OpenMP检测
**用途**: 跨平台OpenMP支持检测和性能验证  
**适用**: macOS, Linux, Windows  
**功能**:
- 检查C++扩展是否链接了OpenMP库
- 使用平台特定工具检测动态库依赖 (otool/ldd)
- 基础性能测试

**使用方法**:
```bash
python developer/test_openmp.py
```

#### `check_windows_openmp.py` - Windows专用检测
**用途**: Windows环境下的详细OpenMP检测  
**适用**: 仅Windows  
**功能**:
- Windows环境信息检查
- Visual Studio Build Tools检测
- MSVC编译器支持验证
- 详细的安装指导

**使用方法**:
```bash
python developer/check_windows_openmp.py
```

---

## 📚 技术文档 (Technical Documentation)

### 核心技术文档

#### `OPENMP_FIX_SUMMARY.md` - OpenMP修复报告 ⭐
**内容**: OpenMP支持的完整实现记录  
**受众**: 开发者、维护者  
**包含**:
- 问题诊断和根本原因分析
- 具体修复内容和代码变更
- 性能提升预期和验证结果
- 完整的技术实施过程

#### `OPENMP_ENHANCEMENT.md` - CI/CD增强指南
**内容**: GitHub Actions中添加OpenMP支持  
**受众**: DevOps工程师、CI/CD维护者  
**包含**:
- 平台特定的OpenMP库安装配置
- cibuildwheel环境设置
- 自动化测试和验证步骤

### 平台特定指南

#### `WINDOWS_OPENMP_GUIDE.md` - Windows开发指南
**内容**: Windows环境下的OpenMP开发配置  
**包含**:
- 快速检测和验证方法
- Visual Studio Build Tools配置
- 常见问题排查

#### `ARCHITECTURE_GUIDE.md` - macOS架构指南
**内容**: macOS多架构编译技术细节  
**包含**:
- Intel vs Apple Silicon编译配置
- Universal Binary构建流程
- 环境变量和编译标志设置

---

## 🚀 快速开始

### 1. 开发者首次设置

**macOS/Linux**:
```bash
export SEQUENZO_ENABLE_OPENMP=1
pip install -e .
python developer/test_openmp.py
```

**Windows**:
```bash
set SEQUENZO_ENABLE_OPENMP=1
pip install -e .
python developer/check_windows_openmp.py
```

### 2. 验证构建质量
```bash
# 检查OpenMP支持
python developer/test_openmp.py

# Windows详细诊断
python developer/check_windows_openmp.py  # Windows only
```

### 3. 理解技术细节
1. 阅读 `OPENMP_FIX_SUMMARY.md` - 了解整体技术架构
2. 查看平台特定指南 - 解决具体环境问题
3. 参考 `OPENMP_ENHANCEMENT.md` - 了解CI/CD配置

---

## 📖 推荐阅读顺序

### 🆕 新开发者
1. `OPENMP_FIX_SUMMARY.md` - 理解OpenMP支持的整体架构
2. 平台特定指南 - 配置本地开发环境
3. 运行相应的检测工具 - 验证环境配置

### 🔧 维护者
1. `OPENMP_FIX_SUMMARY.md` - 技术背景和实现细节
2. `OPENMP_ENHANCEMENT.md` - CI/CD维护和更新
3. 所有检测工具 - 故障诊断和用户支持

### 🚀 贡献者
1. 所有技术文档 - 理解项目架构
2. 运行所有工具 - 验证开发环境
3. 参与改进工具和文档

---

## ⚡ 性能测试前必读

在进行任何性能基准测试之前：

1. **🔍 运行OpenMP检测**: 确认使用的是并行版本
2. **📊 区分并行类型**: sklearn joblib vs sequenzo OpenMP vs numpy BLAS
3. **📝 记录环境配置**: 便于结果复现和问题排查

---

## 🔗 相关目录

- **用户教程**: `../Tutorials/` - 面向用户的使用教程
- **单元测试**: `../tests/` - 自动化测试套件
- **CI/CD配置**: `../.github/workflows/` - 构建和发布流程

---

💡 **重要提醒**: 
- 此目录的所有内容都是**开发者专用**材料
- 用户文档请参见项目根目录的 `README.md` 和 `Tutorials/` 目录
- 技术问题排查时，这里的工具和文档是首要参考资源
