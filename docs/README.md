# 文档目录 (Documentation)

此目录包含sequenzo开发相关的技术文档和指南。

## 📚 文档清单

### OpenMP支持文档

#### `OPENMP_FIX_SUMMARY.md` - OpenMP修复完成报告
**内容**: 详细记录了OpenMP支持的完整修复过程  
**受众**: 开发者、维护者  
**包含**:
- 问题诊断和根本原因
- 具体修复内容和代码变更
- 性能提升预期
- 验证方法和步骤

#### `OPENMP_ENHANCEMENT.md` - OpenMP增强实施指南
**内容**: CI/CD中添加OpenMP支持的技术指南  
**受众**: DevOps工程师、CI/CD维护者  
**包含**:
- GitHub Actions workflow配置
- 平台特定的OpenMP库安装
- 构建环境设置

#### `WINDOWS_OPENMP_GUIDE.md` - Windows OpenMP支持指南
**内容**: Windows用户专用的OpenMP检测和启用指南  
**受众**: Windows用户、学生、研究人员  
**包含**:
- 快速检测方法
- 问题排查步骤
- 重新测试建议
- 常见问题解答

#### `ARCHITECTURE_GUIDE.md` - 架构编译指南
**内容**: macOS多架构编译的详细指南  
**受众**: macOS开发者  
**包含**:
- Intel vs Apple Silicon编译
- Universal Binary构建
- 环境变量配置
- 故障排除

## 📖 阅读顺序建议

### 对于新用户
1. `WINDOWS_OPENMP_GUIDE.md` (Windows用户)
2. `ARCHITECTURE_GUIDE.md` (macOS用户)

### 对于开发者
1. `OPENMP_FIX_SUMMARY.md` - 了解整体架构
2. `ARCHITECTURE_GUIDE.md` - 本地开发配置
3. `OPENMP_ENHANCEMENT.md` - CI/CD配置

### 对于维护者
1. `OPENMP_FIX_SUMMARY.md` - 技术背景
2. `OPENMP_ENHANCEMENT.md` - 部署配置
3. 其他文档 - 用户支持参考

## 🔗 相关资源

- **工具脚本**: 参见 `../tools/` 目录
- **测试**: 参见 `../tests/` 目录
- **教程**: 参见 `../Tutorials/` 目录

---

💡 **注意**: 这些文档记录了sequenzo OpenMP支持的完整实现过程，对理解项目架构和troubleshooting非常有价值。
