# Sequenzo v0.1.21 升级指南

## 🎯 本次更新解决的问题

如果你在使用 Sequenzo v0.1.20 或更早版本时遇到以下错误：

```
ImportError: numpy.core.multiarray failed to import
```

**v0.1.21 已经完全修复了这个问题！**

## 💡 问题原因（简单解释）

- 之前的版本在编译时使用的 NumPy 版本和你安装的 NumPy 版本不匹配
- NumPy 的 C 语言接口在不同版本间不兼容
- 导致 Sequenzo 的 C++ 扩展无法正常加载

## ✅ 如何升级

### 如果你已经安装了旧版本（出现错误）

```bash
# 1. 完全卸载旧版本
pip uninstall sequenzo -y

# 2. 清理 pip 缓存
pip cache purge

# 3. 安装新版本
pip install --upgrade --no-cache-dir sequenzo
```

### 如果你是新用户

直接安装即可：

```bash
pip install sequenzo
```

## 📋 更新内容

### v0.1.21 (2025-10-07)

**主要修复：**
- ✅ 修复了 NumPy 版本兼容性问题
- ✅ 现在支持更广泛的 NumPy 版本范围（1.21.0 - 2.x）
- ✅ 用户可以安装任何兼容的 NumPy 版本而不会出错

**技术改进：**
- 使用 `oldest-supported-numpy` 编译，确保向前兼容
- 更新了 NumPy C API 版本设置（NPY_1_7_API_VERSION）
- 优化了依赖配置，避免版本冲突

## 🧪 验证安装

安装后运行以下命令验证：

```python
import sequenzo
import numpy as np

print(f"Sequenzo 版本: {sequenzo.__version__}")
print(f"NumPy 版本: {np.__version__}")
print("✅ 安装成功！")
```

## 📦 支持的 NumPy 版本

| Python 版本 | 推荐 NumPy 版本 | 支持范围 |
|------------|----------------|---------|
| 3.9        | >= 1.21.0      | 1.21.0 - 1.x |
| 3.10       | >= 1.21.0      | 1.21.0 - 1.x |
| 3.11       | >= 1.23.0      | 1.23.0 - 1.x |
| 3.12       | >= 1.26.0      | 1.26.0 - 2.x |

**注意：** 如果你的其他包需要 NumPy 2.x，Python 3.12 完全支持！

## ❓ 常见问题

### Q: 我需要卸载 NumPy 重新安装吗？

**A:** 不需要！新版本可以与你现有的 NumPy 版本一起工作（只要在支持范围内）。

### Q: 我使用的是 NumPy 2.x，能用吗？

**A:** 如果你使用 Python 3.12，完全可以！其他版本建议使用 NumPy 1.x。

### Q: 升级后还是报错怎么办？

**A:** 尝试以下步骤：

```bash
# 1. 完全清理环境
pip uninstall sequenzo numpy -y
pip cache purge

# 2. 重新安装
pip install numpy
pip install sequenzo

# 3. 如果还有问题，请提供错误信息到 GitHub Issues
```

## 📚 开发者文档

如果你是开发者，想了解技术细节，请参考：
- [developer/NUMPY_COMPATIBILITY.md](developer/NUMPY_COMPATIBILITY.md) - 详细技术说明
- [developer/PACKAGING_EXPLAINED.md](developer/PACKAGING_EXPLAINED.md) - 打包指南

## 🔗 相关链接

- GitHub: https://github.com/Liang-Team/Sequenzo
- 文档: https://sequenzo.yuqi-liang.tech
- 问题反馈: https://github.com/Liang-Team/Sequenzo/issues

---

**感谢你使用 Sequenzo！如有任何问题，欢迎在 GitHub Issues 反馈。**

