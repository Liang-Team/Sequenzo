# Apple Silicon 安装指南

## 🍎 自动 OpenMP 支持

从 Sequenzo 0.1.18 开始，我们为 Apple Silicon Mac 用户提供了**自动 OpenMP 依赖管理**功能。

### ✨ 新功能

- **自动检测**: 安装时自动检测 Apple Silicon Mac
- **自动安装**: 自动通过 Homebrew 安装 `libomp`
- **智能回退**: 如果自动安装失败，提供清晰的指导
- **环境兼容**: 自动识别 Conda 环境，避免冲突

### 🚀 安装步骤

#### 方法 1: 直接安装（推荐）

```bash
pip install sequenzo
```

安装过程会自动：
1. 检测 Apple Silicon Mac
2. 检查 Homebrew 是否可用
3. 自动安装 `libomp`（如果需要）
4. 配置 OpenMP 环境变量

#### 方法 2: 手动安装（如果自动安装失败）

如果自动安装失败，请手动运行：

```bash
# 安装 Homebrew（如果未安装）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 安装 OpenMP 支持
brew install libomp

# 重新安装 Sequenzo
pip install sequenzo
```

### 🔍 验证安装

安装完成后，您可以验证 OpenMP 支持：

```python
# 检查 OpenMP 状态
python -m sequenzo.openmp_setup

# 或者运行测试脚本
python developer/test_openmp.py
```

### 📊 性能提升

启用 OpenMP 支持后，您将获得：

| 操作类型 | 串行版本 | 并行版本 | 提升倍数 |
|---------|---------|---------|---------|
| 距离计算 | 基准 | 2-4x | 2-4x |
| 聚类分析 | 基准 | 1.5-3x | 1.5-3x |
| 大数据处理 | 基准 | 2-8x | 2-8x |

### 🛠️ 故障排除

#### 问题 1: 自动安装失败

**症状**: 安装时显示 "自动安装 OpenMP 失败"

**解决方案**:
```bash
# 检查 Homebrew 是否安装
brew --version

# 如果没有安装 Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 手动安装 libomp
brew install libomp

# 重新安装 Sequenzo
pip install --force-reinstall sequenzo
```

#### 问题 2: 仍然使用串行计算

**症状**: 计算速度没有提升

**解决方案**:
```bash
# 检查 OpenMP 状态
python -m sequenzo.openmp_setup

# 如果显示 libomp 不可用，重新安装
brew reinstall libomp
pip install --force-reinstall sequenzo
```

#### 问题 3: Conda 环境冲突

**症状**: 在 Conda 环境中安装失败

**解决方案**:
```bash
# 在 Conda 环境中使用 conda 安装 OpenMP
conda install -c conda-forge libomp

# 然后安装 Sequenzo
pip install sequenzo
```

### 🔧 高级配置

#### 环境变量

您可以通过环境变量控制 OpenMP 行为：

```bash
# 强制启用 OpenMP
export SEQUENZO_ENABLE_OPENMP=1

# 设置线程数
export OMP_NUM_THREADS=8
```

#### 自定义 Homebrew 路径

如果 Homebrew 安装在非标准位置：

```bash
# 设置 Homebrew 路径
export HOMEBREW_PREFIX=/custom/path/to/homebrew

# 重新安装
pip install --force-reinstall sequenzo
```

### 📝 技术细节

#### 自动检测逻辑

1. **平台检测**: 检查是否为 macOS (`sys.platform == 'darwin'`)
2. **架构检测**: 检查是否为 Apple Silicon (`platform.machine() == 'arm64'`)
3. **环境检测**: 检查是否为 Conda 环境（避免冲突）
4. **依赖检测**: 检查 `libomp` 是否已安装
5. **自动安装**: 通过 Homebrew 安装 `libomp`
6. **环境配置**: 设置必要的环境变量

#### 文件结构

```
sequenzo/
├── openmp_setup.py          # OpenMP 设置模块
├── __init__.py              # 主模块（已更新）
└── ...

scripts/
└── post_install.py          # 安装后处理脚本

setup.py                     # 构建配置（已更新）
```

### 🆘 获取帮助

如果遇到问题，请：

1. 运行诊断命令：`python -m sequenzo.openmp_setup`
2. 查看详细日志：`pip install sequenzo -v`
3. 提交 Issue：在 GitHub 上报告问题
4. 查看文档：https://sequenzo.yuqi-liang.tech

### 🎉 享受并行计算！

安装完成后，您就可以享受 Sequenzo 的并行计算能力了！

```python
import sequenzo

# 现在所有计算都会自动使用 OpenMP 并行加速
# 无需额外配置！
```
