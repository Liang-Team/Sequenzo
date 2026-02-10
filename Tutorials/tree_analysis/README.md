# Tree Analysis Tutorial - 渲染指南

本目录包含 Tree Analysis 教程的 Quarto Markdown 文件。

## 📁 文件说明

- `tree_analysis_lsog.qmd` - Quarto Markdown 源文件
- `tree_analysis_lsog.ipynb` - Jupyter Notebook 版本
- `render.sh` - 快速渲染脚本
- `RENDER_INSTRUCTIONS.md` - 详细渲染说明

## 🚀 快速开始

### 方法 1: 使用渲染脚本（推荐）

```bash
# 进入目录
cd /Users/lei/Documents/Sequenzo_all_folders/Sequenzo/Tutorials/tree_analysis

# 渲染为 HTML
./render.sh

# 或渲染为 PDF
./render.sh pdf

# 或启动预览模式（实时预览，修改后自动刷新）
./render.sh preview
```

### 方法 2: 直接使用 Quarto 命令

```bash
# 渲染为 HTML
quarto render tree_analysis_lsog.qmd

# 渲染为 PDF
quarto render tree_analysis_lsog.qmd --to pdf

# 预览模式（推荐用于开发）
quarto preview tree_analysis_lsog.qmd
```

## ✅ 确保所有代码块结果都显示

我已经在 `.qmd` 文件中配置了以下设置：

### 1. 全局设置（YAML 头部）

```yaml
execute:
  echo: true      # 显示代码
  output: true    # 显示输出
  eval: true      # 执行代码
```

### 2. 每个代码块都有

```python
#| echo: true
#| output: true
#| eval: true
```

这确保了：
- ✅ 代码会被执行（`eval: true`）
- ✅ 代码会显示（`echo: true`）
- ✅ 输出会显示（`output: true`）

## 📋 前置要求

1. **Quarto**（已安装 ✅ 版本 1.3.450）
   ```bash
   quarto --version
   ```

2. **Python 环境**和所需包：
   ```bash
   # 激活你的环境（如 sequenzo_test）
   conda activate sequenzo_test
   
   # 安装 Sequenzo 和相关包
   pip install sequenzo pandas numpy matplotlib
   
   # ⚠️ 重要：安装 Jupyter 相关包（Quarto 执行代码需要）
   pip install jupyter nbformat ipykernel
   ```

3. **PDF 渲染**（可选，仅渲染 PDF 时需要）：
   ```bash
   brew install --cask basictex
   ```

## ⚠️ 常见错误：ModuleNotFoundError: No module named 'nbformat'

如果渲染时出现这个错误，说明缺少 `nbformat` 模块。虽然 PDF/HTML 可能已生成，但代码块可能没有执行。

**解决方法**：
```bash
# 确保在正确的环境中
conda activate sequenzo_test  # 或你的环境名

# 安装 nbformat
pip install nbformat

# 或者安装完整的 Jupyter 环境
pip install jupyter nbformat ipykernel
```

详细说明请查看 `FIX_NBFORMAT.md`

## 🎯 渲染后的文件

- **HTML**: `tree_analysis_lsog.html` - 可以在浏览器中打开
- **PDF**: `tree_analysis_lsog.pdf` - 适合打印或分享

## 💡 提示

- **开发时**：使用 `quarto preview` 进行实时预览
- **最终版本**：使用 `quarto render` 生成最终文件
- **分享**：HTML 文件可以直接分享，PDF 适合正式文档

## ❓ 常见问题

### 代码块没有执行？

检查：
- Python 环境是否正确
- 所需的包是否已安装
- 代码是否有错误

### 输出没有显示？

确保：
- 代码块中有 `output: true`
- 代码确实产生了输出（print、显示 DataFrame 等）

### 渲染很慢？

可能原因：
- 代码执行时间较长（如计算距离矩阵）
- 数据集较大

解决方案：
- 减少数据集大小（如使用 `.head(60)`）
- 减少 permutation 次数（如 `R=100`）

## 📚 更多信息

详细说明请查看 `RENDER_INSTRUCTIONS.md`
