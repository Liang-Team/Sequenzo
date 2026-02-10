# 如何渲染 Quarto Markdown 文件

本指南将教你如何将 `tree_analysis_lsog.qmd` 渲染为 HTML 或 PDF，并确保所有代码块的结果都显示出来。

## 前置要求

1. **安装 Quarto**：
   ```bash
   # macOS (使用 Homebrew)
   brew install quarto
   
   # 或者从官网下载
   # https://quarto.org/docs/get-started/
   ```

2. **安装 Python 和所需包**：
   ```bash
   pip install sequenzo pandas numpy matplotlib
   ```

3. **验证 Quarto 安装**：
   ```bash
   quarto --version
   ```

## 渲染为 HTML

### 基本命令

```bash
cd /Users/lei/Documents/Sequenzo_all_folders/Sequenzo/Tutorials/tree_analysis
quarto render tree_analysis_lsog.qmd
```

这会在同一目录下生成 `tree_analysis_lsog.html` 文件。

### 高级选项

```bash
# 渲染并自动打开浏览器
quarto render tree_analysis_lsog.qmd --to html --open

# 指定输出文件名
quarto render tree_analysis_lsog.qmd --output my_tutorial.html

# 预览模式（实时预览，修改后自动刷新）
quarto preview tree_analysis_lsog.qmd
```

## 渲染为 PDF

### 基本命令

```bash
quarto render tree_analysis_lsog.qmd --to pdf
```

### 前置要求（PDF）

PDF 渲染需要 LaTeX：

```bash
# macOS (使用 Homebrew)
brew install --cask basictex

# 或者安装完整版
brew install --cask mactex-no-gui
```

安装后，可能需要安装额外的 LaTeX 包：

```bash
sudo tlmgr update --self
sudo tlmgr install collection-fontsrecommended
```

### 如果遇到中文问题

如果 PDF 中包含中文，可能需要额外配置。可以在 `.qmd` 文件头部添加：

```yaml
format:
  pdf:
    pdf-engine: xelatex
    include-in-header:
      text: |
        \usepackage{xeCJK}
```

## 确保所有代码块结果都显示

我已经在 `.qmd` 文件中配置了以下设置：

1. **全局设置**（在 YAML 头部）：
   ```yaml
   execute:
     echo: true      # 显示代码
     output: true    # 显示输出
     eval: true      # 执行代码
   ```

2. **每个代码块**都有：
   ```python
   #| echo: true
   #| output: true
   #| eval: true
   ```

这确保了：
- ✅ 代码会被执行（`eval: true`）
- ✅ 代码会显示（`echo: true`）
- ✅ 输出会显示（`output: true`）

## 常见问题

### 1. 代码块没有执行

如果代码块没有执行，检查：
- Python 环境是否正确
- 所需的包是否已安装
- 代码是否有错误

### 2. 输出没有显示

确保代码块中有 `output: true`，并且代码确实产生了输出（print、显示 DataFrame 等）。

### 3. 渲染很慢

如果渲染很慢，可能是因为：
- 代码执行时间较长（如计算距离矩阵）
- 数据集较大

可以：
- 减少数据集大小（如使用 `.head(60)`）
- 减少 permutation 次数（如 `R=100` 而不是 `R=1000`）

### 4. PDF 渲染失败

常见原因：
- LaTeX 未安装或配置不正确
- 缺少必要的 LaTeX 包
- 字体问题

解决方案：
- 先尝试渲染 HTML（通常更简单）
- 检查 LaTeX 安装
- 查看错误信息并安装缺失的包

## 快速开始示例

```bash
# 1. 进入目录
cd /Users/lei/Documents/Sequenzo_all_folders/Sequenzo/Tutorials/tree_analysis

# 2. 渲染 HTML（推荐，最简单）
quarto render tree_analysis_lsog.qmd

# 3. 打开生成的 HTML
open tree_analysis_lsog.html

# 或者使用预览模式（推荐用于开发）
quarto preview tree_analysis_lsog.qmd
```

## 验证输出

渲染完成后，打开生成的 HTML/PDF 文件，检查：

1. ✅ 所有代码块都显示了代码
2. ✅ 所有代码块都显示了输出结果
3. ✅ 图表和表格都正确显示
4. ✅ 格式和样式正确

## 提示

- **开发时**：使用 `quarto preview` 进行实时预览
- **最终版本**：使用 `quarto render` 生成最终文件
- **分享**：HTML 文件可以直接分享，PDF 适合打印或正式文档

## 更多资源

- Quarto 官方文档：https://quarto.org/
- Quarto Python 支持：https://quarto.org/docs/computations/python.html
- 问题反馈：查看 Quarto 社区论坛
