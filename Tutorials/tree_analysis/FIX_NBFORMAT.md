# 修复 nbformat 错误

## 问题描述

渲染 Quarto 文件时出现错误：
```
ModuleNotFoundError: No module named 'nbformat'
```

虽然 PDF/HTML 可能已经生成，但代码块可能没有执行，导致输出结果缺失。

## 原因

Quarto 需要 `nbformat` 模块来执行 Python 代码块。这个模块是 Jupyter 生态系统的一部分。

## 解决方案

### 方法 1: 安装 Jupyter 相关包（推荐）

如果你在虚拟环境中（如 `sequenzo_test`），先激活环境：

```bash
# 激活你的虚拟环境（如果是 conda）
conda activate sequenzo_test

# 或者如果是 venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

然后安装必要的包：

```bash
# 安装 Jupyter 核心包（包含 nbformat）
pip install jupyter nbformat ipykernel

# 或者只安装 nbformat（最小安装）
pip install nbformat
```

### 方法 2: 使用 conda 安装

```bash
conda activate sequenzo_test
conda install nbformat -c conda-forge
```

### 方法 3: 安装完整的 Jupyter 环境

```bash
pip install jupyter notebook ipykernel nbformat
```

## 验证安装

安装后，验证是否成功：

```bash
python3 -c "import nbformat; print('nbformat version:', nbformat.__version__)"
```

应该输出类似：
```
nbformat version: 5.x.x
```

## 重新渲染

安装完成后，重新渲染：

```bash
cd /Users/lei/Documents/Sequenzo_all_folders/Sequenzo/Tutorials/tree_analysis

# 渲染 HTML
./render.sh

# 或渲染 PDF
./render.sh pdf
```

## 检查代码是否执行

渲染后，打开生成的 HTML/PDF 文件，检查：

1. ✅ 代码块是否显示了代码
2. ✅ 代码块下方是否有输出结果
3. ✅ print 语句的输出是否显示
4. ✅ DataFrame 是否显示

如果没有输出，说明代码没有执行，需要安装上述包。

## 完整依赖列表（推荐）

为了确保 Quarto 能正常执行 Python 代码，建议安装：

```bash
pip install \
    jupyter \
    nbformat \
    ipykernel \
    ipython \
    pandas \
    numpy \
    matplotlib \
    sequenzo
```

## 常见问题

### Q: 我已经安装了 Jupyter，为什么还是报错？

A: 可能是 Quarto 使用的 Python 环境与你安装 Jupyter 的环境不同。检查：

```bash
# 查看 Quarto 使用的 Python
quarto check

# 确保在正确的环境中安装
which python3
pip install nbformat
```

### Q: 如何指定 Quarto 使用特定的 Python 环境？

A: 在 `.qmd` 文件头部添加：

```yaml
jupyter: python3
```

或者在项目根目录创建 `_quarto.yml`：

```yaml
project:
  type: default

jupyter:
  python3: /path/to/your/python
```

### Q: PDF 渲染成功了，但代码没有执行？

A: 这是正常的。PDF 渲染可能成功，但代码执行需要 `nbformat`。安装后重新渲染即可。
