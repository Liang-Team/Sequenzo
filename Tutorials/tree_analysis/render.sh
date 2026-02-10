#!/bin/bash
# 渲染 Quarto Markdown 文件为 HTML 或 PDF
# 使用方法: ./render.sh [html|pdf]

cd "$(dirname "$0")"

if [ "$1" == "pdf" ]; then
    echo "📄 Rendering to PDF..."
    quarto render tree_analysis_lsog.qmd --to pdf
    echo "✅ PDF rendered: tree_analysis_lsog.pdf"
elif [ "$1" == "preview" ]; then
    echo "👀 Starting preview mode (auto-refresh)..."
    quarto preview tree_analysis_lsog.qmd
else
    echo "🌐 Rendering to HTML..."
    quarto render tree_analysis_lsog.qmd --to html
    echo "✅ HTML rendered: tree_analysis_lsog.html"
    echo ""
    echo "💡 Tip: Use './render.sh preview' for live preview mode"
    echo "💡 Tip: Use './render.sh pdf' to render as PDF"
fi
