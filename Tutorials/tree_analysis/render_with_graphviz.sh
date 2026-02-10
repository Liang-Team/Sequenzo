#!/bin/bash
# 渲染 Quarto Markdown 文件为 HTML（包含 GraphViz 可视化）
# 使用方法: ./render_with_graphviz.sh

cd "$(dirname "$0")"

echo "🌐 Rendering to HTML with GraphViz visualizations..."
echo ""

# Check if GraphViz is installed
if ! command -v dot &> /dev/null; then
    echo "⚠️  Warning: GraphViz (dot) not found in PATH"
    echo "   Install with: brew install graphviz"
    echo "   Continuing anyway..."
    echo ""
fi

# Render HTML
quarto render tree_analysis_lsog.qmd --to html

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ HTML rendered successfully: tree_analysis_lsog.html"
    echo ""
    echo "📊 GraphViz visualizations:"
    echo "   - Sequence tree: seqtreedisplay() output"
    echo "   - Distance tree: disstreedisplay() output"
    echo "   - DOT files: tree_analysis_lsog_seqtree.dot, tree_analysis_lsog_disstree.dot"
    echo ""
    echo "💡 To view: open tree_analysis_lsog.html"
else
    echo ""
    echo "❌ Rendering failed. Check errors above."
    exit 1
fi
