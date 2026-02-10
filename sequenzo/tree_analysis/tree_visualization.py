"""
@Author  : Yuqi Liang 梁彧祺
@File    : tree_visualization.py
@Time    : 2026-02-10
@Desc    : Tree visualization functions for disstree and seqtree.

This module provides functions to visualize regression trees built from
distance matrices or sequence data. It supports both text-based and
graphical representations.

Corresponds to TraMineR functions: plot.disstree(), plot.seqtree(),
disstreedisplay(), seqtreedisplay()
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, TYPE_CHECKING
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager

if TYPE_CHECKING:
    import graphviz

try:
    import graphviz
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False
    graphviz = None  # Placeholder for type hints


def plot_tree(
    tree: Dict[str, Any],
    filename: Optional[str] = None,
    show_tree: bool = True,
    show_depth: bool = False,
    figsize: tuple = (12, 8),
    dpi: int = 100
) -> None:
    """
    Plot a distance tree or sequence tree.
    
    **Corresponds to TraMineR function: `plot.disstree()` and `plot.seqtree()`**
    
    Parameters
    ----------
    tree : dict
        Tree object from build_distance_tree() or build_sequence_tree()
    filename : str, optional
        If provided, save the plot to this file
    show_tree : bool, optional
        If True, display the tree plot
        Default: True
    show_depth : bool, optional
        If True, order splits by global pseudo-R²
        Default: False
    figsize : tuple, optional
        Figure size (width, height) in inches
        Default: (12, 8)
    dpi : int, optional
        Resolution for saved figures
        Default: 100
        
    Examples
    --------
    >>> from sequenzo.tree_analysis import build_distance_tree, plot_tree
    >>> 
    >>> # Build tree
    >>> tree = build_distance_tree(...)
    >>> 
    >>> # Plot tree
    >>> plot_tree(tree, filename="tree_plot.png")
    """
    root = tree['root']
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Recursively plot tree
    _plot_node(ax, root, tree, x=0.5, y=0.95, width=0.9, depth=0, show_depth=show_depth)
    
    # Add title
    tree_type = "Sequence Tree" if 'seqdata' in tree else "Distance Tree"
    ax.set_title(f"{tree_type} Visualization", fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"[✓] Tree plot saved to {filename}")
    
    if show_tree:
        plt.show()
    else:
        plt.close()


def _plot_node(
    ax: plt.Axes,
    node: Any,
    tree: Dict[str, Any],
    x: float,
    y: float,
    width: float,
    depth: int,
    show_depth: bool
):
    """
    Recursively plot a tree node.
    
    Parameters
    ----------
    ax : matplotlib.Axes
        Axes to plot on
    node : DissTreeNode
        Node to plot
    tree : dict
        Tree object
    x : float
        X position (0-1)
    y : float
        Y position (0-1)
    width : float
        Width of subtree (0-1)
    depth : int
        Current depth
    show_depth : bool
        Whether to show depth information
    """
    # Import here to avoid circular imports
    try:
        from .tree_node import DissTreeNode
    except ImportError:
        # Fallback if import fails
        return
    
    if not hasattr(node, 'id') or not hasattr(node, 'is_terminal'):
        return
    
    # Node info
    node_id = node.id
    n = node.info.get('n', 0)
    variance = node.info.get('vardis', 0.0)
    
    # Node text
    if node.is_terminal:
        node_text = f"Leaf {node_id}\nn={n}\nvariance={variance:.3f}"
        node_color = 'lightblue'
    else:
        split = node.split
        var_name = tree['data'].columns[split.varindex]
        
        if split.breaks is not None:
            split_text = f"{var_name} <= {split.breaks:.2f}"
        else:
            left_groups = [split.labels[i] for i in range(len(split.index)) if split.index[i] == 1]
            split_text = f"{var_name} in {left_groups}"
        
        node_text = f"Node {node_id}\n{split_text}\nn={n}\nvariance={variance:.3f}"
        node_color = 'lightgreen'
    
    # Draw node
    node_box = mpatches.FancyBboxPatch(
        (x - width/2, y - 0.03), width, 0.06,
        boxstyle="round,pad=0.01",
        facecolor=node_color,
        edgecolor='black',
        linewidth=1.5
    )
    ax.add_patch(node_box)
    
    # Add text
    ax.text(x, y, node_text, ha='center', va='center', fontsize=8, family='monospace')
    
    # Plot children
    if not node.is_terminal and node.kids:
        left_child = node.kids[0]
        right_child = node.kids[1]
        
        # Calculate positions for children
        child_width = width * 0.45
        left_x = x - width/2 + child_width/2
        right_x = x + width/2 - child_width/2
        child_y = y - 0.15
        
        # Draw connecting lines
        ax.plot([x, left_x], [y - 0.03, child_y + 0.03], 'k-', linewidth=1)
        ax.plot([x, right_x], [y - 0.03, child_y + 0.03], 'k-', linewidth=1)
        
        # Add edge labels
        ax.text((x + left_x) / 2, (y + child_y) / 2, "Yes", ha='center', va='center', fontsize=7)
        ax.text((x + right_x) / 2, (y + child_y) / 2, "No", ha='center', va='center', fontsize=7)
        
        # Recursively plot children
        _plot_node(ax, left_child, tree, left_x, child_y, child_width, depth + 1, show_depth)
        _plot_node(ax, right_child, tree, right_x, child_y, child_width, depth + 1, show_depth)


def print_tree(
    tree: Dict[str, Any],
    digits: int = 3,
    medoid: bool = True
) -> None:
    """
    Print tree structure in text format.
    
    **Corresponds to TraMineR function: `print.disstree()` and `print.seqtree()`**
    
    Parameters
    ----------
    tree : dict
        Tree object from build_distance_tree() or build_sequence_tree()
    digits : int, optional
        Number of digits to display for numeric values
        Default: 3
    medoid : bool, optional
        If True, show medoid sequences for leaf nodes (for seqtree)
        Default: True
        
    Examples
    --------
    >>> from sequenzo.tree_analysis import build_distance_tree, print_tree
    >>> 
    >>> tree = build_distance_tree(...)
    >>> print_tree(tree)
    """
    root = tree['root']
    _print_node(root, tree, prefix="", is_last=True, digits=digits, medoid=medoid)


def _print_node(
    node: Any,
    tree: Dict[str, Any],
    prefix: str,
    is_last: bool,
    digits: int,
    medoid: bool
):
    """Recursively print tree node."""
    # Import here to avoid circular imports
    try:
        from .tree_node import DissTreeNode
    except ImportError:
        return
    
    if not hasattr(node, 'id') or not hasattr(node, 'is_terminal'):
        return
    
    # Node marker
    marker = "└── " if is_last else "├── "
    
    # Node info
    node_id = node.id
    n = node.info.get('n', 0)
    variance = node.info.get('vardis', 0.0)
    
    if node.is_terminal:
        node_str = f"Leaf {node_id} (n={n}, variance={variance:.{digits}f})"
    else:
        split = node.split
        var_name = tree['data'].columns[split.varindex]
        
        if split.breaks is not None:
            split_str = f"{var_name} <= {split.breaks:.{digits}f}"
        else:
            left_groups = [split.labels[i] for i in range(len(split.index)) if split.index[i] == 1]
            split_str = f"{var_name} in {left_groups}"
        
        R2 = split.info.get('R2', 0.0)
        node_str = f"Node {node_id}: {split_str} (n={n}, variance={variance:.{digits}f}, R²={R2:.{digits}f})"
    
    print(prefix + marker + node_str)
    
    # Print children
    if not node.is_terminal and node.kids:
        extension = "    " if is_last else "│   "
        for i, child in enumerate(node.kids):
            is_last_child = (i == len(node.kids) - 1)
            _print_node(child, tree, prefix + extension, is_last_child, digits, medoid)


def export_tree_to_dot(
    tree: Dict[str, Any],
    filename: str,
    show_depth: bool = False
) -> None:
    """
    Export tree to GraphViz DOT format.
    
    **Corresponds to TraMineR function: `disstree2dot()` and `seqtree2dot()`**
    
    Parameters
    ----------
    tree : dict
        Tree object from build_distance_tree() or build_sequence_tree()
    filename : str
        Output filename (without extension)
    show_depth : bool, optional
        If True, order splits by global pseudo-R²
        Default: False
        
    Examples
    --------
    >>> from sequenzo.tree_analysis import build_distance_tree, export_tree_to_dot
    >>> 
    >>> tree = build_distance_tree(...)
    >>> export_tree_to_dot(tree, "tree")
    >>> # Then use: dot -Tpng tree.dot -o tree.png
    """
    if not HAS_GRAPHVIZ:
        raise ImportError(
            "[!] graphviz package is required for DOT export. "
            "Install with: pip install graphviz"
        )
    
    root = tree['root']
    dot = graphviz.Digraph(comment='Tree')
    
    # Recursively add nodes and edges
    _add_node_to_dot(dot, root, tree, show_depth)
    
    # Save DOT file
    dot_file = filename + ".dot"
    dot.save(dot_file)
    print(f"[✓] Tree exported to {dot_file}")
    print(f"[>] Render with: dot -Tpng {dot_file} -o {filename}.png")


def _add_node_to_dot(
    dot: Any,  # graphviz.Digraph
    node: Any,
    tree: Dict[str, Any],
    show_depth: bool
):
    """Recursively add nodes to GraphViz DOT graph."""
    # Import here to avoid circular imports
    try:
        from .tree_node import DissTreeNode
    except ImportError:
        return
    
    if not hasattr(node, 'id') or not hasattr(node, 'is_terminal'):
        return
    
    node_id = str(node.id)
    n = node.info.get('n', 0)
    variance = node.info.get('vardis', 0.0)
    
    if node.is_terminal:
        label = f"Leaf {node_id}\\nn={n}\\nvariance={variance:.3f}"
        dot.node(node_id, label, shape='box', style='filled', fillcolor='lightblue')
    else:
        split = node.split
        var_name = tree['data'].columns[split.varindex]
        
        if split.breaks is not None:
            split_text = f"{var_name} <= {split.breaks:.2f}"
        else:
            left_groups = [split.labels[i] for i in range(len(split.index)) if split.index[i] == 1]
            split_text = f"{var_name} in {left_groups}"
        
        R2 = split.info.get('R2', 0.0)
        label = f"Node {node_id}\\n{split_text}\\nn={n}\\nvariance={variance:.3f}\\nR²={R2:.3f}"
        dot.node(node_id, label, shape='ellipse', style='filled', fillcolor='lightgreen')
        
        # Add children
        if node.kids:
            left_child = node.kids[0]
            right_child = node.kids[1]
            
            dot.edge(node_id, str(left_child.id), label="Yes")
            dot.edge(node_id, str(right_child.id), label="No")
            
            _add_node_to_dot(dot, left_child, tree, show_depth)
            _add_node_to_dot(dot, right_child, tree, show_depth)
