"""
@Author  : Yuqi Liang 梁彧祺
@File    : tree_node.py
@Time    : 2026-02-09 14:35
@Desc    : Data structures for tree nodes in distance-based tree analysis.

           Corresponds to TraMineR internal structures: DissTreeNode, DissTreeSplit
"""

from typing import Optional, List, Dict, Any
import numpy as np


class DissTreeSplit:
    """
    Represents a split in a distance tree.
    
    This class stores information about how a node is split into two children,
    including which variable was used, the split point or grouping, and
    statistical information about the split.
    
    **Corresponds to TraMineR structure: `DissTreeSplit`**
    """
    
    def __init__(
        self,
        varindex: int,
        index: Optional[np.ndarray] = None,
        prob: Optional[np.ndarray] = None,
        info: Optional[Dict[str, Any]] = None,
        breaks: Optional[float] = None,
        naGroup: Optional[int] = None,
        labels: Optional[List[str]] = None
    ):
        """
        Initialize a tree split.
        
        Parameters
        ----------
        varindex : int
            Index of the variable used for splitting (0-based)
            
        index : np.ndarray, optional
            For categorical variables: array indicating which groups go to left (1) or right (2)
            
        prob : np.ndarray, optional
            Probabilities of left and right children [left_prob, right_prob]
            
        info : dict, optional
            Dictionary containing split statistics:
            - lpop: left population size
            - rpop: right population size
            - lvar: left variance
            - rvar: right variance
            - SCres: residual sum of squares
            - R2: R-squared (proportion of variance explained)
            - pval: p-value from permutation test
            
        breaks : float, optional
            For continuous variables: the split point value
            
        naGroup : int, optional
            Which child (1=left, 2=right) should contain missing values
            
        labels : List[str], optional
            For categorical variables: labels for each group level
        """
        self.varindex = varindex
        self.index = index
        self.prob = prob
        self.info = info if info is not None else {}
        self.breaks = breaks
        self.naGroup = naGroup
        self.labels = labels


class DissTreeNode:
    """
    Represents a node in a distance-based regression tree.
    
    Each node can be either a terminal node (leaf) or an internal node
    that splits the data based on a covariate.
    
    **Corresponds to TraMineR structure: `DissTreeNode`**
    """
    
    def __init__(
        self,
        node_id: int,
        indices: np.ndarray,
        variance: float,
        depth: int,
        medoid: Optional[int] = None,
        split: Optional[DissTreeSplit] = None,
        children: Optional[List['DissTreeNode']] = None
    ):
        """
        Initialize a tree node.
        
        Parameters
        ----------
        node_id : int
            Unique identifier for this node
            
        indices : np.ndarray
            Indices of sequences belonging to this node (0-based)
            
        variance : float
            Pseudo-variance (discrepancy) within this node
            
        depth : int
            Depth of this node in the tree (root = 1)
            
        medoid : int, optional
            Index of the medoid sequence for this node
            
        split : DissTreeSplit, optional
            Split information if this is an internal node. None for terminal nodes.
            
        children : List[DissTreeNode], optional
            Child nodes [left, right]. None for terminal nodes.
        """
        self.id = node_id
        self.info = {
            'ind': indices,
            'vardis': variance,
            'depth': depth,
            'n': len(indices),  # Will be updated with weights if needed
            'medoid': medoid
        }
        self.split = split
        self.kids = children  # TraMineR uses 'kids', we keep same name for compatibility
        
    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal (leaf) node."""
        return self.split is None or self.kids is None
    
    def __repr__(self) -> str:
        """String representation of the node."""
        node_type = "Terminal" if self.is_terminal else "Internal"
        return (
            f"DissTreeNode(id={self.id}, type={node_type}, "
            f"depth={self.info['depth']}, n={self.info['n']}, "
            f"variance={self.info['vardis']:.4f})"
        )
