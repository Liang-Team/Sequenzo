"""
@Author  : Yuqi Liang 梁彧祺
@File    : test_tree_analysis_lsog.py
@Time    : 2026-02-10 09:27
@Desc    : Functional and correctness tests for tree analysis using lsog (dyadic_children) dataset.

**Purpose: Functional Testing**
This test module focuses on validating that tree analysis functions work correctly
and produce reasonable results. These tests do NOT require TraMineR reference files
and can run independently. They verify:

- Function correctness: Functions execute without errors
- Data structure integrity: Return values have correct structure and types
- Logical consistency: Results satisfy expected mathematical properties
- Edge cases: Functions handle boundary conditions appropriately

**Key Characteristics:**
- No external dependencies: Does not require TraMineR reference CSV files
- Fast execution: Tests basic functionality without heavy computations
- Development-friendly: Can run during development to catch bugs early
- CI/CD ready: Suitable for continuous integration pipelines

**What is tested:**
- Function returns correct data structures
- Values are within reasonable ranges (e.g., variance > 0, R² between 0 and 1)
- Tree structures are valid (all sequences assigned to leaves, etc.)
- Weighted vs unweighted behavior is consistent
- Edge cases (empty groups, single sequences, etc.)

**When to use:**
- During development to ensure functions work correctly
- In CI/CD pipelines for quick validation
- When you want to test functionality without setting up R/TraMineR

**For numerical consistency with TraMineR, see:**
- `test_traminer_consistency.py`: Detailed numerical comparison with TraMineR

Corresponds to TraMineR functions: seqtree(), disstree(), dissvar(), dissassoc()
"""

import pytest
import pandas as pd
import numpy as np
import os
from sequenzo import SequenceData
from sequenzo.datasets import load_dataset
from sequenzo.dissimilarity_measures import get_distance_matrix
from sequenzo.tree_analysis import (
    compute_pseudo_variance,
    compute_distance_association,
    build_distance_tree,
    build_sequence_tree,
    get_leaf_membership,
    get_classification_rules,
    assign_to_leaves,
    plot_tree,
    print_tree,
    export_tree_to_dot,
)


# Test dataset setup - using dyadic_children (lsog)
@pytest.fixture
def lsog_seqdata():
    """
    Load and prepare dyadic_children dataset (lsog) for testing.
    
    Returns:
        SequenceData: Prepared sequence data object with states 1-6
    """
    # Load dataset
    df = load_dataset("dyadic_children")
    
    # Extract time columns (numeric columns)
    time_list = [c for c in df.columns if str(c).isdigit()]
    time_list = sorted(time_list, key=int)
    
    # Use first 20 rows for faster testing
    df = df.head(20)
    
    # Define states (1-indexed, matching TraMineR convention)
    states = [1, 2, 3, 4, 5, 6]
    
    # Create SequenceData object
    seqdata = SequenceData(
        df,
        time=time_list,
        id_col="dyadID",
        states=states,
    )
    
    return seqdata


@pytest.fixture
def lsog_predictors():
    """Create simple predictors for lsog data."""
    # Create binary grouping variable
    n = 20
    predictors = pd.DataFrame({
        'group': ['A'] * (n // 2) + ['B'] * (n // 2),
        'numeric_var': np.random.randn(n)
    })
    return predictors


@pytest.fixture
def lsog_distance_matrix(lsog_seqdata):
    """Compute distance matrix for lsog data."""
    print("[>] Computing distance matrix for lsog data...")
    dist_matrix = get_distance_matrix(
        seqdata=lsog_seqdata,
        method="LCS",
        norm="auto"
    )
    if isinstance(dist_matrix, pd.DataFrame):
        dist_matrix = dist_matrix.values
    return dist_matrix


# ============================================================================
# Test compute_pseudo_variance (dissvar)
# ============================================================================

def test_compute_pseudo_variance_unweighted(lsog_distance_matrix):
    """Test pseudo-variance computation without weights."""
    variance = compute_pseudo_variance(lsog_distance_matrix, weights=None, squared=False)
    
    # Check that variance is positive
    assert variance > 0, "Pseudo-variance should be positive"
    
    # Check that variance is reasonable (less than max distance)
    max_dist = np.max(lsog_distance_matrix)
    assert variance <= max_dist, "Pseudo-variance should not exceed maximum distance"
    
    print(f"[✓] Unweighted pseudo-variance: {variance:.6f}")


def test_compute_pseudo_variance_weighted(lsog_distance_matrix):
    """Test pseudo-variance computation with weights."""
    n = lsog_distance_matrix.shape[0]
    weights = np.ones(n) * 2.0  # All weights = 2
    
    variance_weighted = compute_pseudo_variance(
        lsog_distance_matrix, weights=weights, squared=False
    )
    variance_unweighted = compute_pseudo_variance(
        lsog_distance_matrix, weights=None, squared=False
    )
    
    # Weighted variance should be similar to unweighted when all weights are equal
    # (allowing for small numerical differences)
    assert np.abs(variance_weighted - variance_unweighted) < 1e-6, \
        "Weighted variance with equal weights should match unweighted"
    
    print(f"[✓] Weighted pseudo-variance: {variance_weighted:.6f}")


def test_compute_pseudo_variance_squared(lsog_distance_matrix):
    """Test pseudo-variance computation with squared distances."""
    variance_normal = compute_pseudo_variance(
        lsog_distance_matrix, weights=None, squared=False
    )
    variance_squared = compute_pseudo_variance(
        lsog_distance_matrix, weights=None, squared=True
    )
    
    # Squared variance should be different (and typically larger)
    assert variance_squared != variance_normal, \
        "Squared variance should differ from normal variance"
    
    print(f"[✓] Normal variance: {variance_normal:.6f}, "
          f"Squared variance: {variance_squared:.6f}")


# ============================================================================
# Test compute_distance_association (dissassoc)
# ============================================================================

def test_compute_distance_association_basic(lsog_distance_matrix, lsog_predictors):
    """Test basic distance association computation."""
    groups = lsog_predictors['group'].values
    
    result = compute_distance_association(
        distance_matrix=lsog_distance_matrix,
        group=groups,
        weights=None,
        R=10,  # Small R for speed
        weight_permutation="none",
        squared=False
    )
    
    # Check result structure
    assert 'pseudo_f' in result
    assert 'pseudo_r2' in result
    assert 'groups' in result
    assert 'anova_table' in result
    
    # Check that pseudo R² is between 0 and 1
    assert 0 <= result['pseudo_r2'] <= 1, \
        "Pseudo R² should be between 0 and 1"
    
    # Check that pseudo F is non-negative
    assert result['pseudo_f'] >= 0, \
        "Pseudo F-statistic should be non-negative"
    
    print(f"[✓] Pseudo R²: {result['pseudo_r2']:.4f}")
    print(f"[✓] Pseudo F: {result['pseudo_f']:.4f}")


def test_compute_distance_association_with_permutation(lsog_distance_matrix, lsog_predictors):
    """Test distance association with permutation test (R > 1)."""
    groups = lsog_predictors['group'].values
    
    result = compute_distance_association(
        distance_matrix=lsog_distance_matrix,
        group=groups,
        weights=None,
        R=10,  # Small R for speed
        weight_permutation="none",
        squared=False
    )
    
    # Check that permutation test was run (R > 1)
    assert result['R'] == 10, "Number of permutations should match R parameter"
    
    # P-value should be computed (not NaN) when R > 1
    # Note: p-value might be NaN if permutation test failed, but structure should be there
    assert 'pseudo_f_pval' in result, "Result should contain p-value field"
    
    # If p-value is computed, it should be between 0 and 1
    if not np.isnan(result['pseudo_f_pval']):
        assert 0 <= result['pseudo_f_pval'] <= 1, \
            "P-value should be between 0 and 1"
    
    print(f"[✓] Permutation test completed: p-value = {result['pseudo_f_pval']}")


# ============================================================================
# Test build_distance_tree (disstree)
# ============================================================================

def test_build_distance_tree_basic(lsog_distance_matrix, lsog_predictors):
    """Test basic distance tree building."""
    tree = build_distance_tree(
        distance_matrix=lsog_distance_matrix,
        predictors=lsog_predictors,
        weights=None,
        min_size=0.1,  # 10% minimum
        max_depth=3,
        R=10,  # Small R for speed
        pval=0.1,
        weight_permutation="none",
        squared=False
    )
    
    # Check tree structure
    assert 'root' in tree
    assert 'fitted' in tree
    assert 'info' in tree
    assert 'data' in tree
    
    # Check that fitted has correct number of rows
    assert len(tree['fitted']) == lsog_distance_matrix.shape[0], \
        "Fitted should have one row per sequence"
    
    # Check that all sequences are assigned to leaves
    leaf_ids = tree['fitted']['(fitted)'].values
    assert np.all(leaf_ids > 0), \
        "All sequences should be assigned to leaf nodes"
    
    n_leaves = len(np.unique(leaf_ids))
    print(f"[✓] Tree built with {n_leaves} leaf nodes")


def test_build_distance_tree_with_weights(lsog_distance_matrix, lsog_predictors):
    """Test distance tree building with weights."""
    n = lsog_distance_matrix.shape[0]
    weights = np.ones(n) * 2.0
    
    tree = build_distance_tree(
        distance_matrix=lsog_distance_matrix,
        predictors=lsog_predictors,
        weights=weights,
        min_size=0.1,
        max_depth=3,
        R=10,
        pval=0.1,
        weight_permutation="none",
        squared=False
    )
    
    # Check that tree was built successfully
    assert tree['info']['n'] == np.sum(weights), \
        "Tree should account for weights in total population"
    
    print(f"[✓] Weighted tree built successfully")


# ============================================================================
# Test build_sequence_tree (seqtree)
# ============================================================================

def test_build_sequence_tree_basic(lsog_seqdata, lsog_predictors):
    """Test basic sequence tree building."""
    tree = build_sequence_tree(
        seqdata=lsog_seqdata,
        predictors=lsog_predictors,
        distance_matrix=None,
        distance_method="LCS",
        distance_params={'norm': 'auto'},
        weighted=True,
        min_size=0.1,
        max_depth=3,
        R=10,
        pval=0.1,
        weight_permutation="replicate",
        squared=False
    )
    
    # Check tree structure
    assert 'root' in tree
    assert 'fitted' in tree
    assert 'seqdata' in tree
    
    # Check that tree was built
    leaf_ids = tree['fitted']['(fitted)'].values
    n_leaves = len(np.unique(leaf_ids))
    print(f"[✓] Sequence tree built with {n_leaves} leaf nodes")


def test_build_sequence_tree_with_precomputed_dist(lsog_seqdata, lsog_predictors, lsog_distance_matrix):
    """Test sequence tree building with precomputed distance matrix."""
    tree = build_sequence_tree(
        seqdata=lsog_seqdata,
        predictors=lsog_predictors,
        distance_matrix=lsog_distance_matrix,
        weighted=True,
        min_size=0.1,
        max_depth=3,
        R=10,
        pval=0.1
    )
    
    # Check that tree was built successfully
    assert len(tree['fitted']) == lsog_seqdata.seqdata.shape[0]
    print(f"[✓] Sequence tree built with precomputed distance matrix")


# ============================================================================
# Test get_leaf_membership
# ============================================================================

def test_get_leaf_membership(lsog_distance_matrix, lsog_predictors):
    """Test getting leaf memberships."""
    tree = build_distance_tree(
        distance_matrix=lsog_distance_matrix,
        predictors=lsog_predictors,
        min_size=0.1,
        max_depth=3,
        R=10,
        pval=0.1
    )
    
    # Get leaf IDs
    leaf_ids = get_leaf_membership(tree, label=False)
    
    # Check that all sequences have leaf assignments
    assert len(leaf_ids) == lsog_distance_matrix.shape[0]
    assert np.all(leaf_ids > 0)
    
    # Get leaf labels
    leaf_labels = get_leaf_membership(tree, label=True)
    
    # Check that labels are strings
    assert all(isinstance(label, str) for label in leaf_labels)
    
    print(f"[✓] Leaf membership retrieved: {len(np.unique(leaf_ids))} unique leaves")


# ============================================================================
# Test get_classification_rules and assign_to_leaves
# ============================================================================

def test_get_classification_rules(lsog_distance_matrix, lsog_predictors):
    """Test getting classification rules from tree."""
    tree = build_distance_tree(
        distance_matrix=lsog_distance_matrix,
        predictors=lsog_predictors,
        min_size=0.1,
        max_depth=3,
        R=10,
        pval=0.1
    )
    
    # Get classification rules
    rules = get_classification_rules(tree)
    
    # Check that rules are returned
    assert isinstance(rules, list), "Rules should be a list"
    assert len(rules) > 0, "Should have at least one rule"
    
    # Check that rules are strings
    assert all(isinstance(rule, str) for rule in rules), \
        "All rules should be strings"
    
    # Check if tree has splits (not just root node)
    # If tree has splits, rules should contain Python syntax
    # If tree is just root (no splits), rules may be "True" (always true condition)
    leaf_ids = get_leaf_membership(tree, label=False)
    n_leaves = len(np.unique(leaf_ids))
    
    if n_leaves > 1:
        # Tree has splits, so rules should contain condition syntax
        has_python_syntax = any(
            (' in ' in rule or '<=' in rule or '>' in rule or '&' in rule)
            for rule in rules
        )
        assert has_python_syntax, \
            f"Rules should contain Python condition syntax when tree has splits. Rules: {rules}"
    else:
        # Tree is just root node (no splits), so rules should be "True" (always true condition)
        # This is acceptable - all sequences belong to the same leaf
        assert all(rule.strip() == "True" for rule in rules), \
            f"When tree has no splits, rules should be 'True'. Got: {rules}"
        print(f"[✓] Tree has only root node (no splits), {len(rules)} rule(s) with 'True' condition")
    
    print(f"[✓] Classification rules retrieved: {len(rules)} rules")
    if len(rules) > 0:
        print(f"    Example rule: {rules[0][:80]}...")


def test_assign_to_leaves(lsog_distance_matrix, lsog_predictors):
    """Test assigning profiles to leaves using classification rules."""
    tree = build_distance_tree(
        distance_matrix=lsog_distance_matrix,
        predictors=lsog_predictors,
        min_size=0.1,
        max_depth=3,
        R=10,
        pval=0.1
    )
    
    # Get classification rules
    rules = get_classification_rules(tree)
    
    # Create a profile matching the predictors
    profile = lsog_predictors.copy()
    
    # Assign to leaves
    assignments = assign_to_leaves(rules, profile)
    
    # Check that assignments are returned
    assert isinstance(assignments, np.ndarray), "Assignments should be numpy array"
    assert len(assignments) == len(profile), \
        "Should have one assignment per profile row"
    
    # Check that assignments are valid (either leaf index or NaN)
    valid_assignments = assignments[~np.isnan(assignments)]
    if len(valid_assignments) > 0:
        assert np.all(valid_assignments >= 1), \
            "Leaf assignments should be >= 1 (1-based indexing)"
        assert np.all(valid_assignments <= len(rules)), \
            "Leaf assignments should not exceed number of rules"
    
    print(f"[✓] Profiles assigned to leaves: {len(valid_assignments)}/{len(assignments)} assigned")


# ============================================================================
# Test tree visualization functions
# ============================================================================

def test_print_tree(lsog_distance_matrix, lsog_predictors):
    """Test printing tree structure."""
    tree = build_distance_tree(
        distance_matrix=lsog_distance_matrix,
        predictors=lsog_predictors,
        min_size=0.1,
        max_depth=3,
        R=10,
        pval=0.1
    )
    
    # Test that print_tree runs without error
    # We can't easily capture stdout in pytest, so we just check it doesn't crash
    try:
        print_tree(tree, digits=3, medoid=False)
        print("[✓] print_tree() executed successfully")
    except Exception as e:
        pytest.fail(f"print_tree() raised an exception: {e}")


def test_plot_tree(lsog_distance_matrix, lsog_predictors):
    """Test plotting tree structure."""
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    tree = build_distance_tree(
        distance_matrix=lsog_distance_matrix,
        predictors=lsog_predictors,
        min_size=0.1,
        max_depth=3,
        R=10,
        pval=0.1
    )
    
    # Test that plot_tree runs without error (with show_tree=False)
    try:
        plot_tree(tree, filename=None, show_tree=False, figsize=(8, 6))
        print("[✓] plot_tree() executed successfully")
    except Exception as e:
        pytest.fail(f"plot_tree() raised an exception: {e}")


def test_export_tree_to_dot(lsog_distance_matrix, lsog_predictors):
    """Test exporting tree to GraphViz DOT format."""
    import tempfile
    import os
    
    tree = build_distance_tree(
        distance_matrix=lsog_distance_matrix,
        predictors=lsog_predictors,
        min_size=0.1,
        max_depth=3,
        R=10,
        pval=0.1
    )
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.dot', delete=False) as tmp:
        tmp_filename = tmp.name
    
    try:
        # Test export (may fail if graphviz not installed, which is OK)
        try:
            export_tree_to_dot(tree, tmp_filename.replace('.dot', ''), show_depth=False)
            # Check that file was created
            assert os.path.exists(tmp_filename), "DOT file should be created"
            print("[✓] export_tree_to_dot() executed successfully")
        except ImportError:
            # graphviz not installed - skip this test
            pytest.skip("graphviz package not installed, skipping DOT export test")
    finally:
        # Clean up
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)


# ============================================================================
# Integration test: Compare with TraMineR (if reference files exist)
# ============================================================================

def test_compare_with_traminer_dissvar(lsog_distance_matrix):
    """
    Compare compute_pseudo_variance with TraMineR dissvar().
    
    This test requires a reference file generated by TraMineR.
    To generate reference:
        Rscript tests/tree_analysis/traminer_reference.R
    """
    # TODO: Implement comparison with TraMineR reference
    # For now, just test that function works
    variance = compute_pseudo_variance(lsog_distance_matrix)
    assert variance > 0
    print(f"[✓] Variance computed: {variance:.6f} (TraMineR comparison pending)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
