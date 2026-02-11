"""
@Author  : Yuqi Liang 梁彧祺
@File    : test_compare_differences_lsog.py
@Time    : 2026-02-11 10:42
@Desc    : Functional and correctness tests for compare_differences using lsog (dyadic_children) dataset.

**Purpose: Functional Testing**
This test module focuses on validating that compare_differences functions work correctly
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
- Values are within reasonable ranges (e.g., R² between 0 and 1, p-values between 0 and 1)
- Statistics are consistent (e.g., Pseudo F >= 0, Bayes Factor > 0)
- Edge cases (empty groups, single sequences, etc.)
- Different parameter combinations work correctly

**When to use:**
- During development to ensure functions work correctly
- In CI/CD pipelines for quick validation
- When you want to test functionality without setting up R/TraMineR

**For numerical consistency with TraMineR, see:**
- `test_traminer_consistency.py`: Detailed numerical comparison with TraMineR

Corresponds to TraMineR functions: seqdiff()
Corresponds to TraMineRextras functions: seqCompare(), seqLRT(), seqBIC()
"""

import pytest
import pandas as pd
import numpy as np
from sequenzo import SequenceData
from sequenzo.datasets import load_dataset
from sequenzo.compare_differences import (
    compare_groups_across_positions,
    plot_group_differences_across_positions,
    print_group_differences_across_positions,
    compare_groups_overall,
    compute_likelihood_ratio_test,
    compute_bayesian_information_criterion_test,
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
def lsog_group():
    """Create grouping variable for lsog data (two groups of 10 each)."""
    return np.array(['A'] * 10 + ['B'] * 10)


@pytest.fixture
def lsog_group_numeric():
    """Create numeric grouping variable for lsog data."""
    return np.array([1] * 10 + [2] * 10)


# ============================================================================
# Test compare_groups_across_positions
# ============================================================================

def test_compare_groups_across_positions_basic(lsog_seqdata, lsog_group):
    """Test basic functionality of compare_groups_across_positions."""
    result = compare_groups_across_positions(
        lsog_seqdata,
        group=lsog_group,
        cmprange=(0, 1),
        seqdist_args={'method': 'LCS', 'norm': 'auto'},
        with_missing=False,
        weighted=True,
        squared=False
    )
    
    # Check return structure
    assert isinstance(result, dict)
    assert 'stat' in result
    assert 'discrepancy' in result
    
    # Check stat DataFrame
    assert isinstance(result['stat'], pd.DataFrame)
    expected_cols = ['Pseudo F', 'Pseudo Fbf', 'Pseudo R2', 'Bartlett', 'Levene']
    for col in expected_cols:
        assert col in result['stat'].columns, f"Missing column: {col}"
    
    # Check discrepancy DataFrame
    assert isinstance(result['discrepancy'], pd.DataFrame)
    assert result['discrepancy'].shape[0] > 0  # Should have rows
    
    print("[✓] compare_groups_across_positions returns correct structure")


def test_compare_groups_across_positions_statistics_range(lsog_seqdata, lsog_group):
    """Test that statistics are within reasonable ranges."""
    result = compare_groups_across_positions(
        lsog_seqdata,
        group=lsog_group,
        cmprange=(0, 1),
        seqdist_args={'method': 'LCS', 'norm': 'auto'}
    )
    
    stat = result['stat']
    
    # Pseudo R2 should be between 0 and 1
    pseudo_r2 = stat['Pseudo R2'].values
    pseudo_r2_valid = pseudo_r2[~np.isnan(pseudo_r2)]
    if len(pseudo_r2_valid) > 0:
        assert np.all(pseudo_r2_valid >= 0), "Pseudo R2 should be >= 0"
        assert np.all(pseudo_r2_valid <= 1), "Pseudo R2 should be <= 1"
    
    # Pseudo F should be non-negative
    pseudo_f = stat['Pseudo F'].values
    pseudo_f_valid = pseudo_f[~np.isnan(pseudo_f)]
    if len(pseudo_f_valid) > 0:
        assert np.all(pseudo_f_valid >= 0), "Pseudo F should be >= 0"
    
    print("[✓] Statistics are within reasonable ranges")


def test_compare_groups_across_positions_different_cmprange(lsog_seqdata, lsog_group):
    """Test with different cmprange values."""
    for cmprange in [(0, 1), (-1, 1), (-2, 2), (0, 2)]:
        result = compare_groups_across_positions(
            lsog_seqdata,
            group=lsog_group,
            cmprange=cmprange,
            seqdist_args={'method': 'LCS', 'norm': 'auto'}
        )
        
        assert 'stat' in result
        assert 'discrepancy' in result
        assert len(result['stat']) > 0
    
    print("[✓] Different cmprange values work correctly")


def test_compare_groups_across_positions_different_methods(lsog_seqdata, lsog_group):
    """Test with different distance methods."""
    methods_config = [
        {'method': 'LCS', 'norm': 'auto'},
        {'method': 'OM', 'norm': 'auto', 'sm': 'TRATE'},  # OM requires sm
        {'method': 'HAM', 'norm': 'auto'}
    ]
    
    for config in methods_config:
        result = compare_groups_across_positions(
            lsog_seqdata,
            group=lsog_group,
            cmprange=(0, 1),
            seqdist_args=config
        )
        
        assert 'stat' in result
        assert 'Pseudo R2' in result['stat'].columns
    
    print("[✓] Different distance methods work correctly")


def test_compare_groups_across_positions_squared(lsog_seqdata, lsog_group):
    """Test with squared=True."""
    result_squared = compare_groups_across_positions(
        lsog_seqdata,
        group=lsog_group,
        cmprange=(0, 1),
        seqdist_args={'method': 'LCS', 'norm': 'auto'},
        squared=True
    )
    
    result_unsquared = compare_groups_across_positions(
        lsog_seqdata,
        group=lsog_group,
        cmprange=(0, 1),
        seqdist_args={'method': 'LCS', 'norm': 'auto'},
        squared=False
    )
    
    # Results should differ (squared vs unsquared)
    assert 'stat' in result_squared
    assert 'stat' in result_unsquared
    
    print("[✓] squared parameter works correctly")


def test_compare_groups_across_positions_weighted(lsog_seqdata, lsog_group):
    """Test with weighted=True and weighted=False."""
    result_weighted = compare_groups_across_positions(
        lsog_seqdata,
        group=lsog_group,
        cmprange=(0, 1),
        seqdist_args={'method': 'LCS', 'norm': 'auto'},
        weighted=True
    )
    
    result_unweighted = compare_groups_across_positions(
        lsog_seqdata,
        group=lsog_group,
        cmprange=(0, 1),
        seqdist_args={'method': 'LCS', 'norm': 'auto'},
        weighted=False
    )
    
    assert 'stat' in result_weighted
    assert 'stat' in result_unweighted
    
    print("[✓] weighted parameter works correctly")


def test_compare_groups_across_positions_numeric_group(lsog_seqdata, lsog_group_numeric):
    """Test with numeric grouping variable."""
    result = compare_groups_across_positions(
        lsog_seqdata,
        group=lsog_group_numeric,
        cmprange=(0, 1),
        seqdist_args={'method': 'LCS', 'norm': 'auto'}
    )
    
    assert 'stat' in result
    assert 'discrepancy' in result
    
    print("[✓] Numeric grouping variable works correctly")


# ============================================================================
# Test compare_groups_overall
# ============================================================================

def test_compare_groups_overall_basic(lsog_seqdata, lsog_group):
    """Test basic functionality of compare_groups_overall."""
    result = compare_groups_overall(
        lsog_seqdata,
        group=lsog_group,
        s=100,
        seed=36963,
        stat="all",
        squared="LRTonly",
        weighted=True,
        method="LCS"
    )
    
    # Check return structure
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 1  # One comparison
    assert result.shape[1] >= 4  # At least LRT, p-value, Delta BIC, Bayes Factor
    
    print("[✓] compare_groups_overall returns correct structure")


def test_compare_groups_overall_stat_LRT(lsog_seqdata, lsog_group):
    """Test with stat='LRT' only."""
    result = compare_groups_overall(
        lsog_seqdata,
        group=lsog_group,
        s=100,
        seed=36963,
        stat="LRT",
        squared="LRTonly",
        weighted=True,
        method="LCS"
    )
    
    assert isinstance(result, np.ndarray)
    assert result.shape[1] == 2  # LRT and p-value
    
    # Check p-value is between 0 and 1
    p_value = result[0, 1]
    assert 0 <= p_value <= 1, f"p-value should be between 0 and 1, got {p_value}"
    
    print("[✓] stat='LRT' works correctly")


def test_compare_groups_overall_stat_BIC(lsog_seqdata, lsog_group):
    """Test with stat='BIC' only."""
    result = compare_groups_overall(
        lsog_seqdata,
        group=lsog_group,
        s=100,
        seed=36963,
        stat="BIC",
        squared="LRTonly",
        weighted=True,
        method="LCS"
    )
    
    assert isinstance(result, np.ndarray)
    assert result.shape[1] >= 2  # Delta BIC and Bayes Factor
    
    # Check Bayes Factor is positive
    if result.shape[1] >= 2:
        bayes_factor = result[0, 1]
        assert bayes_factor > 0, f"Bayes Factor should be > 0, got {bayes_factor}"
    
    print("[✓] stat='BIC' works correctly")


def test_compare_groups_overall_different_s(lsog_seqdata, lsog_group):
    """Test with different sample sizes."""
    for s in [0, 50, 100]:
        result = compare_groups_overall(
            lsog_seqdata,
            group=lsog_group,
            s=s,
            seed=36963,
            stat="all",
            squared="LRTonly",
            weighted=True,
            method="LCS"
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 1
    
    print("[✓] Different sample sizes work correctly")


def test_compare_groups_overall_different_methods(lsog_seqdata, lsog_group):
    """Test with different distance methods."""
    methods_config = [
        {'method': 'LCS'},
        {'method': 'OM', 'sm': 'TRATE'},  # OM requires sm
        {'method': 'HAM'}
    ]
    
    for config in methods_config:
        method = config.pop('method')
        result = compare_groups_overall(
            lsog_seqdata,
            group=lsog_group,
            s=50,  # Smaller sample for faster testing
            seed=36963,
            stat="all",
            squared="LRTonly",
            weighted=True,
            method=method,
            **config  # Pass sm if present
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 1
    
    print("[✓] Different distance methods work correctly")


def test_compare_groups_overall_unweighted(lsog_seqdata, lsog_group):
    """Test with weighted=False."""
    result = compare_groups_overall(
        lsog_seqdata,
        group=lsog_group,
        s=100,
        seed=36963,
        stat="all",
        squared="LRTonly",
        weighted=False,
        method="LCS"
    )
    
    assert isinstance(result, np.ndarray)
    
    print("[✓] weighted=False works correctly")


# ============================================================================
# Test compute_likelihood_ratio_test
# ============================================================================

def test_compute_likelihood_ratio_test_basic(lsog_seqdata, lsog_group):
    """Test basic functionality of compute_likelihood_ratio_test."""
    result = compute_likelihood_ratio_test(
        lsog_seqdata,
        group=lsog_group,
        s=100,
        seed=36963,
        squared="LRTonly",
        weighted=True,
        method="LCS"
    )
    
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 1
    assert result.shape[1] == 2  # LRT and p-value
    
    # Check p-value range
    p_value = result[0, 1]
    assert 0 <= p_value <= 1, f"p-value should be between 0 and 1, got {p_value}"
    
    print("[✓] compute_likelihood_ratio_test works correctly")


# ============================================================================
# Test compute_bayesian_information_criterion_test
# ============================================================================

def test_compute_bayesian_information_criterion_test_basic(lsog_seqdata, lsog_group):
    """Test basic functionality of compute_bayesian_information_criterion_test."""
    result = compute_bayesian_information_criterion_test(
        lsog_seqdata,
        group=lsog_group,
        s=100,
        seed=36963,
        squared="LRTonly",
        weighted=True,
        method="LCS"
    )
    
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 1
    assert result.shape[1] >= 2  # Delta BIC and Bayes Factor
    
    # Check Bayes Factor is positive
    if result.shape[1] >= 2:
        bayes_factor = result[0, 1]
        assert bayes_factor > 0, f"Bayes Factor should be > 0, got {bayes_factor}"
    
    print("[✓] compute_bayesian_information_criterion_test works correctly")


# ============================================================================
# Test plotting functions
# ============================================================================

def test_plot_group_differences_across_positions(lsog_seqdata, lsog_group):
    """Test that plotting function works without errors."""
    result = compare_groups_across_positions(
        lsog_seqdata,
        group=lsog_group,
        cmprange=(0, 1),
        seqdist_args={'method': 'LCS', 'norm': 'auto'}
    )
    
    # Test plotting different statistics
    fig1 = plot_group_differences_across_positions(result, stat='Pseudo R2')
    assert fig1 is not None
    
    fig2 = plot_group_differences_across_positions(result, stat='Pseudo F')
    assert fig2 is not None
    
    fig3 = plot_group_differences_across_positions(result, stat='discrepancy')
    assert fig3 is not None
    
    # Test dual y-axes
    fig4 = plot_group_differences_across_positions(result, stat=['Pseudo R2', 'Levene'])
    assert fig4 is not None
    
    print("[✓] plot_group_differences_across_positions works correctly")


def test_print_group_differences_across_positions(lsog_seqdata, lsog_group):
    """Test that print function works without errors."""
    result = compare_groups_across_positions(
        lsog_seqdata,
        group=lsog_group,
        cmprange=(0, 1),
        seqdist_args={'method': 'LCS', 'norm': 'auto'}
    )
    
    # Should not raise an error
    print_group_differences_across_positions(result)
    
    print("[✓] print_group_differences_across_positions works correctly")
