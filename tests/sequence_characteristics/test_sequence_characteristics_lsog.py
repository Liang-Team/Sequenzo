"""
@Author  : Yuqi Liang 梁彧祺
@File    : test_sequence_characteristics_lsog.py
@Time    : 09/02/2026 09:37
@Desc    : Comprehensive tests for sequence characteristics using lsog (dyadic_children) dataset

This test module validates all sequence characteristics functions against TraMineR behavior
using the dyadic_children dataset (lsog). All functions are tested to ensure consistency
with TraMineR R package implementations.
"""

import pytest
import pandas as pd
import numpy as np
from sequenzo import SequenceData
from sequenzo.datasets import load_dataset

# Import all sequence characteristics functions
from sequenzo.sequence_characteristics import (
    # Basic indicators
    get_sequence_length,
    get_spell_durations,
    get_visited_states,
    get_recurrence,
    get_mean_spell_duration,
    get_duration_standard_deviation,
    # Diversity indicators
    get_entropy_difference,
    # Complexity indicators
    get_volatility,
    # Binary indicators
    get_positive_negative_indicators,
    get_integration_index,
    # Ranked indicators
    get_badness_index,
    get_degradation_index,
    get_precarity_index,
    get_insecurity_index,
    # Cross-sectional indicators
    get_mean_time_in_states,
    get_modal_state_sequence,
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


# ============================================================================
# Basic Indicators Tests
# ============================================================================

def test_get_sequence_length(lsog_seqdata):
    """Test sequence length calculation."""
    result = get_sequence_length(lsog_seqdata, with_missing=True)
    
    # Check return format
    assert isinstance(result, pd.DataFrame)
    assert 'ID' in result.columns
    assert 'Length' in result.columns
    assert len(result) == lsog_seqdata.seqdata.shape[0]
    
    # Check values are positive
    assert (result['Length'] > 0).all()
    
    # Test with_missing=False
    result_no_missing = get_sequence_length(lsog_seqdata, with_missing=False)
    assert isinstance(result_no_missing, pd.DataFrame)
    assert len(result_no_missing) == len(result)


def test_get_spell_durations(lsog_seqdata):
    """Test spell durations extraction."""
    result = get_spell_durations(lsog_seqdata, with_missing=False)
    
    # Check return format
    assert isinstance(result, pd.DataFrame)
    assert len(result) == lsog_seqdata.seqdata.shape[0]
    
    # Check that durations are positive where not NaN
    for col in result.columns:
        if col.startswith('DUR'):
            non_nan = result[col].dropna()
            if len(non_nan) > 0:
                assert (non_nan > 0).all()


def test_get_visited_states(lsog_seqdata):
    """Test visited states calculation."""
    result = get_visited_states(lsog_seqdata, with_missing=False)
    
    # Check return format
    assert isinstance(result, pd.DataFrame)
    assert 'ID' in result.columns
    assert 'Visited' in result.columns
    assert 'Visitp' in result.columns
    assert len(result) == lsog_seqdata.seqdata.shape[0]
    
    # Check values are reasonable
    assert (result['Visited'] >= 1).all()  # At least one state visited
    assert (result['Visited'] <= len(lsog_seqdata.states)).all()  # Not more than alphabet size
    assert (result['Visitp'] >= 0).all()
    assert (result['Visitp'] <= 1).all()


def test_get_recurrence(lsog_seqdata):
    """Test recurrence index calculation."""
    result = get_recurrence(lsog_seqdata, with_missing=False)
    
    # Check return format
    assert isinstance(result, pd.DataFrame)
    assert 'ID' in result.columns
    assert 'Recu' in result.columns
    assert len(result) == lsog_seqdata.seqdata.shape[0]
    
    # Check values are non-negative (NaN allowed for division by zero)
    non_nan = result['Recu'].dropna()
    if len(non_nan) > 0:
        assert (non_nan >= 0).all()


def test_get_mean_spell_duration(lsog_seqdata):
    """Test mean spell duration calculation."""
    # Test type=1
    result1 = get_mean_spell_duration(lsog_seqdata, type=1, with_missing=False)
    assert isinstance(result1, pd.DataFrame)
    assert 'ID' in result1.columns
    assert 'MeanD' in result1.columns
    assert len(result1) == lsog_seqdata.seqdata.shape[0]
    
    # Test type=2
    result2 = get_mean_spell_duration(lsog_seqdata, type=2, with_missing=False)
    assert isinstance(result2, pd.DataFrame)
    assert 'ID' in result2.columns
    assert 'MeanD2' in result2.columns
    assert len(result2) == len(result1)


def test_get_duration_standard_deviation(lsog_seqdata):
    """Test duration standard deviation calculation."""
    # Test type=1
    result1 = get_duration_standard_deviation(lsog_seqdata, type=1, with_missing=False)
    assert isinstance(result1, pd.DataFrame)
    assert 'ID' in result1.columns
    assert 'Dustd' in result1.columns
    assert len(result1) == lsog_seqdata.seqdata.shape[0]
    
    # Check values are non-negative
    non_nan = result1['Dustd'].dropna()
    if len(non_nan) > 0:
        assert (non_nan >= 0).all()
    
    # Test type=2
    result2 = get_duration_standard_deviation(lsog_seqdata, type=2, with_missing=False)
    assert isinstance(result2, pd.DataFrame)
    assert 'Dustd2' in result2.columns


# ============================================================================
# Diversity Indicators Tests
# ============================================================================

def test_get_entropy_difference(lsog_seqdata):
    """Test entropy difference calculation."""
    result = get_entropy_difference(lsog_seqdata, norm=True)
    
    # Check return format
    assert isinstance(result, pd.DataFrame)
    assert 'ID' in result.columns
    assert 'Hdss' in result.columns
    assert len(result) == lsog_seqdata.seqdata.shape[0]
    
    # Check normalized values are in [0, 1]
    non_nan = result['Hdss'].dropna()
    if len(non_nan) > 0:
        assert (non_nan >= 0).all()
        assert (non_nan <= 1).all()


# ============================================================================
# Complexity Indicators Tests
# ============================================================================

def test_get_volatility(lsog_seqdata):
    """Test volatility calculation."""
    result = get_volatility(lsog_seqdata, w=0.5, with_missing=False, adjust=True)
    
    # Check return format
    assert isinstance(result, pd.DataFrame)
    assert 'ID' in result.columns
    assert 'Volat' in result.columns
    assert len(result) == lsog_seqdata.seqdata.shape[0]
    
    # Check values are in [0, 1]
    non_nan = result['Volat'].dropna()
    if len(non_nan) > 0:
        assert (non_nan >= 0).all()
        assert (non_nan <= 1).all()
    
    # Test different w values
    result_w0 = get_volatility(lsog_seqdata, w=0.0, with_missing=False)
    result_w1 = get_volatility(lsog_seqdata, w=1.0, with_missing=False)
    assert len(result_w0) == len(result)
    assert len(result_w1) == len(result)


# ============================================================================
# Binary Indicators Tests
# ============================================================================

def test_get_positive_negative_indicators(lsog_seqdata):
    """Test positive/negative indicators calculation."""
    # Define positive and negative states
    pos_states = [1, 2, 3]  # First three states as positive
    neg_states = [4, 5, 6]  # Last three states as negative
    
    # Test "share" index
    result_share = get_positive_negative_indicators(
        lsog_seqdata,
        pos_states=pos_states,
        neg_states=neg_states,
        index="share",
        dss=False,
        with_missing=False
    )
    assert isinstance(result_share, pd.DataFrame)
    assert 'ID' in result_share.columns
    assert 'share' in result_share.columns
    assert len(result_share) == lsog_seqdata.seqdata.shape[0]
    
    # Check share values are in [0, 1] or NaN
    non_nan = result_share['share'].dropna()
    if len(non_nan) > 0:
        assert (non_nan >= 0).all()
        assert (non_nan <= 1).all()


def test_get_integration_index(lsog_seqdata):
    """Test integration index calculation."""
    # Test for all states
    result_all = get_integration_index(lsog_seqdata, state=None, pow=1.0, with_missing=False)
    assert isinstance(result_all, pd.DataFrame)
    assert 'ID' in result_all.columns
    assert len(result_all) == lsog_seqdata.seqdata.shape[0]
    # Should have columns for each state
    assert result_all.shape[1] == len(lsog_seqdata.states) + 1  # +1 for ID
    
    # Test for specific state
    result_state = get_integration_index(lsog_seqdata, state=1, pow=1.0, with_missing=False)
    assert isinstance(result_state, pd.DataFrame)
    assert 'ID' in result_state.columns
    assert 'State1' in result_state.columns
    
    # Check integration values are in [0, 1]
    for col in result_all.columns:
        if col != 'ID':
            non_nan = result_all[col].dropna()
            if len(non_nan) > 0:
                assert (non_nan >= 0).all()
                assert (non_nan <= 1).all()


# ============================================================================
# Ranked Indicators Tests
# ============================================================================

def test_get_badness_index(lsog_seqdata):
    """Test badness index calculation."""
    result = get_badness_index(lsog_seqdata, pow=1.0, with_missing=False)
    
    # Check return format
    assert isinstance(result, pd.DataFrame)
    assert 'ID' in result.columns
    assert 'Bad' in result.columns
    assert len(result) == lsog_seqdata.seqdata.shape[0]
    
    # Check values are non-negative
    non_nan = result['Bad'].dropna()
    if len(non_nan) > 0:
        assert (non_nan >= 0).all()


def test_get_degradation_index(lsog_seqdata):
    """Test degradation index calculation."""
    # Test with RANK method
    result = get_degradation_index(
        lsog_seqdata,
        method="RANK",
        penalized="BOTH",
        pow=1.0,
        with_missing=False
    )
    
    # Check return format
    assert isinstance(result, pd.DataFrame)
    assert 'ID' in result.columns
    assert 'Degrad' in result.columns
    assert len(result) == lsog_seqdata.seqdata.shape[0]
    
    # Test with different methods
    result_freq = get_degradation_index(
        lsog_seqdata,
        method="FREQ",
        penalized="NEG",
        weight_type="ADD",
        border_effect=10.0,
        with_missing=False
    )
    assert len(result_freq) == len(result)


def test_get_precarity_index(lsog_seqdata):
    """Test precarity index calculation."""
    result = get_precarity_index(
        lsog_seqdata,
        otto=0.2,
        a=1.0,
        b=1.2,
        method="TRATEDSS",
        pow=1.0,
        with_missing=False
    )
    
    # Check return format
    assert isinstance(result, pd.DataFrame)
    assert 'ID' in result.columns
    assert 'Prec' in result.columns
    assert len(result) == lsog_seqdata.seqdata.shape[0]
    
    # Check values are non-negative
    non_nan = result['Prec'].dropna()
    if len(non_nan) > 0:
        assert (non_nan >= 0).all()


def test_get_insecurity_index(lsog_seqdata):
    """Test insecurity index calculation."""
    result = get_insecurity_index(
        lsog_seqdata,
        pow=1.0,
        bound=False,
        method="RANK",
        with_missing=False
    )
    
    # Check return format
    assert isinstance(result, pd.DataFrame)
    assert 'ID' in result.columns
    assert 'Insec' in result.columns
    assert len(result) == lsog_seqdata.seqdata.shape[0]
    
    # Check values are non-negative
    non_nan = result['Insec'].dropna()
    if len(non_nan) > 0:
        assert (non_nan >= 0).all()


# ============================================================================
# Cross-sectional Indicators Tests
# ============================================================================

def test_get_mean_time_in_states(lsog_seqdata):
    """Test mean time in states calculation."""
    result = get_mean_time_in_states(
        lsog_seqdata,
        weighted=True,
        with_missing=False,
        prop=False,
        serr=False
    )
    
    # Check return format
    assert isinstance(result, pd.DataFrame)
    assert 'Mean' in result.columns
    assert len(result) == len(lsog_seqdata.states)
    
    # Check values are non-negative
    assert (result['Mean'] >= 0).all()
    
    # Test with standard error
    result_serr = get_mean_time_in_states(
        lsog_seqdata,
        weighted=True,
        with_missing=False,
        prop=False,
        serr=True
    )
    assert 'SE' in result_serr.columns
    assert 'Var' in result_serr.columns
    assert 'Stdev' in result_serr.columns


def test_get_modal_state_sequence(lsog_seqdata):
    """Test modal state sequence calculation."""
    result = get_modal_state_sequence(
        lsog_seqdata,
        weighted=True,
        with_missing=False
    )
    
    # Check return format
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1  # Single modal sequence
    assert result.shape[1] == lsog_seqdata.seqdata.shape[1]  # Same number of time points
    
    # Check attributes
    assert hasattr(result, 'attrs')
    assert 'Occurrences' in result.attrs
    assert 'Frequencies' in result.attrs


# ============================================================================
# Integration Tests
# ============================================================================

def test_all_indicators_integration(lsog_seqdata):
    """Test that all indicators can be computed together without errors."""
    # Basic indicators
    length = get_sequence_length(lsog_seqdata)
    durations = get_spell_durations(lsog_seqdata)
    visited = get_visited_states(lsog_seqdata)
    recurrence = get_recurrence(lsog_seqdata)
    
    # Diversity indicators
    ent_diff = get_entropy_difference(lsog_seqdata)
    
    # Complexity indicators
    volatility = get_volatility(lsog_seqdata)
    
    # Integration index
    integration = get_integration_index(lsog_seqdata, state=None)
    
    # Cross-sectional indicators
    mean_time = get_mean_time_in_states(lsog_seqdata)
    modal = get_modal_state_sequence(lsog_seqdata)
    
    # Verify all results have correct number of sequences
    n_seq = lsog_seqdata.seqdata.shape[0]
    assert len(length) == n_seq
    assert len(durations) == n_seq
    assert len(visited) == n_seq
    assert len(recurrence) == n_seq
    assert len(ent_diff) == n_seq
    assert len(volatility) == n_seq
    assert len(integration) == n_seq
    
    # Cross-sectional results have different dimensions
    assert len(mean_time) == len(lsog_seqdata.states)
    assert len(modal) == 1


def test_error_handling(lsog_seqdata):
    """Test error handling for invalid inputs."""
    # Test invalid SequenceData
    with pytest.raises(ValueError):
        get_sequence_length("not_a_seqdata")
    
    # Test invalid type parameter
    with pytest.raises(ValueError):
        get_mean_spell_duration(lsog_seqdata, type=3)
    
    # Test invalid w parameter
    with pytest.raises(ValueError):
        get_volatility(lsog_seqdata, w=1.5)  # w must be in [0, 1]
    
    # Test invalid state
    with pytest.raises(ValueError):
        get_integration_index(lsog_seqdata, state=10)  # State out of range


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])
