"""
@Author  : Yapeng Wei
@File    : test_seqhmm_posterior.py
@Desc    :
Tests for posterior probability consistency between sequenzo.seqhmm and
R seqHMM (https://github.com/helske/seqHMM).

Posterior probabilities gamma(i, t) = P(hidden state = i | all observations)
are computed via the forward-backward algorithm. This test verifies that
Python (hmmlearn backend) and R (seqHMM C++ backend) produce identical
posterior probability matrices given the same model parameters and data.

Methodology:
  - Same synthetic data as test_seqhmm_loglik.py:
    5 sequences, 8 time points, 3 observed states (A, B, C).
  - Four HMM parameter configurations (A, B, C, D) with fixed parameters.
  - R seqHMM computes posterior_probs(model) as ground truth.
  - Python sequenzo computes HMM.predict_proba() and compares.

Test groups:
  Part 0: Sanity checks (no R needed)
    - predict_proba returns correct shape
    - posterior probs sum to 1 at each time point
    - posterior probs are in [0, 1]
    - predict_proba is deterministic
    - Config D (uniform) posteriors are uniform (all states equal)
    - Most-probable state from posteriors agrees with Viterbi predict()

  Part 1: Posterior probs vs R seqHMM (needs ref CSVs from R script)
    - Config A: 2 states, basic
    - Config B: 3 states
    - Config C: 2 states, sticky
    - Config D: 2 states, uniform

Run seqhmm_reference_posterior.R to generate ref_posterior_*.csv:
  Rscript seqhmm_reference_posterior.R .
"""
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd
import pytest

from sequenzo import SequenceData
from sequenzo.seqhmm import build_hmm


# ============================================================================
# Constants (shared with test_seqhmm_loglik.py -- MUST remain identical)
# ============================================================================

STATES = ["A", "B", "C"]
N_SYMBOLS = len(STATES)
SEQ_IDS = ["s0", "s1", "s2", "s3", "s4"]
TIME_COLS = ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"]
N_SEQUENCES = len(SEQ_IDS)
N_TIMEPOINTS = len(TIME_COLS)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================================
# Synthetic test data (MUST be identical to R script and loglik test)
# ============================================================================

TEST_DATA_RAW = [
    ["A", "A", "B", "B", "C", "A", "A", "B"],  # s0
    ["B", "C", "C", "A", "A", "B", "C", "C"],  # s1
    ["A", "B", "C", "A", "B", "C", "A", "B"],  # s2
    ["C", "C", "B", "A", "A", "A", "B", "C"],  # s3
    ["A", "A", "A", "B", "B", "B", "C", "C"],  # s4
]


# ============================================================================
# HMM parameter configurations (MUST be identical to R script and loglik test)
# ============================================================================

CONFIGS = {
    "A": {
        "n_states": 2,
        "initial_probs": np.array([0.6, 0.4]),
        "transition_probs": np.array([
            [0.7, 0.3],
            [0.2, 0.8],
        ]),
        "emission_probs": np.array([
            [0.5, 0.3, 0.2],
            [0.1, 0.4, 0.5],
        ]),
    },
    "B": {
        "n_states": 3,
        "initial_probs": np.array([0.5, 0.3, 0.2]),
        "transition_probs": np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.1, 0.7],
        ]),
        "emission_probs": np.array([
            [0.6, 0.3, 0.1],
            [0.1, 0.6, 0.3],
            [0.3, 0.1, 0.6],
        ]),
    },
    "C": {
        "n_states": 2,
        "initial_probs": np.array([0.9, 0.1]),
        "transition_probs": np.array([
            [0.95, 0.05],
            [0.05, 0.95],
        ]),
        "emission_probs": np.array([
            [0.8, 0.15, 0.05],
            [0.05, 0.15, 0.8],
        ]),
    },
    "D": {
        "n_states": 2,
        "initial_probs": np.array([0.5, 0.5]),
        "transition_probs": np.array([
            [0.5, 0.5],
            [0.5, 0.5],
        ]),
        "emission_probs": np.array([
            [1.0 / 3, 1.0 / 3, 1.0 / 3],
            [1.0 / 3, 1.0 / 3, 1.0 / 3],
        ]),
    },
}


# ============================================================================
# Helper functions
# ============================================================================

def _make_test_seqdata():
    """Create SequenceData from synthetic test data."""
    df = pd.DataFrame(TEST_DATA_RAW, columns=TIME_COLS)
    df.insert(0, "id", SEQ_IDS)
    return SequenceData(df, time=TIME_COLS, states=STATES, id_col="id")


def _build_hmm_for_config(seqdata, config_name):
    """Build an HMM with the specified config."""
    cfg = CONFIGS[config_name]
    return build_hmm(
        observations=seqdata,
        n_states=cfg["n_states"],
        initial_probs=cfg["initial_probs"].copy(),
        transition_probs=cfg["transition_probs"].copy(),
        emission_probs=cfg["emission_probs"].copy(),
    )


def _reshape_posteriors(posteriors_flat, n_sequences, n_timepoints, n_states):
    """Reshape flat posterior array (total_timepoints, n_states) to
    (n_sequences, n_timepoints, n_states) for structured comparison.

    hmmlearn returns posteriors as a single concatenated array where
    rows 0..7 = sequence 0, rows 8..15 = sequence 1, etc.
    """
    expected_total = n_sequences * n_timepoints
    assert posteriors_flat.shape[0] == expected_total, (
        f"Expected {expected_total} rows, got {posteriors_flat.shape[0]}"
    )
    assert posteriors_flat.shape[1] == n_states, (
        f"Expected {n_states} columns, got {posteriors_flat.shape[1]}"
    )
    return posteriors_flat.reshape(n_sequences, n_timepoints, n_states)


def _run_r_reference(outdir, timeout=60):
    """Run seqhmm_reference_posterior.R to generate ref_posterior_*.csv."""
    r_script = os.path.join(THIS_DIR, "seqhmm_reference_posterior.R")
    if not os.path.isfile(r_script):
        return False
    try:
        result = subprocess.run(
            ["Rscript", r_script, outdir],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=THIS_DIR,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _load_ref_posterior(ref_dir, config_name):
    """Load ref_posterior_<config>.csv and return as numpy array
    of shape (n_sequences, n_timepoints, n_states).

    CSV format from R script:
      seq_idx (0-based), time_idx (0-based), State_1, State_2, ...
    """
    path = os.path.join(ref_dir, f"ref_posterior_{config_name}.csv")
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)

    # Identify state columns (everything except seq_idx and time_idx)
    state_cols = [c for c in df.columns if c.startswith("State_")]
    n_states = len(state_cols)

    # Sort by seq_idx then time_idx to ensure consistent ordering
    df = df.sort_values(["seq_idx", "time_idx"]).reset_index(drop=True)

    # Extract posterior values and reshape
    posteriors_flat = df[state_cols].values
    n_seq = int(df["seq_idx"].max()) + 1
    n_time = int(df["time_idx"].max()) + 1
    return posteriors_flat.reshape(n_seq, n_time, n_states)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def seqdata():
    """SequenceData from synthetic test data."""
    return _make_test_seqdata()


@pytest.fixture(scope="module")
def ref_posteriors():
    """Load R reference posterior probabilities for all configs.

    Returns dict: config_name -> numpy array (n_seq, n_time, n_states).
    """
    # Check if ref files already exist in THIS_DIR
    refs = {}
    all_found = True
    for config_name in CONFIGS:
        arr = _load_ref_posterior(THIS_DIR, config_name)
        if arr is not None:
            refs[config_name] = arr
        else:
            all_found = False
            break

    if all_found:
        return refs

    # Try running R script to generate references
    outdir = tempfile.mkdtemp()
    ok = _run_r_reference(outdir)
    if ok:
        refs = {}
        for config_name in CONFIGS:
            arr = _load_ref_posterior(outdir, config_name)
            if arr is not None:
                refs[config_name] = arr
        if len(refs) == len(CONFIGS):
            return refs

    pytest.skip(
        "R/seqHMM not available and ref_posterior_*.csv not found. "
        "Run: Rscript seqhmm_reference_posterior.R . "
        "to generate reference values."
    )


# ============================================================================
# Part 0: Sanity checks (no R reference needed)
# ============================================================================

POSTERIOR_ATOL = 1e-6


class TestPosteriorSanity:
    """Sanity checks for posterior probabilities (no R needed)."""

    def test_predict_proba_shape(self, seqdata):
        """predict_proba returns (total_timepoints, n_states) array."""
        hmm = _build_hmm_for_config(seqdata, "A")
        posteriors = hmm.predict_proba()
        total_timepoints = N_SEQUENCES * N_TIMEPOINTS  # 5 * 8 = 40
        assert posteriors.shape == (total_timepoints, 2), (
            f"Expected shape ({total_timepoints}, 2), got {posteriors.shape}"
        )

    def test_predict_proba_shape_3states(self, seqdata):
        """Config B (3 states): predict_proba returns (..., 3) columns."""
        hmm = _build_hmm_for_config(seqdata, "B")
        posteriors = hmm.predict_proba()
        total_timepoints = N_SEQUENCES * N_TIMEPOINTS
        assert posteriors.shape == (total_timepoints, 3), (
            f"Expected shape ({total_timepoints}, 3), got {posteriors.shape}"
        )

    def test_posteriors_sum_to_one(self, seqdata):
        """Posterior probabilities must sum to 1 at every time point."""
        for config_name in CONFIGS:
            hmm = _build_hmm_for_config(seqdata, config_name)
            posteriors = hmm.predict_proba()
            row_sums = posteriors.sum(axis=1)
            assert np.allclose(row_sums, 1.0, atol=1e-10), (
                f"Config {config_name}: posterior row sums deviate from 1. "
                f"Max deviation: {np.max(np.abs(row_sums - 1.0))}"
            )

    def test_posteriors_in_unit_interval(self, seqdata):
        """All posterior probabilities must be in [0, 1]."""
        for config_name in CONFIGS:
            hmm = _build_hmm_for_config(seqdata, config_name)
            posteriors = hmm.predict_proba()
            assert np.all(posteriors >= 0), (
                f"Config {config_name}: negative posterior probabilities found"
            )
            assert np.all(posteriors <= 1.0 + 1e-10), (
                f"Config {config_name}: posterior probabilities exceed 1"
            )

    def test_predict_proba_deterministic(self, seqdata):
        """Same model + same data must always produce the same posteriors."""
        hmm = _build_hmm_for_config(seqdata, "A")
        p1 = hmm.predict_proba()
        p2 = hmm.predict_proba()
        assert np.array_equal(p1, p2), "predict_proba is not deterministic"

    def test_uniform_model_posteriors(self, seqdata):
        """Config D (uniform): all hidden states should have equal posteriors.

        With uniform emissions, transitions, and initial probs, no observation
        provides any evidence about the hidden state. So the posterior should
        be uniform: P(state_i | observations) = 1/n_states = 0.5 for 2 states.
        """
        hmm = _build_hmm_for_config(seqdata, "D")
        posteriors = hmm.predict_proba()
        n_states = CONFIGS["D"]["n_states"]
        expected = 1.0 / n_states
        assert np.allclose(posteriors, expected, atol=1e-10), (
            f"Uniform model posteriors should all be {expected}. "
            f"Max deviation: {np.max(np.abs(posteriors - expected))}"
        )

    def test_posteriors_vary_across_configs(self, seqdata):
        """Different configs should produce different posterior matrices."""
        posteriors = {}
        for name in ["A", "C"]:
            hmm = _build_hmm_for_config(seqdata, name)
            posteriors[name] = hmm.predict_proba()
        assert not np.allclose(posteriors["A"], posteriors["C"], atol=1e-4), (
            "Configs A and C should produce different posteriors"
        )

    def test_sticky_model_first_timepoint(self, seqdata):
        """Config C (sticky): posterior at t=0 should strongly reflect
        initial_probs and the first observation.

        With init_probs = [0.9, 0.1] and emission_probs:
          State 1: P(A)=0.8, P(B)=0.15, P(C)=0.05
          State 2: P(A)=0.05, P(B)=0.15, P(C)=0.8

        For sequence s0 starting with 'A': prior strongly favors State 1
        (init=0.9), and emission also strongly favors State 1 (P(A|S1)=0.8
        vs P(A|S2)=0.05). So posterior for State 1 at t=0 should be > 0.99.
        """
        hmm = _build_hmm_for_config(seqdata, "C")
        posteriors = hmm.predict_proba()
        # First time point of first sequence (s0, t=0, obs='A')
        posterior_s0_t0 = posteriors[0, :]
        assert posterior_s0_t0[0] > 0.99, (
            f"Config C, s0 t=0 (obs=A): P(State 1) should be > 0.99, "
            f"got {posterior_s0_t0[0]}"
        )

    def test_argmax_consistent_with_predict(self, seqdata):
        """argmax of posteriors should match predict() (Viterbi) in most cases.

        Note: This is not guaranteed in general (posterior decoding != Viterbi),
        but for well-separated models it should hold for most time points.
        We check >= 80% agreement.
        """
        hmm = _build_hmm_for_config(seqdata, "C")  # sticky = well-separated
        posteriors = hmm.predict_proba()
        viterbi = hmm.predict()

        argmax_states = np.argmax(posteriors, axis=1)
        agreement = np.mean(argmax_states == viterbi)
        assert agreement >= 0.8, (
            f"Posterior argmax agrees with Viterbi only {agreement:.0%} "
            f"of the time (expected >= 80%)"
        )


# ============================================================================
# Part 1: Posterior probs vs R seqHMM
# ============================================================================
# Tolerance notes:
#   Posterior probabilities are computed from forward and backward variables
#   via: gamma(i,t) = alpha(i,t) * beta(i,t) / P(Y)
#   Both R and Python use log-space computations for numerical stability.
#   Expected difference: < 1e-10 for identical parameters.
#   We use atol=1e-6 as a conservative margin.


def _compare_posteriors(python_posteriors_flat, r_posteriors_3d, config_name,
                        n_sequences, n_timepoints, n_states, atol=POSTERIOR_ATOL):
    """Compare Python posterior matrix (flat) with R posterior matrix (3D).

    Python: shape (total_timepoints, n_states), concatenated
    R: shape (n_sequences, n_timepoints, n_states)
    """
    py_3d = _reshape_posteriors(
        python_posteriors_flat, n_sequences, n_timepoints, n_states
    )
    assert py_3d.shape == r_posteriors_3d.shape, (
        f"Config {config_name}: shape mismatch: "
        f"Python {py_3d.shape} vs R {r_posteriors_3d.shape}"
    )
    max_diff = np.max(np.abs(py_3d - r_posteriors_3d))
    assert np.allclose(py_3d, r_posteriors_3d, atol=atol), (
        f"Config {config_name}: posterior probabilities differ. "
        f"Max absolute difference: {max_diff:.2e} (tolerance: {atol}). "
        f"First mismatch location: {_first_mismatch(py_3d, r_posteriors_3d, atol)}"
    )


def _first_mismatch(a, b, atol):
    """Find (seq_idx, time_idx, state_idx) of first large difference."""
    diff = np.abs(a - b)
    idx = np.unravel_index(np.argmax(diff), diff.shape)
    return (
        f"seq={idx[0]}, t={idx[1]}, state={idx[2]}: "
        f"Python={a[idx]:.8f}, R={b[idx]:.8f}"
    )


def test_posterior_config_A_matches_seqhmm(seqdata, ref_posteriors):
    """Config A (2 states, basic): posterior probs match R seqHMM."""
    hmm = _build_hmm_for_config(seqdata, "A")
    posteriors = hmm.predict_proba()
    _compare_posteriors(
        posteriors, ref_posteriors["A"], "A",
        N_SEQUENCES, N_TIMEPOINTS, CONFIGS["A"]["n_states"],
    )


def test_posterior_config_B_matches_seqhmm(seqdata, ref_posteriors):
    """Config B (3 states): posterior probs match R seqHMM."""
    hmm = _build_hmm_for_config(seqdata, "B")
    posteriors = hmm.predict_proba()
    _compare_posteriors(
        posteriors, ref_posteriors["B"], "B",
        N_SEQUENCES, N_TIMEPOINTS, CONFIGS["B"]["n_states"],
    )


def test_posterior_config_C_matches_seqhmm(seqdata, ref_posteriors):
    """Config C (2 states, sticky): posterior probs match R seqHMM."""
    hmm = _build_hmm_for_config(seqdata, "C")
    posteriors = hmm.predict_proba()
    _compare_posteriors(
        posteriors, ref_posteriors["C"], "C",
        N_SEQUENCES, N_TIMEPOINTS, CONFIGS["C"]["n_states"],
    )


def test_posterior_config_D_matches_seqhmm(seqdata, ref_posteriors):
    """Config D (uniform): posterior probs match R seqHMM."""
    hmm = _build_hmm_for_config(seqdata, "D")
    posteriors = hmm.predict_proba()
    _compare_posteriors(
        posteriors, ref_posteriors["D"], "D",
        N_SEQUENCES, N_TIMEPOINTS, CONFIGS["D"]["n_states"],
    )
