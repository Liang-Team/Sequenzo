"""
@Author  : Yapeng Wei
@File    : test_seqhmm_viterbi.py
@Desc    :
Tests for Viterbi (hidden_paths) consistency between sequenzo.seqhmm
and R seqHMM (https://github.com/helske/seqHMM).

The Viterbi algorithm finds the single most probable hidden state sequence
given the observations and model parameters. Unlike posterior probabilities
(which give marginal P(state|obs) at each time point independently),
Viterbi finds the joint-optimal path.

This test verifies that Python (hmmlearn backend) and R (seqHMM C++ backend)
produce identical Viterbi paths given the same model parameters and data.

Methodology:
  - Same synthetic data: 5 sequences, 8 time points, 3 observed states.
  - Four HMM configs (A, B, C, D) with fixed parameters (no EM).
  - R seqHMM: hidden_paths(model) -> state sequence + log_prob
  - Python:   HMM.predict()       -> state index sequence (0-based)

Test groups:
  Part 0: Sanity checks (no R needed)
    - predict returns correct shape and dtype
    - predict is deterministic
    - state indices within valid range
    - Viterbi path is a valid state sequence (non-zero transitions/emissions)

  Part 1: Viterbi paths vs R seqHMM (needs ref CSVs from R script)
    - Configs A/B/C/D: exact path match (discrete, must be identical)
    - Viterbi log-probability comparison

Run seqhmm_reference_viterbi.R to generate ref_viterbi_*.csv:
  Rscript seqhmm_reference_viterbi.R .
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
# Constants (shared across all seqhmm test files -- MUST remain identical)
# ============================================================================

STATES = ["A", "B", "C"]
N_SYMBOLS = len(STATES)
SEQ_IDS = ["s0", "s1", "s2", "s3", "s4"]
TIME_COLS = ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"]
N_SEQUENCES = len(SEQ_IDS)
N_TIMEPOINTS = len(TIME_COLS)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================================
# Synthetic test data (MUST be identical to R script and other test files)
# ============================================================================

TEST_DATA_RAW = [
    ["A", "A", "B", "B", "C", "A", "A", "B"],  # s0
    ["B", "C", "C", "A", "A", "B", "C", "C"],  # s1
    ["A", "B", "C", "A", "B", "C", "A", "B"],  # s2
    ["C", "C", "B", "A", "A", "A", "B", "C"],  # s3
    ["A", "A", "A", "B", "B", "B", "C", "C"],  # s4
]


# ============================================================================
# HMM parameter configurations (MUST be identical across all test files)
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


def _reshape_viterbi(viterbi_flat, n_sequences, n_timepoints):
    """Reshape flat Viterbi array to (n_sequences, n_timepoints).

    hmmlearn predict() returns a 1D array of length total_timepoints,
    concatenated: [seq0_t0, seq0_t1, ..., seq0_t7, seq1_t0, ...].
    """
    expected_total = n_sequences * n_timepoints
    assert viterbi_flat.shape == (expected_total,), (
        f"Expected shape ({expected_total},), got {viterbi_flat.shape}"
    )
    return viterbi_flat.reshape(n_sequences, n_timepoints)


def _run_r_reference(outdir, timeout=60):
    """Run seqhmm_reference_viterbi.R to generate ref_viterbi_*.csv."""
    r_script = os.path.join(THIS_DIR, "seqhmm_reference_viterbi.R")
    if not os.path.isfile(r_script):
        return False
    try:
        result = subprocess.run(
            ["Rscript", r_script, outdir],
            capture_output=True, text=True, timeout=timeout, cwd=THIS_DIR,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _load_ref_viterbi(ref_dir, config_name):
    """Load ref_viterbi_<config>.csv.

    Returns:
        paths: np.ndarray of shape (n_sequences, n_timepoints), int, 0-based
        logprobs: np.ndarray of shape (n_sequences,), per-sequence Viterbi
                  log-probabilities.
    Returns (None, None) if file not found.
    """
    fpath = os.path.join(ref_dir, f"ref_viterbi_{config_name}.csv")
    if not os.path.isfile(fpath):
        return None, None
    df = pd.read_csv(fpath)

    df = df.sort_values(["seq_idx", "time_idx"]).reset_index(drop=True)
    n_seq = int(df["seq_idx"].max()) + 1
    n_time = int(df["time_idx"].max()) + 1

    paths = df["state_idx"].values.astype(int).reshape(n_seq, n_time)

    # Extract per-sequence Viterbi log-probabilities
    logprobs = np.array([
        df.loc[df["seq_idx"] == s, "viterbi_logprob"].iloc[0]
        for s in range(n_seq)
    ])

    return paths, logprobs


# ============================================================================
# Helper: manually compute Viterbi log-probability from path
# ============================================================================

def _compute_viterbi_logprobs(paths_2d, config):
    """Compute log P(observations, hidden_path | model) for each sequence.

    This is the joint log-probability of the observation sequence AND the
    Viterbi path, which is what R seqHMM attr(hidden_paths, "log_prob")
    returns.

    log P(Y, Q | model) = log pi(q_0) + log B(q_0, y_0)
                        + sum_{t=1}^{T-1} [log A(q_{t-1}, q_t) + log B(q_t, y_t)]
    """
    init = config["initial_probs"]
    trans = config["transition_probs"]
    emiss = config["emission_probs"]
    state_to_int = {s: i for i, s in enumerate(STATES)}

    logprobs = np.zeros(paths_2d.shape[0])
    for s in range(paths_2d.shape[0]):
        path = paths_2d[s, :]
        obs = [state_to_int[o] for o in TEST_DATA_RAW[s]]

        lp = np.log(init[path[0]]) + np.log(emiss[path[0], obs[0]])
        for t in range(1, len(path)):
            lp += np.log(trans[path[t - 1], path[t]])
            lp += np.log(emiss[path[t], obs[t]])
        logprobs[s] = lp

    return logprobs


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def seqdata():
    """SequenceData from synthetic test data."""
    return _make_test_seqdata()


@pytest.fixture(scope="module")
def ref_viterbi():
    """Load R reference Viterbi paths and log-probs for all configs.

    Returns dict: config_name -> (paths_array, logprobs_array).
    """
    # Check if ref files already exist in THIS_DIR
    refs = {}
    all_found = True
    for config_name in CONFIGS:
        paths, logprobs = _load_ref_viterbi(THIS_DIR, config_name)
        if paths is not None:
            refs[config_name] = (paths, logprobs)
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
            paths, logprobs = _load_ref_viterbi(outdir, config_name)
            if paths is not None:
                refs[config_name] = (paths, logprobs)
        if len(refs) == len(CONFIGS):
            return refs

    pytest.skip(
        "R/seqHMM not available and ref_viterbi_*.csv not found. "
        "Run: Rscript seqhmm_reference_viterbi.R . "
        "to generate reference values."
    )


# ============================================================================
# Part 0: Sanity checks (no R reference needed)
# ============================================================================

class TestViterbiSanity:
    """Sanity checks for Viterbi decoding (no R needed)."""

    def test_predict_returns_1d_array(self, seqdata):
        """predict() returns a flat 1D integer array."""
        hmm = _build_hmm_for_config(seqdata, "A")
        viterbi = hmm.predict()
        total = N_SEQUENCES * N_TIMEPOINTS
        assert viterbi.shape == (total,), (
            f"Expected shape ({total},), got {viterbi.shape}"
        )
        assert np.issubdtype(viterbi.dtype, np.integer), (
            f"Expected integer dtype, got {viterbi.dtype}"
        )

    def test_predict_shape_3states(self, seqdata):
        """Config B (3 states): predict still returns 1D array."""
        hmm = _build_hmm_for_config(seqdata, "B")
        viterbi = hmm.predict()
        total = N_SEQUENCES * N_TIMEPOINTS
        assert viterbi.shape == (total,), (
            f"Expected shape ({total},), got {viterbi.shape}"
        )

    def test_predict_deterministic(self, seqdata):
        """Same model + same data must always produce the same path."""
        hmm = _build_hmm_for_config(seqdata, "A")
        v1 = hmm.predict()
        v2 = hmm.predict()
        assert np.array_equal(v1, v2), "predict() is not deterministic"

    def test_state_indices_in_range(self, seqdata):
        """All predicted state indices must be in [0, n_states)."""
        for config_name, cfg in CONFIGS.items():
            hmm = _build_hmm_for_config(seqdata, config_name)
            viterbi = hmm.predict()
            assert np.all(viterbi >= 0), (
                f"Config {config_name}: negative state indices found"
            )
            assert np.all(viterbi < cfg["n_states"]), (
                f"Config {config_name}: state index >= n_states found. "
                f"Max index: {viterbi.max()}, n_states: {cfg['n_states']}"
            )

    def test_paths_vary_across_configs(self, seqdata):
        """Different configs should (generally) produce different paths."""
        paths = {}
        for name in ["A", "C"]:
            hmm = _build_hmm_for_config(seqdata, name)
            paths[name] = hmm.predict()
        assert not np.array_equal(paths["A"], paths["C"]), (
            "Configs A and C should produce different Viterbi paths"
        )

    def test_sticky_model_stays_in_state(self, seqdata):
        """Config C (sticky): Viterbi path should have few transitions.

        With transition_probs diagonal = 0.95, the optimal path strongly
        prefers staying in the same state. For 5 sequences x 8 timepoints
        = 40 total steps, we expect far fewer than 40 transitions.
        """
        hmm = _build_hmm_for_config(seqdata, "C")
        viterbi = hmm.predict()
        paths_2d = _reshape_viterbi(viterbi, N_SEQUENCES, N_TIMEPOINTS)

        total_transitions = 0
        for s in range(N_SEQUENCES):
            seq_path = paths_2d[s, :]
            transitions = np.sum(seq_path[1:] != seq_path[:-1])
            total_transitions += transitions

        max_possible = N_SEQUENCES * (N_TIMEPOINTS - 1)  # 35
        assert total_transitions < max_possible * 0.5, (
            f"Sticky model has {total_transitions} transitions out of "
            f"{max_possible} possible -- expected far fewer"
        )

    def test_viterbi_path_has_nonzero_probability(self, seqdata):
        """The Viterbi path must have non-zero probability under the model.

        For each transition s_t -> s_{t+1} in the Viterbi path,
        transition_probs[s_t, s_{t+1}] > 0.
        For each (s_t, obs_t), emission_probs[s_t, obs_t] > 0.
        """
        config_name = "A"
        cfg = CONFIGS[config_name]
        hmm = _build_hmm_for_config(seqdata, config_name)
        viterbi = hmm.predict()
        paths_2d = _reshape_viterbi(viterbi, N_SEQUENCES, N_TIMEPOINTS)

        trans = cfg["transition_probs"]
        emiss = cfg["emission_probs"]
        state_to_int = {s: i for i, s in enumerate(STATES)}

        for s in range(N_SEQUENCES):
            seq_path = paths_2d[s, :]
            obs_seq = [state_to_int[o] for o in TEST_DATA_RAW[s]]

            # Check emission probs
            for t in range(N_TIMEPOINTS):
                assert emiss[seq_path[t], obs_seq[t]] > 0, (
                    f"Seq {s}, t={t}: emission prob is 0 for "
                    f"state={seq_path[t]}, obs={obs_seq[t]}"
                )

            # Check transition probs
            for t in range(N_TIMEPOINTS - 1):
                assert trans[seq_path[t], seq_path[t + 1]] > 0, (
                    f"Seq {s}, t={t}->{t+1}: transition prob is 0 for "
                    f"state {seq_path[t]} -> {seq_path[t + 1]}"
                )

    def test_uniform_model_all_states_valid(self, seqdata):
        """Config D (uniform): all paths are equally likely, so any
        output from Viterbi is acceptable. Just check it runs and
        returns valid indices.
        """
        hmm = _build_hmm_for_config(seqdata, "D")
        viterbi = hmm.predict()
        n_states = CONFIGS["D"]["n_states"]
        assert np.all(viterbi >= 0) and np.all(viterbi < n_states)


# ============================================================================
# Part 1: Viterbi paths vs R seqHMM
# ============================================================================
# Viterbi is deterministic and discrete -- paths must match EXACTLY.
# No tolerance needed for the state indices.
# For Viterbi log-probabilities, we use atol=1e-6 (same as loglik tests).

LOGPROB_ATOL = 1e-6


def _assert_viterbi_paths_or_tied(py_paths, r_paths, config_name, config,
                                   atol=1e-10):
    """Assert Viterbi paths match, or if they differ, that both paths have
    equal log-probability (i.e. it is a tie-breaking difference, not a bug).

    The Viterbi algorithm is deterministic but when multiple paths share the
    same maximum joint log-probability, different implementations (R C++ vs
    Python hmmlearn Cython) may break ties differently. Both paths are
    equally optimal, so either answer is correct.
    """
    if np.array_equal(py_paths, r_paths):
        return  # exact match, nothing more to check

    # Paths differ -- compute log-probability of each per sequence
    py_lp = _compute_viterbi_logprobs(py_paths, config)
    r_lp = _compute_viterbi_logprobs(r_paths, config)

    # For every sequence where paths differ, the log-probs must be equal
    mismatches = []
    for s in range(py_paths.shape[0]):
        if not np.array_equal(py_paths[s], r_paths[s]):
            if not np.isclose(py_lp[s], r_lp[s], atol=atol):
                mismatches.append(
                    f"  seq {s}: Python path={list(py_paths[s])} "
                    f"logP={py_lp[s]:.8f}\n"
                    f"          R path     ={list(r_paths[s])} "
                    f"logP={r_lp[s]:.8f}  (diff={abs(py_lp[s]-r_lp[s]):.2e})"
                )

    if mismatches:
        detail = "\n".join(mismatches)
        pytest.fail(
            f"Config {config_name}: Viterbi paths differ AND "
            f"log-probabilities are NOT equal (not a tie):\n{detail}"
        )
    # else: paths differ but log-probs match -> tie-breaking, both correct


def test_viterbi_config_A_paths_match(seqdata, ref_viterbi):
    """Config A (2 states, basic): Viterbi paths match R, or tied."""
    hmm = _build_hmm_for_config(seqdata, "A")
    py_paths = _reshape_viterbi(hmm.predict(), N_SEQUENCES, N_TIMEPOINTS)
    r_paths, _ = ref_viterbi["A"]
    _assert_viterbi_paths_or_tied(py_paths, r_paths, "A", CONFIGS["A"])


def test_viterbi_config_B_paths_match(seqdata, ref_viterbi):
    """Config B (3 states): Viterbi paths match R, or tied."""
    hmm = _build_hmm_for_config(seqdata, "B")
    py_paths = _reshape_viterbi(hmm.predict(), N_SEQUENCES, N_TIMEPOINTS)
    r_paths, _ = ref_viterbi["B"]
    _assert_viterbi_paths_or_tied(py_paths, r_paths, "B", CONFIGS["B"])


def test_viterbi_config_C_paths_match(seqdata, ref_viterbi):
    """Config C (2 states, sticky): Viterbi paths match R, or tied."""
    hmm = _build_hmm_for_config(seqdata, "C")
    py_paths = _reshape_viterbi(hmm.predict(), N_SEQUENCES, N_TIMEPOINTS)
    r_paths, _ = ref_viterbi["C"]
    _assert_viterbi_paths_or_tied(py_paths, r_paths, "C", CONFIGS["C"])


def test_viterbi_config_D_paths_match(seqdata, ref_viterbi):
    """Config D (uniform): all paths equally likely, tie-breaking expected."""
    hmm = _build_hmm_for_config(seqdata, "D")
    py_paths = _reshape_viterbi(hmm.predict(), N_SEQUENCES, N_TIMEPOINTS)
    r_paths, _ = ref_viterbi["D"]
    _assert_viterbi_paths_or_tied(py_paths, r_paths, "D", CONFIGS["D"])


def test_viterbi_config_A_logprob(seqdata, ref_viterbi):
    """Config A: per-sequence Viterbi log-probabilities match R."""
    _, r_logprobs = ref_viterbi["A"]
    hmm = _build_hmm_for_config(seqdata, "A")
    py_paths = _reshape_viterbi(hmm.predict(), N_SEQUENCES, N_TIMEPOINTS)
    py_logprobs = _compute_viterbi_logprobs(py_paths, CONFIGS["A"])

    assert np.allclose(py_logprobs, r_logprobs, atol=LOGPROB_ATOL), (
        f"Config A Viterbi log-probs differ:\n"
        f"  Python: {py_logprobs}\n"
        f"  R:      {r_logprobs}\n"
        f"  Diff:   {np.abs(py_logprobs - r_logprobs)}"
    )


def test_viterbi_config_B_logprob(seqdata, ref_viterbi):
    """Config B: per-sequence Viterbi log-probabilities match R."""
    _, r_logprobs = ref_viterbi["B"]
    hmm = _build_hmm_for_config(seqdata, "B")
    py_paths = _reshape_viterbi(hmm.predict(), N_SEQUENCES, N_TIMEPOINTS)
    py_logprobs = _compute_viterbi_logprobs(py_paths, CONFIGS["B"])

    assert np.allclose(py_logprobs, r_logprobs, atol=LOGPROB_ATOL), (
        f"Config B Viterbi log-probs differ:\n"
        f"  Python: {py_logprobs}\n"
        f"  R:      {r_logprobs}\n"
        f"  Diff:   {np.abs(py_logprobs - r_logprobs)}"
    )


def test_viterbi_config_C_logprob(seqdata, ref_viterbi):
    """Config C: per-sequence Viterbi log-probabilities match R."""
    _, r_logprobs = ref_viterbi["C"]
    hmm = _build_hmm_for_config(seqdata, "C")
    py_paths = _reshape_viterbi(hmm.predict(), N_SEQUENCES, N_TIMEPOINTS)
    py_logprobs = _compute_viterbi_logprobs(py_paths, CONFIGS["C"])

    assert np.allclose(py_logprobs, r_logprobs, atol=LOGPROB_ATOL), (
        f"Config C Viterbi log-probs differ:\n"
        f"  Python: {py_logprobs}\n"
        f"  R:      {r_logprobs}\n"
        f"  Diff:   {np.abs(py_logprobs - r_logprobs)}"
    )
