"""
@Author  : Yapeng Wei
@File    : test_seqhmm_simulate.py
@Desc    :
Tests for HMM/MHMM simulation consistency between sequenzo.seqhmm
and R seqHMM.

Actual Python API:
  simulate_hmm(n_sequences, initial_probs, transition_probs, emission_probs,
               sequence_length, alphabet=, state_names=, random_state=) -> dict
    Returns: {'observations': list, 'states': list, 'observations_df': DataFrame,
              'alphabet': list, 'state_names': list}

  simulate_mhmm(n_sequences, n_clusters, initial_probs, transition_probs,
                emission_probs, cluster_probs=, sequence_length=, alphabet=,
                state_names=, cluster_names=, formula=, data=, coefficients=,
                random_state=) -> dict
    Returns: {'observations': list, 'states': list, 'clusters': list,
              'observations_df': DataFrame, ...}

Since R and Python use different RNGs, we compare statistical properties.

Test groups:
  Part 0: Sanity checks (no R)
  Part 1: Cross-language statistical comparison (needs ref CSVs)

Run: Rscript seqhmm_reference_simulate.R .
"""
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd
import pytest

from sequenzo.seqhmm import simulate_hmm, simulate_mhmm


# ============================================================================
# Constants
# ============================================================================

STATES = ["A", "B", "C"]
N_SYMBOLS = len(STATES)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# HMM simulation parameters
SIM_INIT = np.array([0.6, 0.4])
SIM_TRANS = np.array([
    [0.7, 0.3],
    [0.2, 0.8],
])
SIM_EMISS = np.array([
    [0.5, 0.3, 0.2],
    [0.1, 0.4, 0.5],
])
SIM_N_SEQ = 100
SIM_SEQ_LEN = 20

# MHMM simulation parameters
MHMM_INIT_1 = np.array([0.8, 0.2])
MHMM_TRANS_1 = np.array([[0.9, 0.1], [0.2, 0.8]])
MHMM_EMISS_1 = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])

MHMM_INIT_2 = np.array([0.3, 0.7])
MHMM_TRANS_2 = np.array([[0.6, 0.4], [0.1, 0.9]])
MHMM_EMISS_2 = np.array([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]])

MHMM_CLUSTER_PROBS = np.array([0.6, 0.4])

# Tolerance for statistical tests
FREQ_ATOL = 0.10
TRANS_ATOL = 0.12


# ============================================================================
# Helpers
# ============================================================================

def _compute_stationary(trans):
    """Compute stationary distribution of a transition matrix."""
    n = trans.shape[0]
    A = np.vstack([trans.T - np.eye(n), np.ones(n)])
    b = np.zeros(n + 1)
    b[-1] = 1.0
    pi, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return pi


def _empirical_transition_probs(state_sequences, state_names, n_states):
    """Compute empirical transition probs from string-encoded state sequences."""
    name_to_idx = {name: i for i, name in enumerate(state_names)}
    counts = np.zeros((n_states, n_states))
    for seq in state_sequences:
        for t in range(len(seq) - 1):
            i = name_to_idx.get(seq[t], seq[t])
            j = name_to_idx.get(seq[t + 1], seq[t + 1])
            if isinstance(i, int) and isinstance(j, int):
                counts[i, j] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return counts / row_sums


def _run_r_reference(outdir, timeout=120):
    r_script = os.path.join(THIS_DIR, "seqhmm_reference_simulate.R")
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


def _load_ref_kv(ref_dir, filename):
    fpath = os.path.join(ref_dir, filename)
    if not os.path.isfile(fpath):
        return None
    df = pd.read_csv(fpath)
    return dict(zip(df["key"], df["value"]))


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def ref_sim():
    refs = {}
    files = {
        "hmm_stats": "ref_sim_hmm_stats.csv",
        "mhmm_stats": "ref_sim_mhmm_stats.csv",
    }

    all_found = True
    for key, fname in files.items():
        ref = _load_ref_kv(THIS_DIR, fname)
        if ref is not None:
            refs[key] = ref
        else:
            all_found = False
            break

    if all_found:
        return refs

    outdir = tempfile.mkdtemp()
    ok = _run_r_reference(outdir)
    if ok:
        refs = {}
        for key, fname in files.items():
            ref = _load_ref_kv(outdir, fname)
            if ref is not None:
                refs[key] = ref
        if len(refs) == len(files):
            return refs

    pytest.skip(
        "R/seqHMM not available and ref_sim_*.csv not found. "
        "Run: Rscript seqhmm_reference_simulate.R ."
    )


# ============================================================================
# Part 0: Sanity checks
# ============================================================================

class TestSimulateHMMSanity:
    """Sanity checks for simulate_hmm."""

    def test_returns_observations_and_states(self):
        """simulate_hmm returns dict with observations and states."""
        result = simulate_hmm(
            n_sequences=10,
            initial_probs=SIM_INIT.copy(),
            transition_probs=SIM_TRANS.copy(),
            emission_probs=SIM_EMISS.copy(),
            sequence_length=20,
            alphabet=STATES,
            random_state=42,
        )
        assert "observations" in result
        assert "states" in result

    def test_output_dimensions(self):
        """Simulated sequences have correct dimensions."""
        n_seq, seq_len = 15, 25
        result = simulate_hmm(
            n_sequences=n_seq,
            initial_probs=SIM_INIT.copy(),
            transition_probs=SIM_TRANS.copy(),
            emission_probs=SIM_EMISS.copy(),
            sequence_length=seq_len,
            alphabet=STATES,
            random_state=42,
        )
        obs = result["observations"]
        assert len(obs) == n_seq
        for s in obs:
            assert len(s) == seq_len

    def test_valid_symbols(self):
        """All simulated observations belong to valid alphabet."""
        result = simulate_hmm(
            n_sequences=50,
            initial_probs=SIM_INIT.copy(),
            transition_probs=SIM_TRANS.copy(),
            emission_probs=SIM_EMISS.copy(),
            sequence_length=20,
            alphabet=STATES,
            random_state=42,
        )
        for seq in result["observations"]:
            for s in seq:
                assert s in STATES, f"Invalid symbol: {s}"

    def test_observations_df_present(self):
        """Result includes observations_df DataFrame."""
        result = simulate_hmm(
            n_sequences=10,
            initial_probs=SIM_INIT.copy(),
            transition_probs=SIM_TRANS.copy(),
            emission_probs=SIM_EMISS.copy(),
            sequence_length=20,
            alphabet=STATES,
            random_state=42,
        )
        assert "observations_df" in result
        assert isinstance(result["observations_df"], pd.DataFrame)
        assert result["observations_df"].shape[0] == 10

    def test_reproducibility_with_seed(self):
        """Same random_state produces identical sequences."""
        kwargs = dict(
            n_sequences=20,
            initial_probs=SIM_INIT.copy(),
            transition_probs=SIM_TRANS.copy(),
            emission_probs=SIM_EMISS.copy(),
            sequence_length=15,
            alphabet=STATES,
            random_state=12345,
        )
        r1 = simulate_hmm(**kwargs)
        r2 = simulate_hmm(**kwargs)
        assert r1["observations"] == r2["observations"]

    def test_different_seeds_differ(self):
        """Different random_state produces different sequences."""
        kwargs = dict(
            n_sequences=20,
            initial_probs=SIM_INIT.copy(),
            transition_probs=SIM_TRANS.copy(),
            emission_probs=SIM_EMISS.copy(),
            sequence_length=15,
            alphabet=STATES,
        )
        r1 = simulate_hmm(random_state=1, **kwargs)
        r2 = simulate_hmm(random_state=9999, **kwargs)
        assert r1["observations"] != r2["observations"]

    def test_state_frequency_convergence(self):
        """With many samples, state frequencies approach stationary dist."""
        result = simulate_hmm(
            n_sequences=500,
            initial_probs=SIM_INIT.copy(),
            transition_probs=SIM_TRANS.copy(),
            emission_probs=SIM_EMISS.copy(),
            sequence_length=50,
            state_names=["S0", "S1"],
            random_state=42,
        )
        states = result["states"]
        states_flat = [s for seq in states for s in seq]

        n_states = SIM_TRANS.shape[0]
        state_names = result["state_names"]
        emp_freq = np.zeros(n_states)
        for i, name in enumerate(state_names):
            emp_freq[i] = sum(1 for s in states_flat if s == name) / len(states_flat)

        stat_dist = _compute_stationary(SIM_TRANS)
        assert np.allclose(emp_freq, stat_dist, atol=FREQ_ATOL), (
            f"State frequencies {emp_freq} don't match "
            f"stationary distribution {stat_dist}"
        )

    def test_transition_count_convergence(self):
        """With many samples, empirical transition probs approach true probs."""
        result = simulate_hmm(
            n_sequences=500,
            initial_probs=SIM_INIT.copy(),
            transition_probs=SIM_TRANS.copy(),
            emission_probs=SIM_EMISS.copy(),
            sequence_length=50,
            state_names=["S0", "S1"],
            random_state=42,
        )
        state_seqs = result["states"]
        state_names = result["state_names"]
        emp_trans = _empirical_transition_probs(
            state_seqs, state_names, n_states=SIM_TRANS.shape[0]
        )
        assert np.allclose(emp_trans, SIM_TRANS, atol=TRANS_ATOL), (
            f"Empirical transitions\n{emp_trans}\n"
            f"don't match true transitions\n{SIM_TRANS}"
        )


class TestSimulateMHMMSanity:
    """Sanity checks for simulate_mhmm."""

    def test_returns_result(self):
        """simulate_mhmm returns a dict."""
        result = simulate_mhmm(
            n_sequences=20,
            n_clusters=2,
            initial_probs=[MHMM_INIT_1.copy(), MHMM_INIT_2.copy()],
            transition_probs=[MHMM_TRANS_1.copy(), MHMM_TRANS_2.copy()],
            emission_probs=[MHMM_EMISS_1.copy(), MHMM_EMISS_2.copy()],
            cluster_probs=MHMM_CLUSTER_PROBS.copy(),
            sequence_length=15,
            alphabet=STATES,
            random_state=42,
        )
        assert result is not None
        assert "observations" in result
        assert "clusters" in result

    def test_output_dimensions(self):
        """MHMM simulated sequences have correct dimensions."""
        n_seq, seq_len = 30, 20
        result = simulate_mhmm(
            n_sequences=n_seq,
            n_clusters=2,
            initial_probs=[MHMM_INIT_1.copy(), MHMM_INIT_2.copy()],
            transition_probs=[MHMM_TRANS_1.copy(), MHMM_TRANS_2.copy()],
            emission_probs=[MHMM_EMISS_1.copy(), MHMM_EMISS_2.copy()],
            cluster_probs=MHMM_CLUSTER_PROBS.copy(),
            sequence_length=seq_len,
            alphabet=STATES,
            random_state=42,
        )
        obs = result["observations"]
        assert len(obs) == n_seq
        assert len(result["clusters"]) == n_seq

    def test_reproducibility_with_seed(self):
        """Same random_state produces identical MHMM simulations."""
        kwargs = dict(
            n_sequences=20,
            n_clusters=2,
            initial_probs=[MHMM_INIT_1.copy(), MHMM_INIT_2.copy()],
            transition_probs=[MHMM_TRANS_1.copy(), MHMM_TRANS_2.copy()],
            emission_probs=[MHMM_EMISS_1.copy(), MHMM_EMISS_2.copy()],
            cluster_probs=MHMM_CLUSTER_PROBS.copy(),
            sequence_length=15,
            alphabet=STATES,
            random_state=999,
        )
        r1 = simulate_mhmm(**kwargs)
        r2 = simulate_mhmm(**kwargs)
        assert r1["observations"] == r2["observations"]
        assert r1["clusters"] == r2["clusters"]

    def test_cluster_assignments_valid(self):
        """All cluster assignments belong to cluster_names."""
        result = simulate_mhmm(
            n_sequences=50,
            n_clusters=2,
            initial_probs=[MHMM_INIT_1.copy(), MHMM_INIT_2.copy()],
            transition_probs=[MHMM_TRANS_1.copy(), MHMM_TRANS_2.copy()],
            emission_probs=[MHMM_EMISS_1.copy(), MHMM_EMISS_2.copy()],
            cluster_probs=MHMM_CLUSTER_PROBS.copy(),
            sequence_length=15,
            alphabet=STATES,
            cluster_names=["C1", "C2"],
            random_state=42,
        )
        valid_names = set(result["cluster_names"])
        for c in result["clusters"]:
            assert c in valid_names, f"Invalid cluster: {c}"


# ============================================================================
# Part 1: Cross-language statistical comparison
# ============================================================================

def test_sim_hmm_symbol_freq_similar_to_r(ref_sim):
    """HMM simulation symbol frequencies are statistically similar to R's."""
    result = simulate_hmm(
        n_sequences=SIM_N_SEQ,
        initial_probs=SIM_INIT.copy(),
        transition_probs=SIM_TRANS.copy(),
        emission_probs=SIM_EMISS.copy(),
        sequence_length=SIM_SEQ_LEN,
        alphabet=STATES,
        random_state=42,
    )
    obs_flat = [s for seq in result["observations"] for s in seq]

    py_freq = {}
    for sym in STATES:
        py_freq[sym] = sum(1 for s in obs_flat if s == sym) / len(obs_flat)

    r_stats = ref_sim["hmm_stats"]
    for sym in STATES:
        r_freq = r_stats.get(f"symbol_freq_{sym}", None)
        if r_freq is not None:
            assert abs(py_freq[sym] - float(r_freq)) < 0.15, (
                f"Symbol '{sym}' freq: Python={py_freq[sym]:.3f}, R={r_freq}"
            )


def test_sim_hmm_transition_probs_similar_to_r(ref_sim):
    """HMM simulation empirical transition probs match R's approximately."""
    result = simulate_hmm(
        n_sequences=SIM_N_SEQ,
        initial_probs=SIM_INIT.copy(),
        transition_probs=SIM_TRANS.copy(),
        emission_probs=SIM_EMISS.copy(),
        sequence_length=SIM_SEQ_LEN,
        state_names=["S0", "S1"],
        random_state=42,
    )
    state_names = result["state_names"]
    py_emp = _empirical_transition_probs(
        result["states"], state_names, n_states=2
    )

    r_stats = ref_sim["hmm_stats"]
    for i in range(2):
        for j in range(2):
            r_val = r_stats.get(f"emp_trans_{i}_{j}", None)
            if r_val is not None:
                assert abs(py_emp[i, j] - float(r_val)) < 0.15, (
                    f"Trans[{i},{j}]: Python={py_emp[i,j]:.3f}, R={r_val}"
                )
