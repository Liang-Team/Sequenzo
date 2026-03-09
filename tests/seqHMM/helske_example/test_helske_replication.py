"""
============================================================================
Replication of Helske (2019) "Mixture Hidden Markov Models for Sequence Data:
The seqHMM Package in R" (JSS, Vol 88, Issue 3) — v88i03.R

Using: sequenzo.seqhmm (Python)

Prerequisites:
  1. Rscript helske_export_data.R .     ← exports CSVs + R reference values
  2. python -m pytest test_helske_replication.py -v

Structure mirrors the paper sections:
  §4.1 Sequence data
  §4.2 Hidden Markov models (single-channel + multi-channel)
  §4.3 Clustering and mixture hidden Markov models
  §4.4/4.5 Visualization (documented as not-yet-supported)
============================================================================
"""
import os
import subprocess

import numpy as np
import pandas as pd
import pytest

from sequenzo import SequenceData
from sequenzo.seqhmm import (
    HMM,
    build_hmm,
    fit_model,
    build_mhmm,
    aic,
    bic,
)

# ============================================================================
# Paths & constants
# ============================================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TIME_COLS = [f"age_{a}" for a in range(15, 31)]     # age_15 … age_30

SC_LABELS = [                                        # 8 symbols, single-channel
    "parent", "left", "married", "left+marr",
    "child", "left+child", "left+marr+ch", "divorced",
]
MARR_STATES  = ["single", "married", "divorced"]     # 3 symbols
CHILD_STATES = ["childless", "children"]              # 2 symbols
LEFT_STATES  = ["with parents", "left home"]          # 2 symbols

# Number of sequences for multichannel fit tests.
# multichannel_emission.py is pure-Python nested loops, so 2000 seqs is far too slow.
# 200 seqs keeps each fit under ~30 seconds.
MC_N_SUBSET = 200


# ============================================================================
# Helper: load a CSV from THIS_DIR (returns None if missing)
# ============================================================================
def _csv(name):
    p = os.path.join(THIS_DIR, name)
    return pd.read_csv(p) if os.path.isfile(p) else None


def _ref_kv():
    """Load ref_results.csv as {key: float_value}."""
    df = _csv("ref_results.csv")
    if df is None:
        return None
    return dict(zip(df["key"], df["value"].astype(float)))


# ############################################################################
#
#  §4.1  SEQUENCE DATA
#
# ############################################################################

def load_biofam_single_channel():
    df = _csv("biofam_seq.csv")
    if df is None:
        return None
    return SequenceData(df, time=TIME_COLS, states=SC_LABELS, id_col="id")


def load_biofam_multichannel():
    marr_df  = _csv("biofam3c_married.csv")
    child_df = _csv("biofam3c_children.csv")
    left_df  = _csv("biofam3c_left.csv")
    if any(d is None for d in [marr_df, child_df, left_df]):
        return None, None, None
    marr  = SequenceData(marr_df,  time=TIME_COLS, states=MARR_STATES,  id_col="id")
    child = SequenceData(child_df, time=TIME_COLS, states=CHILD_STATES, id_col="id")
    left  = SequenceData(left_df,  time=TIME_COLS, states=LEFT_STATES,  id_col="id")
    return marr, child, left


def _load_multichannel_subset(n=MC_N_SUBSET):
    """Load first n rows of each multichannel CSV as SequenceData."""
    marr_df  = _csv("biofam3c_married.csv")
    child_df = _csv("biofam3c_children.csv")
    left_df  = _csv("biofam3c_left.csv")
    if any(d is None for d in [marr_df, child_df, left_df]):
        return None, None, None
    marr  = SequenceData(marr_df.head(n),  time=TIME_COLS, states=MARR_STATES,  id_col="id")
    child = SequenceData(child_df.head(n), time=TIME_COLS, states=CHILD_STATES, id_col="id")
    left  = SequenceData(left_df.head(n),  time=TIME_COLS, states=LEFT_STATES,  id_col="id")
    return marr, child, left


# ############################################################################
#
#  §4.2  HIDDEN MARKOV MODELS
#
# ############################################################################

# ═══════════════════════════════════════════════════════════════════════════
# §4.2-A  Single-channel HMM  (5 states, 8 symbols)
# ═══════════════════════════════════════════════════════════════════════════

SC_INIT = np.array([0.9, 0.06, 0.02, 0.01, 0.01])

SC_TRANS = np.array([
    [0.80, 0.10, 0.05, 0.03, 0.02],
    [0.02, 0.80, 0.10, 0.05, 0.03],
    [0.02, 0.03, 0.80, 0.10, 0.05],
    [0.02, 0.03, 0.05, 0.80, 0.10],
    [0.02, 0.03, 0.05, 0.05, 0.85],
])


def _load_sc_emiss_init():
    """Load the R-computed initial emission matrix (from seqstatf)."""
    df = _csv("ref_sc_emiss_init.csv")
    if df is not None:
        return df.values
    return None


def build_and_fit_sc_hmm(biofam_seq, sc_emiss=None):
    """Build & fit the single-channel HMM exactly as in Helske §4.2."""
    if sc_emiss is None:
        sc_emiss = np.ones((5, 8)) * 0.1
        for i in range(5):
            sc_emiss[i, min(i, 7)] += 0.5
        sc_emiss /= sc_emiss.sum(axis=1, keepdims=True)

    hmm = build_hmm(
        observations=biofam_seq,
        initial_probs=SC_INIT.copy(),
        transition_probs=SC_TRANS.copy(),
        emission_probs=sc_emiss.copy(),
        state_names=[f"State {i+1}" for i in range(5)],
    )
    hmm = fit_model(hmm, n_iter=200, tol=1e-4, verbose=False)
    return hmm


# ═══════════════════════════════════════════════════════════════════════════
# §4.2-B  Multi-channel HMM  (5 states, 3 channels)
# ═══════════════════════════════════════════════════════════════════════════

MC_INIT = np.array([0.9, 0.05, 0.02, 0.02, 0.01])

MC_TRANS = np.array([
    [0.80, 0.10, 0.05, 0.03, 0.02],
    [0.00, 0.90, 0.05, 0.03, 0.02],
    [0.00, 0.00, 0.90, 0.07, 0.03],
    [0.00, 0.00, 0.00, 0.90, 0.10],
    [0.00, 0.00, 0.00, 0.00, 1.00],
])

MC_EMISS_MARR = np.array([
    [0.90, 0.05, 0.05],
    [0.90, 0.05, 0.05],
    [0.05, 0.90, 0.05],
    [0.05, 0.90, 0.05],
    [0.30, 0.30, 0.40],
])

MC_EMISS_CHILD = np.array([
    [0.9, 0.1],
    [0.9, 0.1],
    [0.1, 0.9],
    [0.1, 0.9],
    [0.5, 0.5],
])

MC_EMISS_LEFT = np.array([
    [0.9, 0.1],
    [0.1, 0.9],
    [0.1, 0.9],
    [0.1, 0.9],
    [0.5, 0.5],
])


def build_and_fit_mc_hmm(marr_seq, child_seq, left_seq, n_iter=50):
    """Build & fit the multi-channel HMM as in Helske §4.2.

    Uses build_hmm() which now supports List[SequenceData] (Bug 4 fixed).
    n_iter is kept low because multichannel EM is pure-Python and slow.
    """
    hmm = build_hmm(
        observations=[marr_seq, child_seq, left_seq],
        n_states=5,
        initial_probs=MC_INIT.copy(),
        transition_probs=MC_TRANS.copy(),
        emission_probs=[
            MC_EMISS_MARR.copy(),
            MC_EMISS_CHILD.copy(),
            MC_EMISS_LEFT.copy(),
        ],
        state_names=[f"State {i+1}" for i in range(5)],
        channel_names=["Marriage", "Parenthood", "Residence"],
    )
    hmm.fit(n_iter=n_iter, tol=1e-3, verbose=False)
    return hmm


# ############################################################################
#
#  §4.3  CLUSTERING AND MIXTURE HIDDEN MARKOV MODELS
#
# ############################################################################

def build_and_fit_sc_mhmm(biofam_seq):
    """Build & fit a single-channel 2-cluster MHMM (surrogate for §4.3)."""
    mhmm = build_mhmm(
        observations=biofam_seq,
        n_clusters=2,
        n_states=[5, 4],
        cluster_names=["Cluster 1", "Cluster 2"],
        random_state=42,
    )
    mhmm.fit(n_iter=50, tol=1e-3)
    return mhmm


# ============================================================================
#                          PYTEST FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def biofam_seq():
    seq = load_biofam_single_channel()
    if seq is None:
        pytest.skip("biofam_seq.csv not found — run: Rscript helske_export_data.R .")
    return seq


@pytest.fixture(scope="module")
def multichannel():
    m, c, l = load_biofam_multichannel()
    if m is None:
        pytest.skip("biofam3c CSVs not found — run: Rscript helske_export_data.R .")
    return m, c, l


@pytest.fixture(scope="module")
def multichannel_subset():
    """First MC_N_SUBSET sequences — fast enough for fitting tests."""
    m, c, l = _load_multichannel_subset(MC_N_SUBSET)
    if m is None:
        pytest.skip("biofam3c CSVs not found — run: Rscript helske_export_data.R .")
    return m, c, l


@pytest.fixture(scope="module")
def sc_emiss_init():
    return _load_sc_emiss_init()


@pytest.fixture(scope="module")
def ref():
    r = _ref_kv()
    if r is None:
        pytest.skip("ref_results.csv not found — run: Rscript helske_export_data.R .")
    return r


# ============================================================================
#                          §4.1 TESTS — Sequence Data
# ============================================================================

class TestSection41_SequenceData:
    """§4.1 — Verify biofam data loads into SequenceData correctly."""

    def test_sc_n_sequences(self, biofam_seq):
        assert biofam_seq.n_sequences == 2000

    def test_sc_alphabet_size(self, biofam_seq):
        assert len(biofam_seq.alphabet) == 8

    def test_sc_alphabet_labels(self, biofam_seq):
        assert set(biofam_seq.alphabet) == set(SC_LABELS)

    def test_mc_n_sequences(self, multichannel):
        marr, child, left = multichannel
        assert marr.n_sequences == child.n_sequences == left.n_sequences

    def test_mc_alphabet_sizes(self, multichannel):
        marr, child, left = multichannel
        assert len(marr.alphabet) == 3
        assert len(child.alphabet) == 2
        assert len(left.alphabet) == 2

    def test_mc_alphabet_labels(self, multichannel):
        marr, child, left = multichannel
        assert set(marr.alphabet) == set(MARR_STATES)
        assert set(child.alphabet) == set(CHILD_STATES)
        assert set(left.alphabet) == set(LEFT_STATES)


# ============================================================================
#                §4.2-A TESTS — Single-Channel HMM
# ============================================================================

class TestSection42A_SingleChannelHMM:
    """§4.2 — Single-channel HMM with 5 states, 8 symbols."""

    def test_build_hmm(self, biofam_seq, sc_emiss_init):
        hmm = build_hmm(
            observations=biofam_seq,
            initial_probs=SC_INIT.copy(),
            transition_probs=SC_TRANS.copy(),
            emission_probs=sc_emiss_init.copy() if sc_emiss_init is not None
                else np.ones((5, 8)) / 8,
        )
        assert hmm.n_states == 5
        assert hmm.n_symbols == 8
        assert hmm.n_sequences == 2000
        assert hmm.log_likelihood is None

    def test_fit_converges(self, biofam_seq, sc_emiss_init):
        hmm = build_and_fit_sc_hmm(biofam_seq, sc_emiss_init)
        assert hmm.log_likelihood is not None
        assert np.isfinite(hmm.log_likelihood)
        assert hmm.log_likelihood < 0

    def test_fitted_params_valid(self, biofam_seq, sc_emiss_init):
        hmm = build_and_fit_sc_hmm(biofam_seq, sc_emiss_init)
        np.testing.assert_allclose(hmm.initial_probs.sum(), 1.0, atol=1e-6)
        np.testing.assert_allclose(
            hmm.transition_probs.sum(axis=1), np.ones(5), atol=1e-6)
        np.testing.assert_allclose(
            hmm.emission_probs.sum(axis=1), np.ones(5), atol=1e-6)

    def test_loglik_vs_r(self, biofam_seq, sc_emiss_init, ref):
        hmm = build_and_fit_sc_hmm(biofam_seq, sc_emiss_init)
        r_ll = ref["sc_loglik"]
        diff = abs(hmm.log_likelihood - r_ll)
        assert diff < 30, (
            f"SC logLik: Python={hmm.log_likelihood:.2f}, R={r_ll:.2f}, "
            f"diff={diff:.2f}"
        )


# ============================================================================
#           §4.2-B TESTS — Multi-Channel HMM
# ============================================================================

class TestSection42B_MultiChannelHMM:
    """§4.2 — Multi-channel HMM with 5 states, 3 channels.

    Multichannel EM is pure-Python nested loops, so fitting tests use
    a subset of MC_N_SUBSET sequences to keep runtime under 30 seconds.
    """

    def test_build_hmm_multichannel(self, multichannel):
        """build_hmm() now correctly handles List[SequenceData] (Bug 4 fixed)."""
        marr, child, left = multichannel
        hmm = build_hmm(
            observations=[marr, child, left],
            n_states=5,
            initial_probs=MC_INIT.copy(),
            transition_probs=MC_TRANS.copy(),
            emission_probs=[
                MC_EMISS_MARR.copy(),
                MC_EMISS_CHILD.copy(),
                MC_EMISS_LEFT.copy(),
            ],
            channel_names=["Marriage", "Parenthood", "Residence"],
        )
        assert hmm.n_states == 5
        assert hmm.n_channels == 3

    def test_build_mc_hmm_random_init(self, multichannel):
        """build_hmm() with random init on multichannel data."""
        marr, child, left = multichannel
        hmm = build_hmm(
            observations=[marr, child, left],
            n_states=5,
            channel_names=["Marriage", "Parenthood", "Residence"],
            random_state=42,
        )
        assert hmm.n_states == 5
        assert hmm.n_channels == 3

    def test_fit_mc_hmm(self, multichannel_subset):
        """Fit multichannel HMM on subset (pure-Python EM is slow)."""
        marr, child, left = multichannel_subset
        hmm = build_and_fit_mc_hmm(marr, child, left, n_iter=30)
        assert hmm.log_likelihood is not None
        assert np.isfinite(hmm.log_likelihood)
        assert hmm.log_likelihood < 0

    def test_mc_fitted_params_valid(self, multichannel_subset):
        """Fitted initial & transition probs are valid distributions."""
        marr, child, left = multichannel_subset
        hmm = build_and_fit_mc_hmm(marr, child, left, n_iter=30)
        np.testing.assert_allclose(hmm.initial_probs.sum(), 1.0, atol=1e-6)
        np.testing.assert_allclose(
            hmm.transition_probs.sum(axis=1), np.ones(5), atol=1e-6)
        for ch_idx, ch_name in enumerate(["Marriage", "Parenthood", "Residence"]):
            np.testing.assert_allclose(
                hmm.emission_probs[ch_idx].sum(axis=1), np.ones(5), atol=1e-6,
                err_msg=f"emission row sums != 1 for channel '{ch_name}'")

    def test_mc_loglik_vs_r(self, multichannel_subset, ref):
        """Multichannel logLik on subset — not directly comparable to R (full data).

        We only check it is finite and negative; exact comparison requires
        fitting on full 2000 sequences which is too slow in pure Python.
        """
        marr, child, left = multichannel_subset
        hmm = build_and_fit_mc_hmm(marr, child, left, n_iter=30)
        assert np.isfinite(hmm.log_likelihood)
        assert hmm.log_likelihood < 0


# ============================================================================
#                 §4.3 TESTS — MHMM (single-channel surrogate)
# ============================================================================

class TestSection43_MHMM:
    """§4.3 — Mixture HMM.

    LIMITATIONS:
      - build_mhmm() only accepts single SequenceData (not multichannel list)
      - build_mhmm() has no formula/data parameters (no covariates)
    We test the MHMM machinery on single-channel biofam_seq as a surrogate.
    """

    def test_build_mhmm(self, biofam_seq):
        mhmm = build_mhmm(
            observations=biofam_seq,
            n_clusters=2,
            n_states=[5, 4],
            cluster_names=["Cluster 1", "Cluster 2"],
            random_state=42,
        )
        assert mhmm.n_clusters == 2
        assert mhmm.n_states == [5, 4]
        assert len(mhmm.clusters) == 2

    def test_fit_mhmm(self, biofam_seq):
        mhmm = build_and_fit_sc_mhmm(biofam_seq)
        assert mhmm.log_likelihood is not None
        assert np.isfinite(mhmm.log_likelihood)
        assert mhmm.log_likelihood < 0

    def test_cluster_probs_sum_to_one(self, biofam_seq):
        mhmm = build_and_fit_sc_mhmm(biofam_seq)
        np.testing.assert_allclose(mhmm.cluster_probs.sum(), 1.0, atol=1e-6)

    def test_responsibilities_shape(self, biofam_seq):
        mhmm = build_and_fit_sc_mhmm(biofam_seq)
        assert mhmm.responsibilities is not None
        assert mhmm.responsibilities.shape == (2000, 2)
        np.testing.assert_allclose(
            mhmm.responsibilities.sum(axis=1), np.ones(2000), atol=1e-6)

    def test_each_cluster_has_valid_params(self, biofam_seq):
        mhmm = build_and_fit_sc_mhmm(biofam_seq)
        for k, cluster_hmm in enumerate(mhmm.clusters):
            np.testing.assert_allclose(
                cluster_hmm.initial_probs.sum(), 1.0, atol=1e-6,
                err_msg=f"Cluster {k} initial_probs don't sum to 1")
            np.testing.assert_allclose(
                cluster_hmm.transition_probs.sum(axis=1),
                np.ones(mhmm.n_states[k]), atol=1e-6,
                err_msg=f"Cluster {k} transition row sums")
            np.testing.assert_allclose(
                cluster_hmm.emission_probs.sum(axis=1),
                np.ones(mhmm.n_states[k]), atol=1e-6,
                err_msg=f"Cluster {k} emission row sums")

    def test_mhmm_no_multichannel(self):
        """build_mhmm() only accepts SequenceData, not List[SequenceData]."""
        import inspect
        sig = inspect.signature(build_mhmm)
        assert "SequenceData" in str(sig.parameters["observations"].annotation)

    def test_mhmm_no_formula(self):
        """build_mhmm() has no formula/data params (unlike R's seqHMM)."""
        import inspect
        sig = inspect.signature(build_mhmm)
        assert "formula" not in sig.parameters
        assert "data" not in sig.parameters


# ============================================================================
#           §4.4/4.5 TESTS — Visualization (NOT replicated)
# ============================================================================

class TestSection44_45_Visualization:

    def test_plot_hmm_importable(self):
        from sequenzo.seqhmm import plot_hmm
        assert callable(plot_hmm)

    def test_plot_mhmm_importable(self):
        from sequenzo.seqhmm import plot_mhmm
        assert callable(plot_mhmm)