"""
@Author  : Yapeng Wei
@File    : test_seqhmm_mhmm.py
@Desc    :
Tests for Mixture Hidden Markov Model (MHMM) consistency between
sequenzo.seqhmm and R seqHMM (https://github.com/helske/seqHMM).

MHMM extends HMM by allowing multiple "clusters" of sequences, each
governed by its own set of HMM parameters (initial, transition, emission).

Actual MHMM API (from sequenzo.seqhmm.mhmm):
  - mhmm.n_clusters         number of clusters
  - mhmm.n_states           list of n_states per cluster
  - mhmm.clusters           list of HMM objects (one per cluster)
  - mhmm.cluster_probs      mixture probabilities (numpy array)
  - mhmm.log_likelihood     log-likelihood (None until fit())
  - mhmm.responsibilities   posterior cluster probs (None until fit())
  - mhmm.converged          convergence flag (None until fit())
  - mhmm.n_iter             number of iterations (None until fit())
  - mhmm.fit(n_iter, tol)   EM fitting, returns self
  - mhmm.predict_cluster()  predict cluster membership (requires fit())

  Each mhmm.clusters[k] is an HMM with:
    .initial_probs, .transition_probs, .emission_probs
    .score()                 log-likelihood of data under this HMM
    ._hmm_model.score(X, lengths)

Test groups:
  Part 0: Sanity checks (no R needed)
  Part 1: Cross-language consistency (needs ref CSVs from R script)

Run seqhmm_reference_mhmm.R to generate reference CSVs:
  Rscript seqhmm_reference_mhmm.R .
"""
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd
import pytest

from sequenzo import SequenceData
from sequenzo.seqhmm import build_mhmm
from sequenzo.seqhmm.utils import sequence_data_to_hmmlearn_format


# ============================================================================
# Constants
# ============================================================================

STATES = ["A", "B", "C"]
N_SYMBOLS = len(STATES)
SEQ_IDS = [f"s{i}" for i in range(10)]
TIME_COLS = ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"]
N_SEQUENCES = len(SEQ_IDS)
N_TIMEPOINTS = len(TIME_COLS)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================================
# Synthetic test data: 10 sequences × 8 time points × 3 states
# Must exactly match seqhmm_reference_mhmm.R
# ============================================================================

TEST_DATA_RAW = [
    ["A", "A", "B", "B", "C", "A", "A", "B"],  # s0
    ["B", "C", "C", "A", "A", "B", "C", "C"],  # s1
    ["A", "B", "C", "A", "B", "C", "A", "B"],  # s2
    ["C", "C", "B", "A", "A", "A", "B", "C"],  # s3
    ["A", "A", "A", "B", "B", "B", "C", "C"],  # s4
    ["C", "C", "B", "A", "A", "C", "B", "C"],  # s5
    ["C", "B", "C", "A", "B", "C", "A", "B"],  # s6
    ["B", "B", "C", "C", "B", "A", "B", "C"],  # s7
    ["A", "A", "B", "B", "C", "A", "A", "B"],  # s8
    ["C", "B", "A", "A", "B", "C", "A", "B"],  # s9
]


# ============================================================================
# MHMM parameter configs (must match R script exactly)
# ============================================================================

CLUSTER1 = {
    "initial_probs": np.array([0.7, 0.3]),
    "transition_probs": np.array([
        [0.8, 0.2],
        [0.3, 0.7],
    ]),
    "emission_probs": np.array([
        [0.6, 0.3, 0.1],
        [0.1, 0.4, 0.5],
    ]),
}

CLUSTER2 = {
    "initial_probs": np.array([0.4, 0.6]),
    "transition_probs": np.array([
        [0.6, 0.4],
        [0.2, 0.8],
    ]),
    "emission_probs": np.array([
        [0.1, 0.3, 0.6],
        [0.5, 0.4, 0.1],
    ]),
}

N_CLUSTERS = 2
N_STATES_PER_CLUSTER = 2
CLUSTER_NAMES = ["Cluster1", "Cluster2"]

# Tolerances
LOGLIK_ATOL = 1e-4       # pre-EM loglik (deterministic)
LOGLIK_EM_ATOL = 1.0     # post-EM loglik (different EM implementations)

# EM settings
EM_N_ITER = 200
EM_TOL = 1e-6


# ============================================================================
# Helpers
# ============================================================================

def _make_test_seqdata():
    """Create SequenceData from synthetic test data."""
    df = pd.DataFrame(TEST_DATA_RAW, columns=TIME_COLS)
    df.insert(0, "id", SEQ_IDS)
    return SequenceData(df, time=TIME_COLS, states=STATES, id_col="id")


def _build_test_mhmm(seqdata):
    """Build MHMM with 2 clusters, each 2 states."""
    return build_mhmm(
        observations=seqdata,
        n_clusters=N_CLUSTERS,
        n_states=N_STATES_PER_CLUSTER,
        initial_probs=[
            CLUSTER1["initial_probs"].copy(),
            CLUSTER2["initial_probs"].copy(),
        ],
        transition_probs=[
            CLUSTER1["transition_probs"].copy(),
            CLUSTER2["transition_probs"].copy(),
        ],
        emission_probs=[
            CLUSTER1["emission_probs"].copy(),
            CLUSTER2["emission_probs"].copy(),
        ],
        cluster_names=CLUSTER_NAMES,
    )


def _compute_mhmm_loglik(mhmm):
    """Compute mixture log-likelihood from cluster HMMs.

    logLik = sum_i log( sum_k P(cluster_k) * P(seq_i | HMM_k) )

    This mirrors R's logLik(mhmm) on an unfitted MHMM model.
    """
    X, lengths = sequence_data_to_hmmlearn_format(mhmm.observations)
    n_sequences = len(lengths)

    log_likelihoods = np.zeros((n_sequences, mhmm.n_clusters))

    for k in range(mhmm.n_clusters):
        for seq_idx in range(n_sequences):
            start_idx = int(lengths[:seq_idx].sum())
            end_idx = start_idx + int(lengths[seq_idx])
            seq_X = X[start_idx:end_idx]
            seq_lengths = np.array([int(lengths[seq_idx])])
            log_likelihoods[seq_idx, k] = mhmm.clusters[k]._hmm_model.score(
                seq_X, seq_lengths
            )

    # Add log cluster probabilities
    log_cluster_probs = np.log(mhmm.cluster_probs + 1e-300)
    log_joint = log_likelihoods + log_cluster_probs[np.newaxis, :]

    # log-sum-exp over clusters, then sum over sequences
    max_log = np.max(log_joint, axis=1, keepdims=True)
    log_marginal = max_log.squeeze() + np.log(
        np.sum(np.exp(log_joint - max_log), axis=1)
    )
    return float(np.sum(log_marginal))


def _run_r_reference(outdir, timeout=120):
    """Run seqhmm_reference_mhmm.R to generate reference CSVs."""
    r_script = os.path.join(THIS_DIR, "seqhmm_reference_mhmm.R")
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
    """Load a reference key-value CSV as dict."""
    fpath = os.path.join(ref_dir, filename)
    if not os.path.isfile(fpath):
        return None
    df = pd.read_csv(fpath)
    if "key" in df.columns and "value" in df.columns:
        return dict(zip(df["key"], df["value"]))
    return None


def _load_ref_csv(ref_dir, filename):
    """Load a reference CSV as DataFrame."""
    fpath = os.path.join(ref_dir, filename)
    if not os.path.isfile(fpath):
        return None
    return pd.read_csv(fpath)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def seqdata():
    """SequenceData from synthetic test data."""
    return _make_test_seqdata()


@pytest.fixture(scope="module")
def ref_mhmm():
    """Load R reference MHMM results.

    Returns dict with keys: 'loglik', 'em', 'hidden_paths', 'cluster'.
    Each value is a dict (key-value CSV) or DataFrame.
    """
    refs = {}
    kv_files = {
        "loglik": "ref_mhmm_loglik.csv",
        "em": "ref_mhmm_em.csv",
    }
    df_files = {
        "hidden_paths": "ref_mhmm_hidden_paths.csv",
        "cluster": "ref_mhmm_cluster.csv",
    }

    all_found = True

    for key, fname in kv_files.items():
        ref = _load_ref_kv(THIS_DIR, fname)
        if ref is not None:
            refs[key] = ref
        else:
            all_found = False
            break

    if all_found:
        for key, fname in df_files.items():
            ref = _load_ref_csv(THIS_DIR, fname)
            if ref is not None:
                refs[key] = ref
            else:
                all_found = False
                break

    if all_found:
        return refs

    # Try running R script
    outdir = tempfile.mkdtemp()
    ok = _run_r_reference(outdir)
    if ok:
        refs = {}
        for key, fname in kv_files.items():
            ref = _load_ref_kv(outdir, fname)
            if ref is not None:
                refs[key] = ref
        for key, fname in df_files.items():
            ref = _load_ref_csv(outdir, fname)
            if ref is not None:
                refs[key] = ref
        if len(refs) == len(kv_files) + len(df_files):
            return refs

    pytest.skip(
        "R/seqHMM not available and ref_mhmm_*.csv not found. "
        "Run: Rscript seqhmm_reference_mhmm.R . "
        "to generate reference values."
    )


# ============================================================================
# Part 0: Sanity checks (no R needed)
# ============================================================================

class TestMHMMSanity:
    """Sanity checks for Mixture HMM (no R needed)."""

    def test_build_mhmm_returns_object(self, seqdata):
        """build_mhmm returns a valid MHMM object."""
        mhmm = _build_test_mhmm(seqdata)
        assert mhmm is not None

    def test_n_clusters(self, seqdata):
        """MHMM has the correct number of clusters."""
        mhmm = _build_test_mhmm(seqdata)
        assert mhmm.n_clusters == N_CLUSTERS

    def test_n_states_per_cluster(self, seqdata):
        """Each cluster has the correct number of hidden states."""
        mhmm = _build_test_mhmm(seqdata)
        assert len(mhmm.n_states) == N_CLUSTERS
        for k in range(N_CLUSTERS):
            assert mhmm.n_states[k] == N_STATES_PER_CLUSTER

    def test_clusters_list_matches(self, seqdata):
        """mhmm.clusters is a list of HMM objects matching n_clusters."""
        mhmm = _build_test_mhmm(seqdata)
        assert len(mhmm.clusters) == N_CLUSTERS
        for k in range(N_CLUSTERS):
            assert mhmm.clusters[k] is not None

    def test_cluster_probs_valid(self, seqdata):
        """Cluster probabilities sum to 1 and are non-negative."""
        mhmm = _build_test_mhmm(seqdata)
        assert len(mhmm.cluster_probs) == N_CLUSTERS
        assert np.all(mhmm.cluster_probs >= 0), "Negative cluster probs"
        assert np.isclose(mhmm.cluster_probs.sum(), 1.0, atol=1e-6), (
            f"cluster_probs sum = {mhmm.cluster_probs.sum()}"
        )

    def test_mixture_loglik_computable(self, seqdata):
        """Mixture logLik can be computed from cluster HMMs before fitting."""
        mhmm = _build_test_mhmm(seqdata)
        ll = _compute_mhmm_loglik(mhmm)
        assert np.isfinite(ll), f"Mixture logLik is not finite: {ll}"
        assert ll < 0, f"Mixture logLik should be negative: {ll}"

    def test_cluster_hmm_parameters_valid(self, seqdata):
        """Each cluster HMM's parameters are valid probability distributions."""
        mhmm = _build_test_mhmm(seqdata)
        for k in range(N_CLUSTERS):
            hmm_k = mhmm.clusters[k]

            # Initial probs
            ip = hmm_k.initial_probs
            assert ip is not None, f"Cluster {k}: initial_probs is None"
            assert np.all(ip >= -1e-10), f"Cluster {k}: negative initial_probs"
            assert np.isclose(ip.sum(), 1.0, atol=1e-6), (
                f"Cluster {k}: initial_probs sum = {ip.sum()}"
            )

            # Transition probs
            tp = hmm_k.transition_probs
            assert tp is not None, f"Cluster {k}: transition_probs is None"
            assert np.all(tp >= -1e-10), f"Cluster {k}: negative transition_probs"
            assert np.allclose(tp.sum(axis=1), 1.0, atol=1e-6), (
                f"Cluster {k}: transition_probs row sums = {tp.sum(axis=1)}"
            )

            # Emission probs
            ep = hmm_k.emission_probs
            assert ep is not None, f"Cluster {k}: emission_probs is None"
            assert np.all(ep >= -1e-10), f"Cluster {k}: negative emission_probs"
            assert np.allclose(ep.sum(axis=1), 1.0, atol=1e-6), (
                f"Cluster {k}: emission_probs row sums = {ep.sum(axis=1)}"
            )

    def test_unfitted_state(self, seqdata):
        """Before fit(), log_likelihood/responsibilities/converged are None."""
        mhmm = _build_test_mhmm(seqdata)
        assert mhmm.log_likelihood is None
        assert mhmm.responsibilities is None
        assert mhmm.converged is None

    def test_fit_returns_self(self, seqdata):
        """fit() returns self for method chaining."""
        mhmm = _build_test_mhmm(seqdata)
        result = mhmm.fit(n_iter=10, tol=1e-2)
        assert result is mhmm

    def test_fit_sets_log_likelihood(self, seqdata):
        """After fit(), log_likelihood is a finite negative number."""
        mhmm = _build_test_mhmm(seqdata)
        mhmm.fit(n_iter=50, tol=EM_TOL)
        assert mhmm.log_likelihood is not None
        assert np.isfinite(mhmm.log_likelihood), (
            f"log_likelihood not finite: {mhmm.log_likelihood}"
        )
        assert mhmm.log_likelihood < 0, (
            f"log_likelihood should be negative: {mhmm.log_likelihood}"
        )

    def test_fit_sets_responsibilities(self, seqdata):
        """After fit(), responsibilities shape is (n_sequences, n_clusters)."""
        mhmm = _build_test_mhmm(seqdata)
        mhmm.fit(n_iter=50, tol=EM_TOL)
        assert mhmm.responsibilities is not None
        assert mhmm.responsibilities.shape == (N_SEQUENCES, N_CLUSTERS), (
            f"responsibilities shape: expected ({N_SEQUENCES}, {N_CLUSTERS}), "
            f"got {mhmm.responsibilities.shape}"
        )
        # Each row should sum to 1
        row_sums = mhmm.responsibilities.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), (
            f"responsibilities row sums: {row_sums}"
        )

    def test_em_improves_loglik(self, seqdata):
        """EM fitting should improve (or maintain) log-likelihood."""
        mhmm = _build_test_mhmm(seqdata)
        ll_before = _compute_mhmm_loglik(mhmm)
        mhmm.fit(n_iter=EM_N_ITER, tol=EM_TOL)
        ll_after = mhmm.log_likelihood
        assert ll_after >= ll_before - 1e-4, (
            f"logLik decreased after EM: {ll_before:.4f} -> {ll_after:.4f}"
        )

    def test_em_parameters_valid_after_fitting(self, seqdata):
        """After EM, all cluster parameters remain valid distributions."""
        mhmm = _build_test_mhmm(seqdata)
        mhmm.fit(n_iter=EM_N_ITER, tol=EM_TOL)

        # Cluster probs valid
        assert np.all(mhmm.cluster_probs >= 0)
        assert np.isclose(mhmm.cluster_probs.sum(), 1.0, atol=1e-6)

        # Each cluster HMM valid
        for k in range(N_CLUSTERS):
            hmm_k = mhmm.clusters[k]
            ip = hmm_k.initial_probs
            assert np.all(ip >= -1e-10) and np.isclose(ip.sum(), 1.0, atol=1e-6), (
                f"Cluster {k}: invalid initial_probs after EM"
            )
            tp = hmm_k.transition_probs
            assert np.all(tp >= -1e-10) and np.allclose(tp.sum(axis=1), 1.0, atol=1e-6), (
                f"Cluster {k}: invalid transition_probs after EM"
            )
            ep = hmm_k.emission_probs
            assert np.all(ep >= -1e-10) and np.allclose(ep.sum(axis=1), 1.0, atol=1e-6), (
                f"Cluster {k}: invalid emission_probs after EM"
            )

    def test_em_deterministic(self, seqdata):
        """Same initial params + same data = same EM result."""
        mhmm1 = _build_test_mhmm(seqdata)
        mhmm1.fit(n_iter=EM_N_ITER, tol=EM_TOL)

        mhmm2 = _build_test_mhmm(seqdata)
        mhmm2.fit(n_iter=EM_N_ITER, tol=EM_TOL)

        assert np.isclose(
            mhmm1.log_likelihood, mhmm2.log_likelihood, atol=1e-6
        ), (
            f"EM not deterministic: {mhmm1.log_likelihood} vs "
            f"{mhmm2.log_likelihood}"
        )

    def test_predict_cluster_after_fit(self, seqdata):
        """predict_cluster() returns valid assignments after fit."""
        mhmm = _build_test_mhmm(seqdata)
        mhmm.fit(n_iter=EM_N_ITER, tol=EM_TOL)
        clusters = mhmm.predict_cluster()
        assert len(clusters) == N_SEQUENCES
        assert all(0 <= c < N_CLUSTERS for c in clusters)

    def test_predict_cluster_before_fit_raises(self, seqdata):
        """predict_cluster() raises ValueError before fit."""
        mhmm = _build_test_mhmm(seqdata)
        with pytest.raises(ValueError, match="fitted"):
            mhmm.predict_cluster()


# ============================================================================
# Part 1: Cross-language consistency vs R seqHMM
# ============================================================================

def test_mhmm_loglik_matches_r(seqdata, ref_mhmm):
    """MHMM mixture logLik (before fitting) matches R's logLik(mhmm)."""
    mhmm = _build_test_mhmm(seqdata)
    py_ll = _compute_mhmm_loglik(mhmm)
    r_ll = float(ref_mhmm["loglik"]["loglik"])
    assert np.isclose(py_ll, r_ll, atol=LOGLIK_ATOL), (
        f"MHMM logLik mismatch: Python={py_ll:.6f}, R={r_ll:.6f}, "
        f"diff={abs(py_ll - r_ll):.6f}"
    )


def test_mhmm_em_loglik_before_matches_r(seqdata, ref_mhmm):
    """Pre-EM logLik should match R's logLik before EM (sanity check)."""
    mhmm = _build_test_mhmm(seqdata)
    py_ll = _compute_mhmm_loglik(mhmm)
    r_ll = float(ref_mhmm["em"]["loglik_before"])
    assert np.isclose(py_ll, r_ll, atol=LOGLIK_ATOL), (
        f"Pre-EM logLik mismatch: Python={py_ll:.6f}, R={r_ll:.6f}"
    )


def test_mhmm_em_loglik_matches_r(seqdata, ref_mhmm):
    """MHMM EM converged logLik is close to R's."""
    mhmm = _build_test_mhmm(seqdata)
    mhmm.fit(n_iter=EM_N_ITER, tol=EM_TOL)
    py_ll = mhmm.log_likelihood
    r_ll = float(ref_mhmm["em"]["loglik_after"])
    assert np.isclose(py_ll, r_ll, atol=LOGLIK_EM_ATOL), (
        f"MHMM EM logLik mismatch: Python={py_ll:.4f}, R={r_ll:.4f}, "
        f"diff={abs(py_ll - r_ll):.4f}"
    )


def test_mhmm_em_python_not_worse_than_r(seqdata, ref_mhmm):
    """Python MHMM EM should not be dramatically worse than R."""
    mhmm = _build_test_mhmm(seqdata)
    mhmm.fit(n_iter=EM_N_ITER, tol=EM_TOL)
    py_ll = mhmm.log_likelihood
    r_ll = float(ref_mhmm["em"]["loglik_after"])
    # Python can be up to 2.0 nats worse (different local optima)
    assert py_ll >= r_ll - 2.0, (
        f"Python MHMM EM much worse than R: Python={py_ll:.4f}, R={r_ll:.4f}"
    )
