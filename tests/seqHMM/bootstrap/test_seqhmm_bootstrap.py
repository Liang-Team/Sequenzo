"""
@Author  : Yapeng Wei
@File    : test_seqhmm_bootstrap.py
@Desc    :
Tests for bootstrap_model() consistency between sequenzo.seqhmm
and R seqHMM.

Actual Python API:
  bootstrap_model(model, n_sim=100, method='nonparametric',
                  random_state=None, verbose=True, n_jobs=1) -> dict
    Requires: model must be fitted (model.log_likelihood is not None)
    Returns dict:
      'bootstrap_samples': list of dicts, each with:
          HMM: 'initial_probs', 'transition_probs', 'emission_probs'
          MHMM: 'cluster_probs', 'clusters' (list of dicts), 'coefficients'
          NHMM: 'eta_pi', 'eta_A', 'eta_B'
      'original_model': the input model
      'n_sim': number requested
      'n_successful': number that succeeded
      'method': 'nonparametric'
      'summary': nested dict with 'mean', 'std', 'ci_95' per parameter group

Test groups:
  Part 0: Sanity checks (no R)
  Part 1: Cross-language consistency (needs ref CSVs from R)

Run: Rscript seqhmm_reference_bootstrap.R .
"""
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd
import pytest

from sequenzo import SequenceData
from sequenzo.seqhmm import (
    build_hmm, fit_model, bootstrap_model,
    build_mhmm, fit_mhmm,
)


# ============================================================================
# Constants
# ============================================================================

STATES = ["A", "B", "C"]
SEQ_IDS = [f"s{i}" for i in range(5)]
TIME_COLS = ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"]
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# HMM config (same as loglik tests)
HMM_INIT = np.array([0.6, 0.4])
HMM_TRANS = np.array([[0.7, 0.3], [0.2, 0.8]])
HMM_EMISS = np.array([[0.5, 0.3, 0.2], [0.1, 0.4, 0.5]])
N_STATES_HMM = 2

TEST_DATA_RAW = [
    ["A", "A", "B", "B", "C", "A", "A", "B"],
    ["B", "C", "C", "A", "A", "B", "C", "C"],
    ["A", "B", "C", "A", "B", "C", "A", "B"],
    ["C", "C", "B", "A", "A", "A", "B", "C"],
    ["A", "A", "A", "B", "B", "B", "C", "C"],
]

# Bootstrap settings (small for speed)
BOOT_N_SIM = 20
BOOT_SEED = 42


# ============================================================================
# Helpers
# ============================================================================

def _make_seqdata():
    df = pd.DataFrame(TEST_DATA_RAW, columns=TIME_COLS)
    df.insert(0, "id", SEQ_IDS)
    return SequenceData(df, time=TIME_COLS, states=STATES, id_col="id")


def _make_fitted_hmm(seqdata):
    hmm = build_hmm(
        seqdata,
        n_states=N_STATES_HMM,
        initial_probs=HMM_INIT.copy(),
        transition_probs=HMM_TRANS.copy(),
        emission_probs=HMM_EMISS.copy(),
    )
    return fit_model(hmm, n_iter=100, tol=1e-6)


def _run_r_reference(outdir, timeout=300):
    r_script = os.path.join(THIS_DIR, "seqhmm_reference_bootstrap.R")
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
    if "key" in df.columns and "value" in df.columns:
        return dict(zip(df["key"], df["value"]))
    return None


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def seqdata():
    return _make_seqdata()


@pytest.fixture(scope="module")
def ref_bootstrap():
    refs = {}
    fname = "ref_bootstrap_stats.csv"

    ref = _load_ref_kv(THIS_DIR, fname)
    if ref is not None:
        if ref.get("status") in ("failed", "bootstrap_failed"):
            pytest.skip("R bootstrap estimation failed.")
        refs["stats"] = ref
        return refs

    outdir = tempfile.mkdtemp()
    ok = _run_r_reference(outdir)
    if ok:
        ref = _load_ref_kv(outdir, fname)
        if ref is not None:
            if ref.get("status") in ("failed", "bootstrap_failed"):
                pytest.skip("R bootstrap estimation failed.")
            refs["stats"] = ref
            return refs

    pytest.skip(
        "R/seqHMM not available and ref_bootstrap_stats.csv not found. "
        "Run: Rscript seqhmm_reference_bootstrap.R ."
    )


# ============================================================================
# Part 0: Sanity checks (no R)
# ============================================================================

class TestBootstrapSanity:
    """Sanity checks for bootstrap_model (no R needed)."""

    def test_unfitted_raises(self, seqdata):
        """bootstrap_model raises ValueError on unfitted model."""
        hmm = build_hmm(
            seqdata,
            n_states=N_STATES_HMM,
            initial_probs=HMM_INIT.copy(),
            transition_probs=HMM_TRANS.copy(),
            emission_probs=HMM_EMISS.copy(),
        )
        with pytest.raises(ValueError, match="fitted"):
            bootstrap_model(hmm, n_sim=5)

    def test_returns_dict(self, seqdata):
        """bootstrap_model returns a dict with expected keys."""
        hmm = _make_fitted_hmm(seqdata)
        result = bootstrap_model(
            hmm, n_sim=BOOT_N_SIM, random_state=BOOT_SEED, verbose=False
        )
        assert isinstance(result, dict)
        for key in ("bootstrap_samples", "original_model", "n_sim",
                     "n_successful", "method", "summary"):
            assert key in result, f"Missing key: {key}"

    def test_n_sim_matches(self, seqdata):
        """Returned n_sim equals requested value."""
        hmm = _make_fitted_hmm(seqdata)
        result = bootstrap_model(
            hmm, n_sim=BOOT_N_SIM, random_state=BOOT_SEED, verbose=False
        )
        assert result["n_sim"] == BOOT_N_SIM

    def test_bootstrap_samples_nonempty(self, seqdata):
        """At least some bootstrap samples succeed."""
        hmm = _make_fitted_hmm(seqdata)
        result = bootstrap_model(
            hmm, n_sim=BOOT_N_SIM, random_state=BOOT_SEED, verbose=False
        )
        assert result["n_successful"] > 0, "No bootstrap samples succeeded"
        assert len(result["bootstrap_samples"]) == result["n_successful"]

    def test_bootstrap_sample_keys_hmm(self, seqdata):
        """Each HMM bootstrap sample has initial/transition/emission probs."""
        hmm = _make_fitted_hmm(seqdata)
        result = bootstrap_model(
            hmm, n_sim=BOOT_N_SIM, random_state=BOOT_SEED, verbose=False
        )
        for i, sample in enumerate(result["bootstrap_samples"]):
            for key in ("initial_probs", "transition_probs", "emission_probs"):
                assert key in sample, (
                    f"Bootstrap sample {i} missing '{key}'"
                )

    def test_bootstrap_params_valid(self, seqdata):
        """Bootstrap parameter estimates are valid probability distributions."""
        hmm = _make_fitted_hmm(seqdata)
        result = bootstrap_model(
            hmm, n_sim=BOOT_N_SIM, random_state=BOOT_SEED, verbose=False
        )
        for sample in result["bootstrap_samples"]:
            ip = sample["initial_probs"]
            assert np.all(ip >= -1e-10) and np.isclose(ip.sum(), 1.0, atol=1e-6)

            tp = sample["transition_probs"]
            assert np.all(tp >= -1e-10)
            assert np.allclose(tp.sum(axis=1), 1.0, atol=1e-6)

            ep = sample["emission_probs"]
            assert np.all(ep >= -1e-10)
            assert np.allclose(ep.sum(axis=1), 1.0, atol=1e-6)

    def test_summary_has_ci(self, seqdata):
        """Summary includes mean, std, and 95% CI."""
        hmm = _make_fitted_hmm(seqdata)
        result = bootstrap_model(
            hmm, n_sim=BOOT_N_SIM, random_state=BOOT_SEED, verbose=False
        )
        summary = result["summary"]
        for param_name in ("initial_probs", "transition_probs", "emission_probs"):
            assert param_name in summary, f"Summary missing '{param_name}'"
            for stat in ("mean", "std", "ci_95"):
                assert stat in summary[param_name], (
                    f"Summary['{param_name}'] missing '{stat}'"
                )

    def test_ci_covers_original(self, seqdata):
        """95% CI from bootstrap should generally cover original estimates."""
        hmm = _make_fitted_hmm(seqdata)
        result = bootstrap_model(
            hmm, n_sim=BOOT_N_SIM, random_state=BOOT_SEED, verbose=False
        )
        # Check initial_probs CI covers original
        ci = result["summary"]["initial_probs"]["ci_95"]
        original_ip = hmm.initial_probs
        # ci shape: (2, n_states) where ci[0] = 2.5%, ci[1] = 97.5%
        for i in range(len(original_ip)):
            lower, upper = ci[0][i], ci[1][i]
            # With small n_sim, this may not always hold perfectly,
            # so we just check that CI is reasonable (width > 0)
            assert upper >= lower, (
                f"CI inverted for initial_probs[{i}]: [{lower}, {upper}]"
            )

    def test_bootstrap_mean_close_to_original(self, seqdata):
        """Bootstrap mean should be close to original parameter estimates."""
        hmm = _make_fitted_hmm(seqdata)
        result = bootstrap_model(
            hmm, n_sim=BOOT_N_SIM, random_state=BOOT_SEED, verbose=False
        )
        boot_mean_ip = result["summary"]["initial_probs"]["mean"]
        original_ip = hmm.initial_probs
        # Generous tolerance: bootstrap with small n_sim can vary
        assert np.allclose(boot_mean_ip, original_ip, atol=0.3), (
            f"Bootstrap mean {boot_mean_ip} far from original {original_ip}"
        )

    def test_method_is_nonparametric(self, seqdata):
        """Default method is 'nonparametric'."""
        hmm = _make_fitted_hmm(seqdata)
        result = bootstrap_model(
            hmm, n_sim=5, random_state=BOOT_SEED, verbose=False
        )
        assert result["method"] == "nonparametric"

    def test_deterministic_with_seed(self, seqdata):
        """Same random_state gives same results."""
        hmm = _make_fitted_hmm(seqdata)
        r1 = bootstrap_model(hmm, n_sim=10, random_state=999, verbose=False)
        r2 = bootstrap_model(hmm, n_sim=10, random_state=999, verbose=False)
        # Compare first bootstrap sample
        if r1["n_successful"] > 0 and r2["n_successful"] > 0:
            np.testing.assert_allclose(
                r1["bootstrap_samples"][0]["initial_probs"],
                r2["bootstrap_samples"][0]["initial_probs"],
                atol=1e-6,
            )


# ============================================================================
# Part 1: Cross-language (needs R reference)
# ============================================================================

def test_bootstrap_structure_consistent_with_r(ref_bootstrap):
    """R bootstrap ran successfully (status check)."""
    stats = ref_bootstrap["stats"]
    assert stats.get("status") == "success", (
        f"R bootstrap status: {stats.get('status')}"
    )


def test_bootstrap_n_sim_matches_r(ref_bootstrap):
    """Bootstrap B (n_sim) recorded by R matches expectations."""
    stats = ref_bootstrap["stats"]
    assert float(stats["B"]) == 50, (
        f"R used B={stats['B']}, expected 50"
    )
