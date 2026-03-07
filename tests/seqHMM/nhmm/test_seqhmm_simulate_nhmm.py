"""
@Author  : Yapeng Wei
@File    : test_seqhmm_simulate_nhmm.py
@Desc    :
Tests for simulate_nhmm() consistency between sequenzo.seqhmm
and R seqHMM.

Actual Python API:
  simulate_nhmm(
      n_states, emission_formula, data, id_var, time_var,
      initial_formula=None, transition_formula=None,
      coefs=None, init_sd=None, random_state=None
  ) -> dict
    Returns: {
      'observations': list of lists,
      'states': list of lists,
      'data': DataFrame with simulated response,
      'states_df': DataFrame with (id, time, state),
      'model': dict with n_states, n_symbols, alphabet, state_names,
               eta_pi, eta_A, eta_B, n_covariates_pi/A/B
    }

Since R and Python use different RNGs, we compare statistical properties
rather than exact sequences.

Test groups:
  Part 0: Sanity checks (no R needed)
"""
import numpy as np
import pandas as pd
import pytest

from sequenzo.seqhmm import simulate_nhmm


# ============================================================================
# Constants
# ============================================================================

STATES = ["A", "B", "C"]
N_STATES = 2
N_SEQ = 15
N_TIME = 10
SIM_SEED = 42


# ============================================================================
# Helpers
# ============================================================================

def _make_panel_data(n_seq=N_SEQ, n_time=N_TIME, seed=42):
    """Create panel data with id, time, response, and covariate x."""
    rng = np.random.RandomState(seed)
    ids = np.repeat(np.arange(1, n_seq + 1), n_time)
    times = np.tile(np.arange(1, n_time + 1), n_seq)
    x = np.round(rng.randn(n_seq * n_time), 3)

    # Response: just need valid categorical column to define alphabet
    response = rng.choice(STATES, size=n_seq * n_time)

    return pd.DataFrame({
        "id": ids,
        "time": times,
        "response": response,
        "x": x,
    })


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def panel_data():
    return _make_panel_data()


# ============================================================================
# Part 0: Sanity checks
# ============================================================================

class TestSimulateNHMMSanity:
    """Sanity checks for simulate_nhmm (no R needed)."""

    def test_returns_dict(self, panel_data):
        """simulate_nhmm returns a dict with expected keys."""
        result = simulate_nhmm(
            n_states=N_STATES,
            emission_formula="response ~ 1",
            data=panel_data,
            id_var="id",
            time_var="time",
            random_state=SIM_SEED,
        )
        assert isinstance(result, dict)
        for key in ("observations", "states", "data", "model"):
            assert key in result, f"Missing key: {key}"

    def test_n_sequences(self, panel_data):
        """Number of simulated sequences matches input."""
        result = simulate_nhmm(
            n_states=N_STATES,
            emission_formula="response ~ 1",
            data=panel_data,
            id_var="id",
            time_var="time",
            random_state=SIM_SEED,
        )
        assert len(result["observations"]) == N_SEQ
        assert len(result["states"]) == N_SEQ

    def test_sequence_lengths(self, panel_data):
        """Each simulated sequence has the correct length."""
        result = simulate_nhmm(
            n_states=N_STATES,
            emission_formula="response ~ 1",
            data=panel_data,
            id_var="id",
            time_var="time",
            random_state=SIM_SEED,
        )
        for i, obs_seq in enumerate(result["observations"]):
            assert len(obs_seq) == N_TIME, (
                f"Sequence {i} length {len(obs_seq)}, expected {N_TIME}"
            )
        for i, state_seq in enumerate(result["states"]):
            assert len(state_seq) == N_TIME

    def test_observations_in_alphabet(self, panel_data):
        """All simulated observations are in the defined alphabet."""
        result = simulate_nhmm(
            n_states=N_STATES,
            emission_formula="response ~ 1",
            data=panel_data,
            id_var="id",
            time_var="time",
            random_state=SIM_SEED,
        )
        alphabet = result["model"]["alphabet"]
        for seq in result["observations"]:
            for obs in seq:
                assert obs in alphabet, f"Obs '{obs}' not in alphabet {alphabet}"

    def test_states_are_valid(self, panel_data):
        """All hidden states use state_names from model."""
        result = simulate_nhmm(
            n_states=N_STATES,
            emission_formula="response ~ 1",
            data=panel_data,
            id_var="id",
            time_var="time",
            random_state=SIM_SEED,
        )
        state_names = result["model"]["state_names"]
        for seq in result["states"]:
            for s in seq:
                assert s in state_names, f"State '{s}' not in {state_names}"

    def test_model_info(self, panel_data):
        """Model dict contains n_states, n_symbols, eta coefficients."""
        result = simulate_nhmm(
            n_states=N_STATES,
            emission_formula="response ~ 1",
            data=panel_data,
            id_var="id",
            time_var="time",
            random_state=SIM_SEED,
        )
        model = result["model"]
        assert model["n_states"] == N_STATES
        assert model["n_symbols"] == len(STATES)
        assert "eta_pi" in model
        assert "eta_A" in model
        assert "eta_B" in model

    def test_data_same_shape(self, panel_data):
        """Returned data DataFrame has same shape as input."""
        result = simulate_nhmm(
            n_states=N_STATES,
            emission_formula="response ~ 1",
            data=panel_data,
            id_var="id",
            time_var="time",
            random_state=SIM_SEED,
        )
        assert result["data"].shape == panel_data.shape

    def test_data_response_replaced(self, panel_data):
        """Response column in returned data is replaced with simulated values."""
        result = simulate_nhmm(
            n_states=N_STATES,
            emission_formula="response ~ 1",
            data=panel_data,
            id_var="id",
            time_var="time",
            random_state=SIM_SEED,
        )
        sim_responses = result["data"]["response"].unique()
        # Should contain valid alphabet symbols
        alphabet = result["model"]["alphabet"]
        for val in sim_responses:
            assert val in alphabet

    def test_n_states_too_small_raises(self, panel_data):
        """n_states < 2 raises ValueError."""
        with pytest.raises(ValueError):
            simulate_nhmm(
                n_states=1,
                emission_formula="response ~ 1",
                data=panel_data,
                id_var="id",
                time_var="time",
                random_state=SIM_SEED,
            )

    def test_missing_emission_formula_raises(self, panel_data):
        """Missing emission_formula raises ValueError."""
        with pytest.raises(ValueError):
            simulate_nhmm(
                n_states=N_STATES,
                emission_formula=None,
                data=panel_data,
                id_var="id",
                time_var="time",
                random_state=SIM_SEED,
            )

    def test_bad_id_var_raises(self, panel_data):
        """Non-existent id_var raises ValueError."""
        with pytest.raises(ValueError):
            simulate_nhmm(
                n_states=N_STATES,
                emission_formula="response ~ 1",
                data=panel_data,
                id_var="nonexistent_id",
                time_var="time",
                random_state=SIM_SEED,
            )

    def test_with_covariate_formula(self, panel_data):
        """simulate_nhmm works with covariate in transition_formula."""
        result = simulate_nhmm(
            n_states=N_STATES,
            emission_formula="response ~ 1",
            data=panel_data,
            id_var="id",
            time_var="time",
            transition_formula="~ x",
            random_state=SIM_SEED,
        )
        assert len(result["observations"]) == N_SEQ
        # Transition covariate should add to n_covariates_A
        assert result["model"]["n_covariates_A"] >= 2  # intercept + x

    def test_deterministic_with_seed(self, panel_data):
        """Same random_state gives identical results."""
        r1 = simulate_nhmm(
            n_states=N_STATES,
            emission_formula="response ~ 1",
            data=panel_data,
            id_var="id",
            time_var="time",
            random_state=123,
        )
        r2 = simulate_nhmm(
            n_states=N_STATES,
            emission_formula="response ~ 1",
            data=panel_data,
            id_var="id",
            time_var="time",
            random_state=123,
        )
        assert r1["observations"] == r2["observations"]
        assert r1["states"] == r2["states"]

    def test_different_seeds_differ(self, panel_data):
        """Different random_state gives different results (with high probability)."""
        r1 = simulate_nhmm(
            n_states=N_STATES,
            emission_formula="response ~ 1",
            data=panel_data,
            id_var="id",
            time_var="time",
            random_state=1,
        )
        r2 = simulate_nhmm(
            n_states=N_STATES,
            emission_formula="response ~ 1",
            data=panel_data,
            id_var="id",
            time_var="time",
            random_state=999,
        )
        # At least some observations should differ
        any_diff = any(
            o1 != o2
            for s1, s2 in zip(r1["observations"], r2["observations"])
            for o1, o2 in zip(s1, s2)
        )
        assert any_diff, "Different seeds produced identical sequences"

    def test_all_symbols_appear(self, panel_data):
        """With enough sequences, all alphabet symbols should appear."""
        # Use larger dataset
        big_data = _make_panel_data(n_seq=50, n_time=20, seed=99)
        result = simulate_nhmm(
            n_states=N_STATES,
            emission_formula="response ~ 1",
            data=big_data,
            id_var="id",
            time_var="time",
            random_state=SIM_SEED,
        )
        all_obs = set()
        for seq in result["observations"]:
            all_obs.update(seq)
        alphabet = set(result["model"]["alphabet"])
        assert all_obs == alphabet, (
            f"Not all symbols appeared: got {all_obs}, expected {alphabet}"
        )

    def test_states_df_shape(self, panel_data):
        """states_df has correct number of rows."""
        result = simulate_nhmm(
            n_states=N_STATES,
            emission_formula="response ~ 1",
            data=panel_data,
            id_var="id",
            time_var="time",
            random_state=SIM_SEED,
        )
        if "states_df" in result:
            expected_rows = N_SEQ * N_TIME
            assert len(result["states_df"]) == expected_rows

    def test_with_provided_coefs(self, panel_data):
        """simulate_nhmm accepts user-supplied coefficients."""
        # First simulate to get coefficient shapes
        r1 = simulate_nhmm(
            n_states=N_STATES,
            emission_formula="response ~ 1",
            data=panel_data,
            id_var="id",
            time_var="time",
            random_state=SIM_SEED,
        )
        model = r1["model"]
        coefs = {
            "initial_probs": model["eta_pi"],
            "transition_probs": model["eta_A"],
            "emission_probs": model["eta_B"],
        }
        # Re-simulate with same coefficients
        r2 = simulate_nhmm(
            n_states=N_STATES,
            emission_formula="response ~ 1",
            data=panel_data,
            id_var="id",
            time_var="time",
            coefs=coefs,
            random_state=SIM_SEED,
        )
        # Verify coefs were used (model should store the same etas)
        np.testing.assert_array_equal(r2["model"]["eta_pi"], model["eta_pi"])
        np.testing.assert_array_equal(r2["model"]["eta_A"], model["eta_A"])
        np.testing.assert_array_equal(r2["model"]["eta_B"], model["eta_B"])
        # Basic validity
        assert len(r2["observations"]) == N_SEQ
