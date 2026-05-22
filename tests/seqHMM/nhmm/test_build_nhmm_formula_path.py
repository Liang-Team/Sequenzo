import numpy as np
import pandas as pd
import pytest

from sequenzo import SequenceData
import sequenzo.seqhmm.nhmm as nhmm_module
from sequenzo.seqhmm import build_nhmm
from sequenzo.seqhmm.forward_backward_nhmm import log_likelihood_nhmm
from sequenzo.seqhmm.formulas import Formula


def test_build_nhmm_formula_path_creates_covariate_matrix():
    panel = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "time": [1, 2, 3, 1, 2, 3],
            "response": ["A", "B", "A", "B", "A", "B"],
            "x": [0.1, 0.2, 0.3, -0.1, -0.2, -0.3],
        }
    )
    wide = panel.pivot(index="id", columns="time", values="response").reset_index()
    seq = SequenceData(wide, time=[1, 2, 3], states=["A", "B"], id_col="id")

    nhmm = build_nhmm(
        seq,
        n_states=2,
        emission_formula="~ x",
        data=panel,
        id_var="id",
        time_var="time",
        random_state=42,
    )

    assert nhmm.X.shape == (2, 3, 2)
    assert np.allclose(nhmm.X[:, :, 0], 1.0)
    assert np.allclose(
        nhmm.X[:, :, 1],
        np.array(
            [
                [0.1, 0.2, 0.3],
                [-0.1, -0.2, -0.3],
            ]
        ),
    )


def test_build_nhmm_formula_path_aligns_to_sequence_id_and_time_order():
    panel = pd.DataFrame(
        {
            "id": [1, 2, 1, 2, 1, 2],
            "time": [2, 3, 1, 1, 3, 2],
            "response": ["B", "B", "A", "B", "A", "B"],
            "x": [12.0, 23.0, 11.0, 21.0, 13.0, 22.0],
        }
    )
    wide = pd.DataFrame(
        {
            "id": [2, 1],
            3: ["B", "A"],
            1: ["B", "A"],
            2: ["B", "B"],
        }
    )
    seq = SequenceData(wide, time=[3, 1, 2], states=["A", "B"], id_col="id")

    nhmm = build_nhmm(
        seq,
        n_states=2,
        emission_formula="~ x",
        data=panel,
        id_var="id",
        time_var="time",
        random_state=42,
    )

    assert np.allclose(
        nhmm.X[:, :, 1],
        np.array(
            [
                [23.0, 21.0, 22.0],
                [13.0, 11.0, 12.0],
            ]
        ),
    )


def test_build_nhmm_keeps_separate_formula_design_matrices():
    panel = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "time": [1, 2, 3, 1, 2, 3],
            "response": ["A", "B", "A", "B", "A", "B"],
            "x_pi": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            "x_A": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "x_B": [10.0, 11.0, 12.0, 20.0, 21.0, 22.0],
        }
    )
    wide = panel.pivot(index="id", columns="time", values="response").reset_index()
    seq = SequenceData(wide, time=[1, 2, 3], states=["A", "B"], id_col="id")

    nhmm = build_nhmm(
        seq,
        n_states=2,
        initial_formula="~ x_pi",
        transition_formula="~ x_A",
        emission_formula="~ x_B",
        data=panel,
        id_var="id",
        time_var="time",
        random_state=42,
    )

    assert nhmm.X_pi.shape == (2, 3, 2)
    assert nhmm.X_A.shape == (2, 3, 2)
    assert nhmm.X_B.shape == (2, 3, 2)
    assert np.allclose(nhmm.X_pi[:, :, 1], [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    assert np.allclose(nhmm.X_A[:, :, 1], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    assert np.allclose(nhmm.X_B[:, :, 1], [[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])
    assert nhmm.eta_pi.shape[0] == 2
    assert nhmm.eta_A.shape[0] == 2
    assert nhmm.eta_B.shape[0] == 2


def test_build_nhmm_accepts_separate_direct_design_matrices_without_legacy_x():
    wide = pd.DataFrame(
        {
            "id": [1, 2],
            "t1": ["A", "B"],
            "t2": ["B", "A"],
            "t3": ["A", "B"],
        }
    )
    seq = SequenceData(wide, time=["t1", "t2", "t3"], states=["A", "B"], id_col="id")
    X_pi = np.ones((2, 3, 1))
    X_A = np.dstack([np.ones((2, 3)), np.arange(6).reshape(2, 3)])
    X_B = np.dstack([np.ones((2, 3)), np.full((2, 3), 7.0), np.full((2, 3), -2.0)])

    nhmm = build_nhmm(
        seq,
        n_states=2,
        X_pi=X_pi,
        X_A=X_A,
        X_B=X_B,
        random_state=42,
    )

    assert nhmm.X is nhmm.X_B
    assert nhmm.X_pi.shape == (2, 3, 1)
    assert nhmm.X_A.shape == (2, 3, 2)
    assert nhmm.X_B.shape == (2, 3, 3)
    assert nhmm.eta_pi.shape == (1, 2)
    assert nhmm.eta_A.shape == (2, 2, 2)
    assert nhmm.eta_B.shape == (3, 2, 2)


def test_build_nhmm_rejects_partial_direct_family_matrices_without_legacy_x():
    wide = pd.DataFrame({"id": [1], "t1": ["A"], "t2": ["B"]})
    seq = SequenceData(wide, time=["t1", "t2"], states=["A", "B"], id_col="id")

    with pytest.raises(ValueError, match="X_pi, X_A, and X_B"):
        build_nhmm(seq, n_states=2, X_pi=np.ones((1, 2, 1)))


def test_build_nhmm_formula_missing_families_fall_back_to_legacy_x():
    panel = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "time": [1, 2, 1, 2],
            "response": ["A", "B", "B", "A"],
            "x": [0.0, 1.0, 2.0, 3.0],
        }
    )
    wide = panel.pivot(index="id", columns="time", values="response").reset_index()
    seq = SequenceData(wide, time=[1, 2], states=["A", "B"], id_col="id")
    X = np.ones((2, 2, 3))
    X[:, :, 1] = 7.0
    X[:, :, 2] = -2.0

    nhmm = build_nhmm(
        seq,
        n_states=2,
        X=X,
        emission_formula="~ x",
        data=panel,
        id_var="id",
        time_var="time",
    )

    assert nhmm.X_pi is nhmm.X
    assert nhmm.X_A is nhmm.X
    assert nhmm.X_B.shape == (2, 2, 2)


def test_build_nhmm_formula_path_rejects_missing_covariate_values():
    panel = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "time": [1, 2, 3, 1, 2, 3],
            "response": ["A", "B", "A", "B", "A", "B"],
            "x": [10.0, np.nan, 30.0, 40.0, 50.0, 60.0],
        }
    )
    wide = panel.pivot(index="id", columns="time", values="response").reset_index()
    seq = SequenceData(wide, time=[1, 2, 3], states=["A", "B"], id_col="id")

    with pytest.raises(ValueError, match="non-missing|finite|complete"):
        build_nhmm(
            seq,
            n_states=2,
            emission_formula="~ x",
            data=panel,
            id_var="id",
            time_var="time",
        )


def test_build_nhmm_formula_path_rejects_incomplete_id_time_grid():
    panel = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2],
            "time": [1, 2, 3, 1, 2],
            "response": ["A", "B", "A", "B", "A"],
            "x": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    wide = pd.DataFrame(
        {
            "id": [1, 2],
            1: ["A", "B"],
            2: ["B", "A"],
            3: ["A", "B"],
        }
    )
    seq = SequenceData(wide, time=[1, 2, 3], states=["A", "B"], id_col="id")

    with pytest.raises(ValueError, match="complete id x time grid"):
        build_nhmm(
            seq,
            n_states=2,
            emission_formula="~ x",
            data=panel,
            id_var="id",
            time_var="time",
        )


def test_build_nhmm_formula_path_rejects_duplicate_id_time_cells():
    panel = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2],
            "time": [1, 1, 2, 1, 2],
            "response": ["A", "A", "B", "B", "A"],
            "x": [10.0, 11.0, 20.0, 30.0, 40.0],
        }
    )
    wide = pd.DataFrame({"id": [1, 2], 1: ["A", "B"], 2: ["B", "A"]})
    seq = SequenceData(wide, time=[1, 2], states=["A", "B"], id_col="id")

    with pytest.raises(ValueError, match="duplicate id/time"):
        build_nhmm(
            seq,
            n_states=2,
            emission_formula="~ x",
            data=panel,
            id_var="id",
            time_var="time",
        )


def test_build_nhmm_formula_path_rejects_extra_id_time_cells():
    panel = pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3],
            "time": [1, 2, 1, 2, 1, 2],
            "response": ["A", "B", "B", "A", "A", "A"],
            "x": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )
    wide = pd.DataFrame({"id": [1, 2], 1: ["A", "B"], 2: ["B", "A"]})
    seq = SequenceData(wide, time=[1, 2], states=["A", "B"], id_col="id")

    with pytest.raises(ValueError, match="outside the SequenceData grid"):
        build_nhmm(
            seq,
            n_states=2,
            emission_formula="~ x",
            data=panel,
            id_var="id",
            time_var="time",
        )


def test_build_nhmm_formula_path_ignores_no_extra_categorical_levels():
    panel = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "time": [1, 2, 1, 2],
            "response": ["A", "B", "B", "A"],
            "group": ["a", "a", "a", "a"],
        }
    )
    wide = panel.pivot(index="id", columns="time", values="response").reset_index()
    seq = SequenceData(wide, time=[1, 2], states=["A", "B"], id_col="id")

    nhmm = build_nhmm(
        seq,
        n_states=2,
        emission_formula="~ C(group)",
        data=panel,
        id_var="id",
        time_var="time",
    )

    assert nhmm.X_B.shape == (2, 2, 1)
    assert np.allclose(nhmm.X_B, 1.0)


def test_build_nhmm_lag_follows_sequence_time_order():
    panel = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "time": ["t1", "t2", "t1", "t2"],
            "response": ["A", "B", "B", "A"],
            "x": [10.0, 20.0, 30.0, 40.0],
        }
    )
    wide = pd.DataFrame({"id": [1, 2], "t2": ["B", "A"], "t1": ["A", "B"]})
    seq = SequenceData(wide, time=["t2", "t1"], states=["A", "B"], id_col="id")

    nhmm = build_nhmm(
        seq,
        n_states=2,
        emission_formula="~ lag(x)",
        data=panel,
        id_var="id",
        time_var="time",
    )

    assert np.allclose(nhmm.X_B[:, :, 1], [[0.0, 20.0], [0.0, 40.0]])


def test_build_nhmm_lag_rejects_real_missing_covariates():
    panel = pd.DataFrame(
        {
            "id": [1, 1, 1],
            "time": [1, 2, 3],
            "response": ["A", "B", "A"],
            "x": [2.0, np.nan, 4.0],
        }
    )
    wide = panel.pivot(index="id", columns="time", values="response").reset_index()
    seq = SequenceData(wide, time=[1, 2, 3], states=["A", "B"], id_col="id")

    with pytest.raises(ValueError, match="non-missing|finite"):
        build_nhmm(
            seq,
            n_states=2,
            transition_formula="~ lag(x)",
            data=panel,
            id_var="id",
            time_var="time",
        )


def test_build_nhmm_lag_rejects_nonfinite_source_covariates():
    panel = pd.DataFrame(
        {
            "id": [1, 1, 1],
            "time": [1, 2, 3],
            "response": ["A", "B", "A"],
            "x": [2.0, 3.0, np.inf],
        }
    )
    wide = panel.pivot(index="id", columns="time", values="response").reset_index()
    seq = SequenceData(wide, time=[1, 2, 3], states=["A", "B"], id_col="id")

    with pytest.raises(ValueError, match="non-missing|finite"):
        build_nhmm(
            seq,
            n_states=2,
            transition_formula="~ lag(x)",
            data=panel,
            id_var="id",
            time_var="time",
        )


def test_nhmm_transition_covariates_use_destination_time():
    wide = pd.DataFrame({"id": [1], "t1": ["A"], "t2": ["B"]})
    seq = SequenceData(wide, time=["t1", "t2"], states=["A", "B"], id_col="id")
    X_pi = np.ones((1, 2, 1))
    X_A = np.array([[[1.0, -4.0], [1.0, 4.0]]])
    X_B = np.ones((1, 2, 1))
    eta_pi = np.zeros((1, 2))
    eta_A = np.zeros((2, 2, 2))
    eta_A[1, :, 0] = -1.0
    eta_A[1, :, 1] = 1.0
    eta_B = np.array([[[4.0, -4.0], [-4.0, 4.0]]])

    model = build_nhmm(
        seq,
        n_states=2,
        X_pi=X_pi,
        X_A=X_A,
        X_B=X_B,
        eta_pi=eta_pi,
        eta_A=eta_A,
        eta_B=eta_B,
    )

    initial_probs, transition_probs, emission_probs = model._compute_probs()
    observed = np.array([0, 1])
    destination_time = 1
    expected = 0.0
    for i in range(2):
        for j in range(2):
            expected += (
                initial_probs[0, i]
                * emission_probs[0, 0, i, observed[0]]
                * transition_probs[0, destination_time, i, j]
                * emission_probs[0, 1, j, observed[1]]
            )

    assert np.isclose(log_likelihood_nhmm(model), np.log(expected))


def test_nhmm_internal_missing_preserves_time_axis():
    wide = pd.DataFrame({"id": [1], "t1": ["A"], "t2": [np.nan], "t3": ["B"]})
    seq = SequenceData(wide, time=["t1", "t2", "t3"], states=["A", "B"], id_col="id")
    X_pi = np.ones((1, 3, 1))
    X_A = np.array([[[1.0, -2.0], [1.0, 0.0], [1.0, 2.0]]])
    X_B = np.ones((1, 3, 1))
    eta_pi = np.zeros((1, 2))
    eta_A = np.zeros((2, 2, 2))
    eta_A[1, :, 0] = -1.0
    eta_A[1, :, 1] = 1.0
    eta_B = np.array([[[4.0, -4.0], [-4.0, 4.0]]])

    model = build_nhmm(
        seq,
        n_states=2,
        X_pi=X_pi,
        X_A=X_A,
        X_B=X_B,
        eta_pi=eta_pi,
        eta_A=eta_A,
        eta_B=eta_B,
    )

    initial_probs, transition_probs, emission_probs = model._compute_probs()
    expected = 0.0
    for h0 in range(2):
        for h1 in range(2):
            for h2 in range(2):
                expected += (
                    initial_probs[0, h0]
                    * emission_probs[0, 0, h0, 0]
                    * transition_probs[0, 1, h0, h1]
                    * transition_probs[0, 2, h1, h2]
                    * emission_probs[0, 2, h2, 1]
                )

    compacted_wrong = 0.0
    for h0 in range(2):
        for h1 in range(2):
            compacted_wrong += (
                initial_probs[0, h0]
                * emission_probs[0, 0, h0, 0]
                * transition_probs[0, 1, h0, h1]
                * emission_probs[0, 1, h1, 1]
            )

    actual = log_likelihood_nhmm(model)
    assert np.isclose(actual, np.log(expected))
    assert not np.isclose(actual, np.log(compacted_wrong))


def test_nhmm_all_missing_sequence_preserves_later_sequence_index():
    wide = pd.DataFrame(
        {
            "id": [1, 2],
            "t1": [np.nan, "A"],
            "t2": [np.nan, "B"],
        }
    )
    seq = SequenceData(wide, time=["t1", "t2"], states=["A", "B"], id_col="id")
    X_pi = np.ones((2, 2, 1))
    X_A = np.array(
        [
            [[1.0, -8.0], [1.0, -8.0]],
            [[1.0, 8.0], [1.0, 8.0]],
        ]
    )
    X_B = np.ones((2, 2, 1))
    eta_pi = np.zeros((1, 2))
    eta_A = np.zeros((2, 2, 2))
    eta_A[1, :, 0] = -1.0
    eta_A[1, :, 1] = 1.0
    eta_B = np.array([[[4.0, -4.0], [-4.0, 4.0]]])

    model = build_nhmm(
        seq,
        n_states=2,
        X_pi=X_pi,
        X_A=X_A,
        X_B=X_B,
        eta_pi=eta_pi,
        eta_A=eta_A,
        eta_B=eta_B,
    )
    initial_probs, transition_probs, emission_probs = model._compute_probs()

    eps = 1e-10
    all_missing_prob = 0.0
    for h0 in range(2):
        for h1 in range(2):
            all_missing_prob += (
                (initial_probs[0, h0] + eps)
                * (transition_probs[0, 1, h0, h1] + eps)
            )

    observed_prob = 0.0
    for h0 in range(2):
        for h1 in range(2):
            observed_prob += (
                (initial_probs[1, h0] + eps)
                * (emission_probs[1, 0, h0, 0] + eps)
                * (transition_probs[1, 1, h0, h1] + eps)
                * (emission_probs[1, 1, h1, 1] + eps)
            )

    wrong_observed_prob = 0.0
    for h0 in range(2):
        for h1 in range(2):
            wrong_observed_prob += (
                (initial_probs[0, h0] + eps)
                * (emission_probs[0, 0, h0, 0] + eps)
                * (transition_probs[0, 1, h0, h1] + eps)
                * (emission_probs[0, 1, h1, 1] + eps)
            )

    actual = log_likelihood_nhmm(model)
    expected = np.log(all_missing_prob) + np.log(observed_prob)
    wrong = np.log(all_missing_prob) + np.log(wrong_observed_prob)

    assert np.isclose(actual, expected)
    assert not np.isclose(actual, wrong)


def test_build_nhmm_rejects_nonfinite_direct_covariates():
    wide = pd.DataFrame({"id": [1], "t1": ["A"], "t2": ["B"]})
    seq = SequenceData(wide, time=["t1", "t2"], states=["A", "B"], id_col="id")
    X = np.ones((1, 2, 1))
    X[0, 1, 0] = np.inf

    with pytest.raises(ValueError, match="finite"):
        build_nhmm(seq, n_states=2, X=X)


def test_build_nhmm_rejects_nonfinite_coefficients():
    wide = pd.DataFrame({"id": [1], "t1": ["A"], "t2": ["B"]})
    seq = SequenceData(wide, time=["t1", "t2"], states=["A", "B"], id_col="id")
    X = np.ones((1, 2, 1))
    eta_B = np.zeros((1, 2, 2))
    eta_B[0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        build_nhmm(seq, n_states=2, X=X, eta_B=eta_B)


def test_build_nhmm_rejects_lhs_in_initial_and_transition_formulas():
    panel = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "time": [1, 2, 1, 2],
            "response": ["A", "B", "B", "A"],
            "x": [0.0, 1.0, 2.0, 3.0],
        }
    )
    wide = panel.pivot(index="id", columns="time", values="response").reset_index()
    seq = SequenceData(wide, time=[1, 2], states=["A", "B"], id_col="id")

    with pytest.raises(ValueError, match="initial_formula.*left-hand side"):
        build_nhmm(
            seq,
            n_states=2,
            initial_formula="response ~ x",
            data=panel,
            id_var="id",
            time_var="time",
        )
    with pytest.raises(ValueError, match="transition_formula.*left-hand side"):
        build_nhmm(
            seq,
            n_states=2,
            transition_formula=Formula("response ~ x"),
            data=panel,
            id_var="id",
            time_var="time",
        )


def test_build_nhmm_rejects_lag_in_initial_formula():
    panel = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "time": [1, 2, 1, 2],
            "response": ["A", "B", "B", "A"],
            "x": [0.0, 1.0, 2.0, 3.0],
        }
    )
    wide = panel.pivot(index="id", columns="time", values="response").reset_index()
    seq = SequenceData(wide, time=[1, 2], states=["A", "B"], id_col="id")

    with pytest.raises(ValueError, match="initial_formula.*lag"):
        build_nhmm(
            seq,
            n_states=2,
            initial_formula="~ lag(x)",
            data=panel,
            id_var="id",
            time_var="time",
        )


def test_build_nhmm_rejects_unknown_emission_formula_lhs():
    panel = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "time": [1, 2, 1, 2],
            "response": ["A", "B", "B", "A"],
            "x": [0.0, 1.0, 2.0, 3.0],
        }
    )
    wide = panel.pivot(index="id", columns="time", values="response").reset_index()
    seq = SequenceData(wide, time=[1, 2], states=["A", "B"], id_col="id")

    with pytest.raises(ValueError, match="emission_formula left-hand side"):
        build_nhmm(
            seq,
            n_states=2,
            emission_formula="missing_response ~ x",
            data=panel,
            id_var="id",
            time_var="time",
        )


def test_build_nhmm_rejects_emission_formula_lhs_that_differs_from_observations():
    panel = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "time": [1, 2, 1, 2],
            "response": ["A", "B", "B", "A"],
            "wrong_response": ["B", "A", "A", "B"],
            "x": [0.0, 1.0, 2.0, 3.0],
        }
    )
    wide = panel.pivot(index="id", columns="time", values="response").reset_index()
    seq = SequenceData(wide, time=[1, 2], states=["A", "B"], id_col="id")

    with pytest.raises(ValueError, match="emission_formula left-hand side"):
        build_nhmm(
            seq,
            n_states=2,
            emission_formula="wrong_response ~ x",
            data=panel,
            id_var="id",
            time_var="time",
        )


def test_build_nhmm_emission_formula_lhs_allows_matching_missing_observations():
    panel = pd.DataFrame(
        {
            "id": [1, 1, 1],
            "time": [1, 2, 3],
            "response": ["A", np.nan, "B"],
            "x": [0.0, 1.0, 2.0],
        }
    )
    wide = panel.pivot(index="id", columns="time", values="response").reset_index()
    seq = SequenceData(wide, time=[1, 2, 3], states=["A", "B"], id_col="id")

    nhmm = build_nhmm(
        seq,
        n_states=2,
        emission_formula="response ~ x",
        data=panel,
        id_var="id",
        time_var="time",
    )

    assert nhmm.X_B.shape == (1, 3, 2)


def test_nhmm_formula_path_matches_seqhmm_fixed_eta_loglik():
    panel = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "time": [1, 2, 3, 1, 2, 3],
            "response": ["A", "B", "A", "B", "A", "B"],
            "x": [-1.0, 0.0, 1.0, 0.5, 1.5, 2.5],
            "group": [0, 0, 0, 1, 1, 1],
        }
    )
    wide = panel.pivot(index="id", columns="time", values="response").reset_index()
    seq = SequenceData(wide, time=[1, 2, 3], states=["A", "B"], id_col="id")

    eta_pi = np.array(
        [
            [0.1414213562373095, -0.1414213562373095],
            [-0.07071067811865475, 0.07071067811865475],
        ]
    )
    eta_A = np.zeros((2, 2, 2))
    eta_A[:, 0, :] = np.array(
        [
            [0.07071067811865475, -0.07071067811865475],
            [0.21213203435596423, -0.21213203435596423],
        ]
    )
    eta_A[:, 1, :] = np.array(
        [
            [-0.1414213562373095, 0.1414213562373095],
            [0.035355339059327376, -0.035355339059327376],
        ]
    )
    eta_B = np.zeros((1, 2, 2))
    eta_B[:, 0, :] = np.array(
        [[0.1414213562373095, -0.1414213562373095]]
    )
    eta_B[:, 1, :] = np.array(
        [[-0.21213203435596423, 0.21213203435596423]]
    )

    model = build_nhmm(
        seq,
        n_states=2,
        initial_formula="~ group",
        transition_formula="~ x",
        emission_formula="response ~ 1",
        data=panel,
        id_var="id",
        time_var="time",
        eta_pi=eta_pi,
        eta_A=eta_A,
        eta_B=eta_B,
    )
    initial_probs, transition_probs, emission_probs = model._compute_probs()

    assert np.allclose(model.X_pi[:, :, 1], [[0, 0, 0], [1, 1, 1]])
    assert np.allclose(model.X_A[:, :, 1], [[-1, 0, 1], [0.5, 1.5, 2.5]])
    assert np.allclose(model.X_B[:, :, 0], 1.0)
    assert np.allclose(initial_probs[0], [0.57024303, 0.42975697])
    assert np.allclose(transition_probs[0, 1, 0], [0.53529653, 0.46470347])
    assert np.allclose(emission_probs[0, 0, 0], [0.57024303, 0.42975697])
    assert np.isclose(log_likelihood_nhmm(model), -4.175555320096565)


def test_nhmm_fit_stores_optimizer_result_params(monkeypatch):
    wide = pd.DataFrame({"id": [1], "t1": ["A"], "t2": ["B"]})
    seq = SequenceData(wide, time=["t1", "t2"], states=["A", "B"], id_col="id")
    model = build_nhmm(seq, n_states=2, X=np.ones((1, 2, 1)), random_state=1)

    start = np.concatenate(
        [model.eta_pi.ravel(), model.eta_A.ravel(), model.eta_B.ravel()]
    )
    best = start + 0.25
    rejected = start - 0.5

    class Result:
        x = best
        fun = 0.0
        nit = 1
        success = True

    def fake_minimize(fun, params, method=None, jac=None, options=None):
        if jac:
            fun(rejected)
        else:
            fun(rejected)
        return Result()

    monkeypatch.setattr(nhmm_module, "minimize", fake_minimize)

    model.fit(n_iter=1)
    actual = np.concatenate(
        [model.eta_pi.ravel(), model.eta_A.ravel(), model.eta_B.ravel()]
    )

    assert np.allclose(actual, best)
    assert np.isclose(model.log_likelihood, log_likelihood_nhmm(model))


def test_nhmm_rejects_wrong_length_optimizer_params():
    wide = pd.DataFrame({"id": [1], "t1": ["A"], "t2": ["B"]})
    seq = SequenceData(wide, time=["t1", "t2"], states=["A", "B"], id_col="id")
    model = build_nhmm(seq, n_states=2, X=np.ones((1, 2, 1)), random_state=1)
    params = np.concatenate(
        [model.eta_pi.ravel(), model.eta_A.ravel(), model.eta_B.ravel()]
    )

    with pytest.raises(ValueError, match="params length"):
        model._unpack_params(np.r_[params, 1.0])


def test_nhmm_fit_nonfinite_trial_gradient_does_not_mutate_model(monkeypatch):
    wide = pd.DataFrame({"id": [1], "t1": ["A"], "t2": ["B"]})
    seq = SequenceData(wide, time=["t1", "t2"], states=["A", "B"], id_col="id")
    model = build_nhmm(seq, n_states=2, X=np.ones((1, 2, 1)), random_state=1)

    start = np.concatenate(
        [model.eta_pi.ravel(), model.eta_A.ravel(), model.eta_B.ravel()]
    )
    observed = {}

    class Result:
        x = start
        fun = 0.0
        nit = 1
        success = True

    def fake_minimize(fun, params, method=None, jac=None, options=None):
        value, grad = fun(np.full_like(params, np.inf))
        observed["value"] = value
        observed["grad"] = grad.copy()
        return Result()

    monkeypatch.setattr(nhmm_module, "minimize", fake_minimize)

    model.fit(n_iter=1)

    assert observed["value"] == np.inf
    assert np.allclose(observed["grad"], 0.0)
    assert np.allclose(
        np.concatenate([model.eta_pi.ravel(), model.eta_A.ravel(), model.eta_B.ravel()]),
        start,
    )


def test_nhmm_rejects_overflowed_linear_predictors():
    wide = pd.DataFrame({"id": [1], "t1": ["A"], "t2": ["B"]})
    seq = SequenceData(wide, time=["t1", "t2"], states=["A", "B"], id_col="id")
    X = np.full((1, 2, 1), 1e308)
    eta_pi = np.full((1, 2), 1e308)
    eta_A = np.zeros((1, 2, 2))
    eta_B = np.zeros((1, 2, 2))

    model = build_nhmm(
        seq,
        n_states=2,
        X=X,
        eta_pi=eta_pi,
        eta_A=eta_A,
        eta_B=eta_B,
    )

    with pytest.raises(ValueError, match="linear predictors"):
        model._compute_probs()
