import numpy as np
import pandas as pd
import pytest

from sequenzo import SequenceData
from sequenzo.seqhmm import (
    MNHMM,
    aic,
    build_mnhmm,
    bic,
    compare_models,
    compute_n_observations,
    compute_n_parameters,
    estimate_mnhmm,
    get_emission_probs,
    hidden_paths,
    get_initial_probs,
    get_transition_probs,
    simulate_mnhmm,
)
from sequenzo.seqhmm.formulas import Formula
from sequenzo.seqhmm.mnhmm import (
    _unique_equal_length_sequences,
    seqhmm_contrast_matrix,
)


STATES = ["A", "B"]
TIME_COLS = ["t1", "t2", "t3", "t4", "t5"]


def _make_seqdata():
    data = pd.DataFrame(
        [
            ["s1", "A", "A", "A", "A", "A"],
            ["s2", "A", "A", "B", "A", "A"],
            ["s3", "B", "B", "B", "B", "B"],
            ["s4", "B", "B", "A", "B", "B"],
        ],
        columns=["id", *TIME_COLS],
    )
    return SequenceData(data, time=TIME_COLS, states=STATES, id_col="id")


def _fixed_parameters():
    initial_probs = [
        np.array([0.95, 0.05]),
        np.array([0.05, 0.95]),
    ]
    transition_probs = [
        np.array([[0.98, 0.02], [0.10, 0.90]]),
        np.array([[0.90, 0.10], [0.02, 0.98]]),
    ]
    emission_probs = [
        np.array([[0.98, 0.02], [0.30, 0.70]]),
        np.array([[0.70, 0.30], [0.02, 0.98]]),
    ]
    return initial_probs, transition_probs, emission_probs


def _make_multichannel_seqdata():
    ch1 = SequenceData(
        pd.DataFrame(
            [
                ["s1", "A", "A", "A", "A", "A"],
                ["s2", "A", "A", "B", "A", "A"],
                ["s3", "B", "B", "B", "B", "B"],
                ["s4", "B", "B", "A", "B", "B"],
            ],
            columns=["id", *TIME_COLS],
        ),
        time=TIME_COLS,
        states=["A", "B"],
        id_col="id",
    )
    ch2 = SequenceData(
        pd.DataFrame(
            [
                ["s1", "X", "X", "X", "X", "X"],
                ["s2", "X", "X", "Y", "X", "X"],
                ["s3", "Y", "Y", "Y", "Y", "Y"],
                ["s4", "Y", "Y", "X", "Y", "Y"],
            ],
            columns=["id", *TIME_COLS],
        ),
        time=TIME_COLS,
        states=["X", "Y"],
        id_col="id",
    )
    return [ch1, ch2]


def _make_repeated_seqdata():
    rows = [
        ["s1", "A", "A", "A", "A", "A"],
        ["s2", "A", "A", "A", "A", "A"],
        ["s3", "A", "A", "B", "A", "A"],
        ["s4", "A", "A", "B", "A", "A"],
        ["s5", "B", "B", "B", "B", "B"],
        ["s6", "B", "B", "B", "B", "B"],
        ["s7", "B", "B", "A", "B", "B"],
        ["s8", "B", "B", "A", "B", "B"],
    ]
    data = pd.DataFrame(rows, columns=["id", *TIME_COLS])
    return SequenceData(data, time=TIME_COLS, states=STATES, id_col="id")


def _make_repeated_multichannel_seqdata():
    ch1_rows = [
        ["s1", "A", "A", "A", "A", "A"],
        ["s2", "A", "A", "A", "A", "A"],
        ["s3", "A", "A", "B", "A", "A"],
        ["s4", "A", "A", "B", "A", "A"],
        ["s5", "B", "B", "B", "B", "B"],
        ["s6", "B", "B", "B", "B", "B"],
        ["s7", "B", "B", "A", "B", "B"],
        ["s8", "B", "B", "A", "B", "B"],
    ]
    ch2_rows = [
        ["s1", "X", "X", "X", "X", "X"],
        ["s2", "X", "X", "X", "X", "X"],
        ["s3", "X", "X", "Y", "X", "X"],
        ["s4", "X", "X", "Y", "X", "X"],
        ["s5", "Y", "Y", "Y", "Y", "Y"],
        ["s6", "Y", "Y", "Y", "Y", "Y"],
        ["s7", "Y", "Y", "X", "Y", "Y"],
        ["s8", "Y", "Y", "X", "Y", "Y"],
    ]
    ch1 = SequenceData(
        pd.DataFrame(ch1_rows, columns=["id", *TIME_COLS]),
        time=TIME_COLS,
        states=["A", "B"],
        id_col="id",
    )
    ch2 = SequenceData(
        pd.DataFrame(ch2_rows, columns=["id", *TIME_COLS]),
        time=TIME_COLS,
        states=["X", "Y"],
        id_col="id",
    )
    return [ch1, ch2]


def _multichannel_fixed_parameters():
    initial_probs = [
        np.array([0.90, 0.10]),
        np.array([0.10, 0.90]),
    ]
    transition_probs = [
        np.array([[0.85, 0.15], [0.25, 0.75]]),
        np.array([[0.70, 0.30], [0.10, 0.90]]),
    ]
    emission_probs = [
        [
            np.array([[0.98, 0.02], [0.70, 0.30]]),
            np.array([[0.97, 0.03], [0.65, 0.35]]),
        ],
        [
            np.array([[0.30, 0.70], [0.02, 0.98]]),
            np.array([[0.35, 0.65], [0.03, 0.97]]),
        ],
    ]
    return initial_probs, transition_probs, emission_probs


def _manual_multichannel_loglik(initial, transition, emissions, obs_by_channel):
    per_time_emission = np.ones((len(obs_by_channel[0]), len(initial)), dtype=float)
    for obs, emission in zip(obs_by_channel, emissions):
        per_time_emission *= emission[:, obs].T

    alpha = initial * per_time_emission[0]
    for t in range(1, len(per_time_emission)):
        alpha = (alpha @ transition) * per_time_emission[t]
    return np.log(alpha.sum())


def _brute_force_multichannel_loglik(initial, transition, emissions, obs_by_channel):
    n_states = len(initial)
    n_timepoints = len(obs_by_channel[0])
    total = 0.0
    for path_index in range(n_states ** n_timepoints):
        path = []
        value = path_index
        for _ in range(n_timepoints):
            path.append(value % n_states)
            value //= n_states

        probability = float(initial[path[0]])
        for obs, emission in zip(obs_by_channel, emissions):
            probability *= float(emission[path[0], obs[0]])
        for t in range(1, n_timepoints):
            probability *= float(transition[path[t - 1], path[t]])
            for obs, emission in zip(obs_by_channel, emissions):
                probability *= float(emission[path[t], obs[t]])
        total += probability
    return np.log(total)


def _brute_force_multichannel_timevarying_loglik(
    initial,
    transition,
    emissions,
    obs_by_channel,
):
    n_states = len(initial)
    n_timepoints = len(obs_by_channel[0])
    total = 0.0
    for path_index in range(n_states ** n_timepoints):
        path = []
        value = path_index
        for _ in range(n_timepoints):
            path.append(value % n_states)
            value //= n_states

        probability = float(initial[path[0]])
        for obs, emission in zip(obs_by_channel, emissions):
            probability *= float(emission[0, path[0], obs[0]])
        for t in range(1, n_timepoints):
            probability *= float(transition[t, path[t - 1], path[t]])
            for obs, emission in zip(obs_by_channel, emissions):
                probability *= float(emission[t, path[t], obs[t]])
        total += probability
    return np.log(total)


def _full_eta_pi_from_reduced(reduced):
    return [(seqhmm_contrast_matrix(item.shape[0] + 1) @ item).T for item in reduced]


def _full_eta_A_from_reduced(reduced):
    full = []
    for item in reduced:
        q = seqhmm_contrast_matrix(item.shape[0] + 1)
        n_covariates = item.shape[1]
        n_states = item.shape[2]
        out = np.zeros((n_covariates, n_states, n_states), dtype=float)
        for origin in range(n_states):
            out[:, origin, :] = (q @ item[:, :, origin]).T
        full.append(out)
    return full


def _full_eta_B_from_reduced(reduced):
    full = []
    for item in reduced:
        q = seqhmm_contrast_matrix(item.shape[0] + 1)
        n_covariates = item.shape[1]
        n_states = item.shape[2]
        n_symbols = item.shape[0] + 1
        out = np.zeros((n_covariates, n_states, n_symbols), dtype=float)
        for state in range(n_states):
            out[:, state, :] = (q @ item[:, :, state]).T
        full.append(out)
    return full


def test_seqhmm_contrast_matrix_matches_R_sum_to_zero_basis():
    q2 = seqhmm_contrast_matrix(2)
    np.testing.assert_allclose(q2, np.array([[1.0 / np.sqrt(2.0)], [-1.0 / np.sqrt(2.0)]]))

    q3 = seqhmm_contrast_matrix(3)
    np.testing.assert_allclose(
        q3,
        np.array(
            [
                [1.0 / np.sqrt(2.0), -1.0 / np.sqrt(6.0)],
                [0.0, np.sqrt(2.0 / 3.0)],
                [-1.0 / np.sqrt(2.0), -1.0 / np.sqrt(6.0)],
            ]
        ),
        atol=1e-12,
    )

    q4 = seqhmm_contrast_matrix(4)
    np.testing.assert_allclose(
        q4,
        np.array(
            [
                [1.0 / np.sqrt(2.0), -1.0 / np.sqrt(6.0), -1.0 / np.sqrt(12.0)],
                [0.0, np.sqrt(2.0 / 3.0), -1.0 / np.sqrt(12.0)],
                [0.0, 0.0, np.sqrt(3.0) / 2.0],
                [-1.0 / np.sqrt(2.0), -1.0 / np.sqrt(6.0), -1.0 / np.sqrt(12.0)],
            ]
        ),
        atol=1e-12,
    )
    np.testing.assert_allclose(q4.T @ q4, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(q4.sum(axis=0), np.zeros(3), atol=1e-12)


def test_build_mnhmm_accepts_seqhmm_reduced_eta_coefficients():
    seqdata = _make_seqdata()
    n_sequences = len(seqdata.sequences)
    n_timepoints = len(TIME_COLS)
    X = np.ones((n_sequences, n_timepoints, 2), dtype=float)
    X[:, :, 1] = np.linspace(-1.0, 1.0, n_timepoints)
    X_cluster = np.array(
        [
            [1.0, -1.0],
            [1.0, -0.5],
            [1.0, 0.5],
            [1.0, 1.0],
        ]
    )
    eta_pi_reduced = [
        np.array([[0.2, -0.1], [-0.15, 0.05]]),
        np.array([[-0.3, 0.2], [0.1, -0.25]]),
    ]
    eta_A_reduced = [
        np.array(
            [
                [[0.10, -0.20, 0.30], [0.05, 0.15, -0.05]],
                [[-0.25, 0.10, 0.05], [0.20, -0.10, 0.15]],
            ]
        ),
        np.array(
            [
                [[-0.05, 0.20, -0.15], [0.15, -0.25, 0.05]],
                [[0.30, -0.05, 0.10], [-0.10, 0.05, -0.20]],
            ]
        ),
    ]
    eta_B_reduced = [
        np.array([[[0.2, -0.1, 0.3], [-0.05, 0.15, -0.2]]]),
        np.array([[[-0.3, 0.25, -0.1], [0.2, -0.15, 0.05]]]),
    ]
    eta_omega_reduced = np.array([[0.4, -0.2]])

    reduced_model = build_mnhmm(
        observations=seqdata,
        n_states=3,
        n_clusters=2,
        X_pi=X,
        X_A=X,
        X_B=X,
        X_cluster=X_cluster,
        eta_pi_reduced=eta_pi_reduced,
        eta_A_reduced=eta_A_reduced,
        eta_B_reduced=eta_B_reduced,
        eta_omega_reduced=eta_omega_reduced,
    )
    full_model = build_mnhmm(
        observations=seqdata,
        n_states=3,
        n_clusters=2,
        X_pi=X,
        X_A=X,
        X_B=X,
        X_cluster=X_cluster,
        eta_pi=_full_eta_pi_from_reduced(eta_pi_reduced),
        eta_A=_full_eta_A_from_reduced(eta_A_reduced),
        eta_B=_full_eta_B_from_reduced(eta_B_reduced),
        eta_omega=(seqhmm_contrast_matrix(2) @ eta_omega_reduced).T,
    )

    np.testing.assert_allclose(reduced_model.compute_cluster_probs(), full_model.compute_cluster_probs())
    np.testing.assert_allclose(reduced_model.score(), full_model.score())


def test_fit_preserves_supplied_intercept_reduced_eta_coefficients():
    seqdata = _make_seqdata()
    initial_probs, transition_probs, _ = _fixed_parameters()
    model = build_mnhmm(
        observations=seqdata,
        n_states=2,
        n_clusters=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        eta_B_reduced=[
            np.array([[[0.7, -0.4]]]),
            np.array([[[-0.2, 0.5]]]),
        ],
        cluster_probs=np.array([0.55, 0.45]),
        random_state=7,
    )
    before_score = model.score()

    fitted = model.fit(n_iter=0, tol=0.0)

    assert fitted.estimation_method == "direct_l_bfgs"
    assert fitted.emission_probs is None
    np.testing.assert_allclose(fitted.score(), before_score, atol=1e-12)


def test_objective_and_gradient_does_not_mutate_model_parameters():
    seqdata = _make_seqdata()
    initial_probs, transition_probs, _ = _fixed_parameters()
    model = build_mnhmm(
        observations=seqdata,
        n_states=2,
        n_clusters=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        eta_B_reduced=[
            np.array([[[0.7, -0.4]]]),
            np.array([[[-0.2, 0.5]]]),
        ],
        cluster_probs=np.array([0.55, 0.45]),
    )
    eta_before = [eta.copy() for eta in model.eta_B]
    score_before = model.score()

    first = model.objective_and_gradient(lambda_penalty=0.2)
    second = model.objective_and_gradient(lambda_penalty=0.2)

    for before, after in zip(eta_before, model.eta_B):
        np.testing.assert_allclose(after, before)
    np.testing.assert_allclose(model.score(), score_before, atol=1e-12)
    np.testing.assert_allclose(first["parameters"], second["parameters"])
    np.testing.assert_allclose(first["gradient"], second["gradient"])
    np.testing.assert_allclose(first["objective"], second["objective"])


def test_mnhmm_parameter_unpack_rejects_bad_parameter_vectors():
    seq = _make_seqdata()
    X_B = np.ones((len(seq.sequences), len(seq.time), 2), dtype=float)
    X_B[:, :, 1] = np.linspace(-0.5, 0.5, len(seq.time))
    model = build_mnhmm(
        seq,
        n_states=2,
        n_clusters=2,
        X_B=X_B,
        initial_probs=[np.array([0.8, 0.2]), np.array([0.3, 0.7])],
        transition_probs=[
            np.array([[0.9, 0.1], [0.2, 0.8]]),
            np.array([[0.7, 0.3], [0.4, 0.6]]),
        ],
        cluster_probs=np.array([0.6, 0.4]),
        random_state=1,
    )
    params, entries = model._pack_covariate_parameters()

    with pytest.raises(ValueError, match="parameter vector length"):
        model._unpack_covariate_parameters(np.r_[params, 0.0], entries)

    bad = params.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model._unpack_covariate_parameters(bad, entries)


def test_unique_equal_length_sequences_counts_repeated_rows():
    ch1 = np.array([0, 1, 0, 1, 1, 1, 0, 1], dtype=int)
    ch2 = np.array([1, 0, 1, 0, 0, 0, 1, 0], dtype=int)
    lengths = np.array([2, 2, 2, 2], dtype=int)

    unique_observations, counts = _unique_equal_length_sequences([ch1, ch2], lengths)

    observed = {
        (
            tuple(unique_observations[0][idx].tolist()),
            tuple(unique_observations[1][idx].tolist()),
        ): int(counts[idx])
        for idx in range(len(counts))
    }
    assert observed == {
        ((0, 1), (1, 0)): 3,
        ((1, 1), (0, 0)): 1,
    }


def test_build_mnhmm_formula_object_strips_response_lhs():
    seqdata = _make_seqdata()
    initial_probs, transition_probs, _ = _fixed_parameters()
    long_rows = []
    for seq_idx, seq_id in enumerate(seqdata.ids):
        for time_idx, time_label in enumerate(TIME_COLS):
            long_rows.append(
                {
                    "id": seq_id,
                    "time": time_label,
                    "trend": float(time_idx),
                    "activity": seqdata.sequences[seq_idx][time_idx],
                }
            )
    long_data = pd.DataFrame(long_rows)
    expected_X_B = np.ones((len(seqdata.sequences), len(TIME_COLS), 2), dtype=float)
    expected_X_B[:, :, 1] = np.arange(len(TIME_COLS), dtype=float)

    formula_model = build_mnhmm(
        observations=seqdata,
        n_states=2,
        n_clusters=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_formula=Formula("activity ~ trend"),
        data=long_data,
        id_var="id",
        time_var="time",
        eta_B_reduced=[
            np.array([[[0.7, -0.4], [0.1, -0.2]]]),
            np.array([[[-0.2, 0.5], [0.3, -0.1]]]),
        ],
        cluster_probs=np.array([0.55, 0.45]),
    )
    direct_model = build_mnhmm(
        observations=seqdata,
        n_states=2,
        n_clusters=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        X_B=expected_X_B,
        eta_B_reduced=[
            np.array([[[0.7, -0.4], [0.1, -0.2]]]),
            np.array([[[-0.2, 0.5], [0.3, -0.1]]]),
        ],
        cluster_probs=np.array([0.55, 0.45]),
    )

    np.testing.assert_allclose(formula_model.X_B, expected_X_B)
    np.testing.assert_allclose(formula_model.score(), direct_model.score(), atol=1e-12)


def test_build_mnhmm_accepts_flat_state_names_for_all_clusters():
    seqdata = _make_seqdata()
    initial_probs, transition_probs, emission_probs = _fixed_parameters()

    model = build_mnhmm(
        observations=seqdata,
        n_states=2,
        n_clusters=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        cluster_probs=np.array([0.5, 0.5]),
        state_names=["low", "high"],
    )

    assert model.state_names == [["low", "high"], ["low", "high"]]


def test_build_mnhmm_accepts_fixed_component_parameters():
    seqdata = _make_seqdata()
    initial_probs, transition_probs, emission_probs = _fixed_parameters()

    model = build_mnhmm(
        observations=seqdata,
        n_states=2,
        n_clusters=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        cluster_probs=np.array([0.5, 0.5]),
        cluster_names=["A-like", "B-like"],
        state_names=[["A1", "A2"], ["B1", "B2"]],
    )

    assert isinstance(model, MNHMM)
    assert model.n_clusters == 2
    assert model.n_states == [2, 2]
    assert model.cluster_names == ["A-like", "B-like"]
    assert model.has_complete_parameters
    assert np.isfinite(model.score())
    assert len(get_initial_probs(model)) == 2
    assert get_transition_probs(model, cluster=0).shape == (4, 5, 2, 2)
    assert get_emission_probs(model, cluster=0).shape == (4, 5, 2, 2)


def test_mnhmm_model_comparison_counts_parameters_and_observations():
    seqdata = _make_seqdata()
    initial_probs, transition_probs, emission_probs = _fixed_parameters()
    model = build_mnhmm(
        seqdata,
        n_states=[2, 2],
        n_clusters=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        cluster_probs=np.array([0.45, 0.55]),
    )
    log_likelihood = model.score()
    n_observations = len(seqdata.sequences) * len(TIME_COLS)

    assert compute_n_parameters(model) == 11
    assert compute_n_observations(model) == n_observations
    assert aic(model, log_likelihood=log_likelihood) == pytest.approx(
        -2.0 * log_likelihood + 2.0 * 11
    )
    assert bic(model, log_likelihood=log_likelihood) == pytest.approx(
        -2.0 * log_likelihood + np.log(n_observations) * 11
    )
    comparison = compare_models([model], criterion="AIC")
    assert comparison["best_model"] == "Model 1"
    assert comparison["models"][0]["n_parameters"] == 11


def test_build_mnhmm_cluster_formula_accepts_one_row_per_sequence():
    seqdata = _make_seqdata()
    initial_probs, transition_probs, emission_probs = _fixed_parameters()
    data = pd.DataFrame(
        {
            "id": seqdata.ids,
            "group": [0.0, 0.0, 1.0, 1.0],
        }
    )

    model = build_mnhmm(
        seqdata,
        n_states=[2, 2],
        n_clusters=2,
        cluster_formula="~ group",
        data=data,
        id_var="id",
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        eta_omega=np.array([[0.0, 0.0], [-1.0, 1.0]]),
    )

    np.testing.assert_allclose(model.X_cluster[:, 0], np.ones(len(seqdata.sequences)))
    np.testing.assert_allclose(model.X_cluster[:, 1], np.array([0.0, 0.0, 1.0, 1.0]))
    assert np.isfinite(model.score())


def test_build_mnhmm_accepts_tuple_multichannel_observations():
    channels = tuple(_make_multichannel_seqdata())
    initial_probs, transition_probs, emission_probs = _multichannel_fixed_parameters()

    model = build_mnhmm(
        channels,
        n_states=[2, 2],
        n_clusters=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        cluster_probs=np.array([0.5, 0.5]),
    )

    assert model.n_channels == 2
    assert np.isfinite(model.score(channels))


def test_fixed_parameter_responsibilities_and_prediction():
    seqdata = _make_seqdata()
    initial_probs, transition_probs, emission_probs = _fixed_parameters()
    model = build_mnhmm(
        observations=seqdata,
        n_states=2,
        n_clusters=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        cluster_probs=np.array([0.5, 0.5]),
    )

    responsibilities = model.compute_responsibilities()

    assert responsibilities.shape == (4, 2)
    np.testing.assert_allclose(responsibilities.sum(axis=1), np.ones(4))
    assert responsibilities[0, 0] > responsibilities[0, 1]
    assert responsibilities[1, 0] > responsibilities[1, 1]
    assert responsibilities[2, 1] > responsibilities[2, 0]
    assert responsibilities[3, 1] > responsibilities[3, 0]
    np.testing.assert_array_equal(model.predict_cluster(), np.array([0, 0, 1, 1]))


def test_mnhmm_rejects_fixed_probabilities_that_do_not_sum_to_one():
    seqdata = _make_seqdata()
    initial_probs, transition_probs, emission_probs = _fixed_parameters()

    with pytest.raises(ValueError, match="cluster_probs must sum to 1.0"):
        build_mnhmm(
            observations=seqdata,
            n_states=2,
            n_clusters=2,
            initial_probs=initial_probs,
            transition_probs=transition_probs,
            emission_probs=emission_probs,
            cluster_probs=np.array([2.0, 1.0]),
        )

    bad_transition = [transition_probs[0].copy(), transition_probs[1].copy()]
    bad_transition[0][0] = np.array([2.0, 1.0])
    with pytest.raises(ValueError, match="transition_probs\\[0\\] rows must sum to 1.0"):
        build_mnhmm(
            observations=seqdata,
            n_states=2,
            n_clusters=2,
            initial_probs=initial_probs,
            transition_probs=bad_transition,
            emission_probs=emission_probs,
            cluster_probs=np.array([0.5, 0.5]),
        )


def test_mnhmm_fixed_probability_score_preserves_structural_zeros():
    seqdata = SequenceData(
        pd.DataFrame([["s1", "A", "A"]], columns=["id", "t1", "t2"]),
        time=["t1", "t2"],
        states=["A", "B"],
        id_col="id",
    )
    model = build_mnhmm(
        observations=seqdata,
        n_states=[1, 1],
        n_clusters=2,
        initial_probs=[np.array([1.0]), np.array([1.0])],
        transition_probs=[np.array([[1.0]]), np.array([[1.0]])],
        emission_probs=[np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])],
        cluster_probs=np.array([0.0, 1.0]),
    )

    assert model.score() == -np.inf
    assert model.score(compress=True) == -np.inf
    with pytest.raises(ValueError, match="impossible sequences"):
        model.compute_responsibilities()


def test_multichannel_fixed_parameter_mnhmm_scores_independent_emissions():
    channels = _make_multichannel_seqdata()
    initial_probs, transition_probs, emission_probs = _multichannel_fixed_parameters()
    cluster_probs = np.array([0.5, 0.5])

    model = build_mnhmm(
        observations=channels,
        n_states=2,
        n_clusters=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        cluster_probs=cluster_probs,
        cluster_names=["mostly-AX", "mostly-BY"],
    )

    responsibilities = model.compute_responsibilities()

    assert model.n_channels == 2
    assert model.n_symbols == [2, 2]
    exported_emissions = get_emission_probs(model, cluster=0)
    assert len(exported_emissions) == 2
    assert exported_emissions[0].shape == (4, 5, 2, 2)
    assert exported_emissions[1].shape == (4, 5, 2, 2)
    np.testing.assert_allclose(exported_emissions[0][0, 0], emission_probs[0][0])
    np.testing.assert_allclose(exported_emissions[1][0, 0], emission_probs[0][1])
    assert responsibilities.shape == (4, 2)
    np.testing.assert_allclose(responsibilities.sum(axis=1), np.ones(4))
    np.testing.assert_array_equal(model.predict_cluster(), np.array([0, 0, 1, 1]))

    expected_terms = []
    for seq_idx in range(len(channels[0].sequences)):
        obs_by_channel = [
            np.asarray(channel.sequences[seq_idx], dtype=int) - 1
            for channel in channels
        ]
        component_terms = []
        for cluster_idx in range(2):
            component_terms.append(
                np.log(cluster_probs[cluster_idx])
                + _manual_multichannel_loglik(
                    initial_probs[cluster_idx],
                    transition_probs[cluster_idx],
                    emission_probs[cluster_idx],
                    obs_by_channel,
                )
            )
            np.testing.assert_allclose(
                _manual_multichannel_loglik(
                    initial_probs[cluster_idx],
                    transition_probs[cluster_idx],
                    emission_probs[cluster_idx],
                    obs_by_channel,
                ),
                _brute_force_multichannel_loglik(
                    initial_probs[cluster_idx],
                    transition_probs[cluster_idx],
                    emission_probs[cluster_idx],
                    obs_by_channel,
                ),
                atol=1e-12,
            )
        expected_terms.append(np.logaddexp(component_terms[0], component_terms[1]))
    np.testing.assert_allclose(model.score(), np.sum(expected_terms), atol=1e-12)


def test_multichannel_mnhmm_score_compression_matches_baseline():
    channels = _make_multichannel_seqdata()
    repeated_channels = []
    for channel in channels:
        copies = []
        for rep in range(6):
            copy = channel.data[["id", *TIME_COLS]].copy()
            copy["id"] = [f"{seq_id}_{rep}" for seq_id in channel.ids]
            copies.append(copy)
        wide = pd.concat(copies, ignore_index=True)
        repeated_channels.append(
            SequenceData(wide, time=TIME_COLS, states=channel.alphabet, id_col="id")
        )
    initial_probs, transition_probs, emission_probs = _multichannel_fixed_parameters()
    model = build_mnhmm(
        observations=repeated_channels,
        n_states=2,
        n_clusters=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        cluster_probs=np.array([0.55, 0.45]),
    )

    np.testing.assert_allclose(model.score(compress=True), model.score(), atol=1e-10)


def test_single_channel_mnhmm_score_compression_matches_baseline():
    seq = _make_seqdata()
    copies = []
    for rep in range(8):
        copy = seq.data[["id", *TIME_COLS]].copy()
        copy["id"] = [f"{seq_id}_{rep}" for seq_id in seq.ids]
        copies.append(copy)
    repeated = SequenceData(
        pd.concat(copies, ignore_index=True),
        time=TIME_COLS,
        states=STATES,
        id_col="id",
    )
    initial_probs, transition_probs, emission_probs = _fixed_parameters()
    model = build_mnhmm(
        observations=repeated,
        n_states=2,
        n_clusters=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        cluster_probs=np.array([0.55, 0.45]),
    )

    np.testing.assert_allclose(model.score(compress=True), model.score(), atol=1e-10)


def test_mnhmm_score_compression_falls_back_for_covariate_probabilities():
    seq = _make_seqdata()
    X_B = np.ones((len(seq.sequences), len(TIME_COLS), 2), dtype=float)
    X_B[:, :, 1] = np.linspace(-0.5, 0.5, len(TIME_COLS))
    initial_probs, transition_probs, _ = _fixed_parameters()
    model = build_mnhmm(
        observations=seq,
        n_states=2,
        n_clusters=2,
        X_B=X_B,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        eta_B_reduced=[
            np.array([[[0.40, -0.25], [0.10, 0.05]]]),
            np.array([[[-0.20, 0.30], [0.15, -0.10]]]),
        ],
        cluster_probs=np.array([0.55, 0.45]),
    )

    np.testing.assert_allclose(model.score(compress=True), model.score(), atol=1e-12)


def test_single_channel_mnhmm_fit_compression_matches_direct_em():
    seqdata = _make_repeated_seqdata()

    direct = build_mnhmm(
        observations=seqdata,
        n_states=2,
        n_clusters=2,
        random_state=123,
    ).fit(n_iter=3, tol=0.0, compress=False)
    compressed = build_mnhmm(
        observations=seqdata,
        n_states=2,
        n_clusters=2,
        random_state=123,
    ).fit(n_iter=3, tol=0.0, compress=True)

    np.testing.assert_allclose(compressed.log_likelihood, direct.log_likelihood, atol=1e-10)
    np.testing.assert_allclose(compressed.cluster_probs, direct.cluster_probs, atol=1e-10)
    np.testing.assert_allclose(compressed.responsibilities, direct.responsibilities, atol=1e-10)
    for cluster_idx in range(2):
        np.testing.assert_allclose(
            compressed.initial_probs[cluster_idx],
            direct.initial_probs[cluster_idx],
            atol=1e-10,
        )
        np.testing.assert_allclose(
            compressed.transition_probs[cluster_idx],
            direct.transition_probs[cluster_idx],
            atol=1e-10,
        )
        np.testing.assert_allclose(
            compressed.emission_probs[cluster_idx],
            direct.emission_probs[cluster_idx],
            atol=1e-10,
        )


def test_multichannel_mnhmm_fit_compression_matches_direct_em():
    channels = _make_repeated_multichannel_seqdata()

    direct = estimate_mnhmm(
        observations=channels,
        n_states=2,
        n_clusters=2,
        random_state=321,
        n_iter=3,
        tol=0.0,
        compress=False,
    )
    compressed = estimate_mnhmm(
        observations=channels,
        n_states=2,
        n_clusters=2,
        random_state=321,
        n_iter=3,
        tol=0.0,
        compress=True,
    )

    np.testing.assert_allclose(compressed.log_likelihood, direct.log_likelihood, atol=1e-10)
    np.testing.assert_allclose(compressed.cluster_probs, direct.cluster_probs, atol=1e-10)
    np.testing.assert_allclose(compressed.responsibilities, direct.responsibilities, atol=1e-10)
    for cluster_idx in range(2):
        np.testing.assert_allclose(
            compressed.initial_probs[cluster_idx],
            direct.initial_probs[cluster_idx],
            atol=1e-10,
        )
        np.testing.assert_allclose(
            compressed.transition_probs[cluster_idx],
            direct.transition_probs[cluster_idx],
            atol=1e-10,
        )
        for channel_idx in range(2):
            np.testing.assert_allclose(
                compressed.emission_probs[cluster_idx][channel_idx],
                direct.emission_probs[cluster_idx][channel_idx],
                atol=1e-10,
            )


def test_mnhmm_fit_compression_matches_direct_for_covariate_probabilities():
    seqdata = _make_repeated_seqdata()
    X = np.ones((len(seqdata.sequences), len(TIME_COLS), 2), dtype=float)
    X[:, :, 1] = np.linspace(-1.0, 1.0, len(TIME_COLS))

    direct = build_mnhmm(
        observations=seqdata,
        n_states=2,
        n_clusters=2,
        X_B=X,
        random_state=123,
    ).fit(n_iter=3, tol=0.0, compress=False)
    compressed = build_mnhmm(
        observations=seqdata,
        n_states=2,
        n_clusters=2,
        X_B=X,
        random_state=123,
    ).fit(n_iter=3, tol=0.0, compress=True)

    np.testing.assert_allclose(compressed.log_likelihood, direct.log_likelihood, atol=1e-8)
    np.testing.assert_allclose(compressed.responsibilities, direct.responsibilities, atol=1e-8)
    for cluster_idx in range(2):
        np.testing.assert_allclose(
            compressed.eta_B[cluster_idx],
            direct.eta_B[cluster_idx],
            atol=1e-8,
        )
    assert compressed.estimation_method == "direct_l_bfgs"


def test_multichannel_mnhmm_covariate_fit_compression_matches_direct():
    channels = _make_repeated_multichannel_seqdata()
    n_sequences = len(channels[0].sequences)
    X = np.ones((n_sequences, len(TIME_COLS), 2), dtype=float)
    X[:, :, 1] = np.linspace(-1.0, 1.0, len(TIME_COLS))

    direct = build_mnhmm(
        observations=channels,
        n_states=2,
        n_clusters=2,
        X_B=X,
        random_state=123,
    ).fit(n_iter=2, tol=0.0, compress=False)
    compressed = build_mnhmm(
        observations=channels,
        n_states=2,
        n_clusters=2,
        X_B=X,
        random_state=123,
    ).fit(n_iter=2, tol=0.0, compress=True)

    np.testing.assert_allclose(compressed.log_likelihood, direct.log_likelihood, atol=1e-8)
    np.testing.assert_allclose(compressed.responsibilities, direct.responsibilities, atol=1e-8)
    for cluster_idx in range(2):
        for channel_idx in range(2):
            np.testing.assert_allclose(
                compressed.eta_B[cluster_idx][channel_idx],
                direct.eta_B[cluster_idx][channel_idx],
                atol=1e-8,
            )


def test_mnhmm_covariate_fit_compression_keeps_different_covariates_separate():
    seqdata = _make_repeated_seqdata()
    n_sequences = len(seqdata.sequences)
    X = np.ones((n_sequences, len(TIME_COLS), 2), dtype=float)
    X[:, :, 1] = np.linspace(-1.0, 1.0, len(TIME_COLS))
    X_pi = X.copy()
    X_A = X.copy()
    X_B = X.copy()
    X_cluster = np.ones((n_sequences, 2), dtype=float)
    X_pi[1, :, 1] = 4.0
    X_A[1, :, 1] = 5.0
    X_B[1, :, 1] = 6.0
    X_cluster[1, 1] = 7.0
    model = build_mnhmm(
        observations=seqdata,
        n_states=2,
        n_clusters=2,
        X_pi=X_pi,
        X_A=X_A,
        X_B=X_B,
        X_cluster=X_cluster,
        random_state=123,
    )
    model.enable_cluster_covariate_estimation(scale=0.0)

    indices, counts = model._covariate_direct_compression_indices()

    assert len(indices) == 5
    assert counts.tolist().count(2.0) == 3
    assert 0 in indices
    assert 1 in indices


def test_mnhmm_fixed_component_cluster_fit_compression_matches_direct():
    seqdata = _make_repeated_seqdata()
    initial_probs, transition_probs, emission_probs = _fixed_parameters()
    X_cluster = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )

    direct = estimate_mnhmm(
        observations=seqdata,
        n_states=2,
        n_clusters=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        X_cluster=X_cluster,
        random_state=321,
        n_iter=50,
        tol=1e-9,
        compress=False,
    )
    compressed = estimate_mnhmm(
        observations=seqdata,
        n_states=2,
        n_clusters=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        X_cluster=X_cluster,
        random_state=321,
        n_iter=50,
        tol=1e-9,
        compress=True,
    )

    np.testing.assert_allclose(compressed.log_likelihood, direct.log_likelihood, atol=1e-10)
    np.testing.assert_allclose(compressed.eta_omega, direct.eta_omega, atol=1e-8)
    np.testing.assert_allclose(compressed.responsibilities, direct.responsibilities, atol=1e-8)


def test_mnhmm_cluster_covariate_lambda_penalty_shrinks_eta():
    channels = _make_multichannel_seqdata()
    initial_probs, transition_probs, emission_probs = _multichannel_fixed_parameters()
    X_cluster = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )

    unpenalized = estimate_mnhmm(
        observations=channels,
        n_states=2,
        n_clusters=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        X_cluster=X_cluster,
        random_state=123,
        n_iter=80,
        tol=1e-9,
        lambda_penalty=0.0,
    )
    penalized = estimate_mnhmm(
        observations=channels,
        n_states=2,
        n_clusters=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        X_cluster=X_cluster,
        random_state=123,
        n_iter=80,
        tol=1e-9,
        lambda_penalty=50.0,
    )

    assert np.linalg.norm(penalized.eta_omega) < np.linalg.norm(unpenalized.eta_omega)
    assert penalized.lambda_penalty == 50.0


def test_mnhmm_fit_rejects_negative_lambda_penalty():
    seqdata = _make_seqdata()
    initial_probs, transition_probs, emission_probs = _fixed_parameters()
    model = build_mnhmm(
        observations=seqdata,
        n_states=2,
        n_clusters=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        X_cluster=np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        ),
        random_state=123,
    )

    with pytest.raises(ValueError, match="lambda_penalty"):
        model.fit(n_iter=1, lambda_penalty=-1.0)


def test_build_mnhmm_rejects_nonfinite_direct_covariates():
    seqdata = _make_seqdata()
    X = np.ones((len(seqdata.sequences), len(TIME_COLS), 2), dtype=float)
    X[0, 0, 1] = np.nan

    with pytest.raises(ValueError, match="finite"):
        build_mnhmm(
            observations=seqdata,
            n_states=2,
            n_clusters=2,
            X_B=X,
        )


def test_build_mnhmm_rejects_nonfinite_cluster_covariates():
    seqdata = _make_seqdata()
    X_cluster = np.array(
        [
            [1.0, 0.0],
            [1.0, np.inf],
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )

    with pytest.raises(ValueError, match="finite"):
        build_mnhmm(
            observations=seqdata,
            n_states=2,
            n_clusters=2,
            X_cluster=X_cluster,
        )


def test_single_channel_fixed_component_cluster_covariates_use_cached_optimizer(monkeypatch):
    seqdata = _make_seqdata()
    initial_probs, transition_probs, emission_probs = _fixed_parameters()
    model = build_mnhmm(
        observations=seqdata,
        n_states=2,
        n_clusters=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        X_cluster=np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        ),
        random_state=123,
    )

    def fail_generic_direct(*_args, **_kwargs):
        raise AssertionError("single-channel fixed components should cache likelihoods")

    monkeypatch.setattr(model, "_fit_covariate_direct", fail_generic_direct)
    fitted = model.fit(n_iter=20, tol=1e-9)

    assert fitted.estimation_method == "direct_l_bfgs"
    assert fitted.n_optimized_parameters == 2
    assert np.isfinite(fitted.log_likelihood)


def test_covariate_mnhmm_rejects_newdata_with_different_id_order():
    seq = _make_seqdata()
    X_B = np.ones((len(seq.sequences), len(TIME_COLS), 2), dtype=float)
    X_B[:, :, 1] = np.linspace(-0.5, 0.5, len(TIME_COLS))
    initial_probs, transition_probs, _ = _fixed_parameters()
    model = build_mnhmm(
        observations=seq,
        n_states=2,
        n_clusters=2,
        X_B=X_B,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        eta_B_reduced=[
            np.array([[[0.40, -0.25], [0.10, 0.05]]]),
            np.array([[[-0.20, 0.30], [0.15, -0.10]]]),
        ],
        cluster_probs=np.array([0.55, 0.45]),
    )
    reordered = SequenceData(
        seq.data[["id", *TIME_COLS]].iloc[::-1].reset_index(drop=True),
        time=TIME_COLS,
        states=STATES,
        id_col="id",
    )

    with pytest.raises(ValueError, match="same sequence IDs and time order"):
        model.score(sequences=reordered)


def test_hidden_paths_mnhmm_rejects_covariate_newdata_with_different_id_order():
    seq = _make_seqdata()
    X_B = np.ones((len(seq.sequences), len(TIME_COLS), 2), dtype=float)
    X_B[:, :, 1] = np.linspace(-0.5, 0.5, len(TIME_COLS))
    initial_probs, transition_probs, _ = _fixed_parameters()
    model = build_mnhmm(
        observations=seq,
        n_states=2,
        n_clusters=2,
        X_B=X_B,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        eta_B_reduced=[
            np.array([[[0.40, -0.25], [0.10, 0.05]]]),
            np.array([[[-0.20, 0.30], [0.15, -0.10]]]),
        ],
        cluster_probs=np.array([0.55, 0.45]),
    )
    reordered = SequenceData(
        seq.data[["id", *TIME_COLS]].iloc[::-1].reset_index(drop=True),
        time=TIME_COLS,
        states=STATES,
        id_col="id",
    )

    with pytest.raises(ValueError, match="same sequence IDs and time order"):
        hidden_paths(model, newdata=reordered)


def test_cluster_covariate_mnhmm_rejects_newdata_with_different_id_order():
    seq = _make_seqdata()
    initial_probs, transition_probs, emission_probs = _fixed_parameters()
    model = build_mnhmm(
        observations=seq,
        n_states=2,
        n_clusters=2,
        X_cluster=np.array(
            [
                [1.0, -1.0],
                [1.0, -0.5],
                [1.0, 0.5],
                [1.0, 1.0],
            ]
        ),
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        eta_omega=np.array([[0.0, 0.0], [-1.0, 1.0]]),
    )
    reordered = SequenceData(
        seq.data[["id", *TIME_COLS]].iloc[::-1].reset_index(drop=True),
        time=TIME_COLS,
        states=STATES,
        id_col="id",
    )

    with pytest.raises(ValueError, match="same sequence IDs and time order"):
        model.compute_responsibilities(reordered)


def test_fixed_probability_mnhmm_scores_newdata_with_different_sequence_count():
    seq = _make_seqdata()
    initial_probs, transition_probs, emission_probs = _fixed_parameters()
    model = build_mnhmm(
        observations=seq,
        n_states=2,
        n_clusters=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        cluster_probs=np.array([0.55, 0.45]),
    )
    new_sequences = SequenceData(
        pd.DataFrame(
            [
                ["n1", "A", "A", "A", "A", "A"],
                ["n2", "B", "B", "B", "B", "B"],
                ["n3", "A", "B", "A", "B", "A"],
            ],
            columns=["id", *TIME_COLS],
        ),
        time=TIME_COLS,
        states=STATES,
        id_col="id",
    )

    np.testing.assert_allclose(
        model.score(sequences=new_sequences),
        model.score(sequences=new_sequences, compress=True),
        atol=1e-12,
    )


def test_multichannel_fixed_probability_mnhmm_scores_newdata_with_different_count():
    channels = _make_multichannel_seqdata()
    initial_probs, transition_probs, emission_probs = _multichannel_fixed_parameters()
    model = build_mnhmm(
        observations=channels,
        n_states=2,
        n_clusters=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        cluster_probs=np.array([0.55, 0.45]),
    )
    ids = ["n1", "n2", "n3"]
    ch1 = SequenceData(
        pd.DataFrame(
            [
                ["n1", "A", "A", "A", "A", "A"],
                ["n2", "B", "B", "B", "B", "B"],
                ["n3", "A", "B", "A", "B", "A"],
            ],
            columns=["id", *TIME_COLS],
        ),
        time=TIME_COLS,
        states=["A", "B"],
        id_col="id",
    )
    ch2 = SequenceData(
        pd.DataFrame(
            [
                [ids[0], "X", "X", "X", "X", "X"],
                [ids[1], "Y", "Y", "Y", "Y", "Y"],
                [ids[2], "X", "Y", "X", "Y", "X"],
            ],
            columns=["id", *TIME_COLS],
        ),
        time=TIME_COLS,
        states=["X", "Y"],
        id_col="id",
    )

    np.testing.assert_allclose(
        model.score(sequences=[ch1, ch2]),
        model.score(sequences=[ch1, ch2], compress=True),
        atol=1e-12,
    )


def test_mnhmm_rejects_newdata_with_different_state_order():
    seq = _make_seqdata()
    initial_probs, transition_probs, emission_probs = _fixed_parameters()
    model = build_mnhmm(
        observations=seq,
        n_states=2,
        n_clusters=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        cluster_probs=np.array([0.55, 0.45]),
    )
    reordered = SequenceData(
        seq.data[["id", *TIME_COLS]].copy(),
        time=TIME_COLS,
        states=["B", "A"],
        id_col="id",
    )

    with pytest.raises(ValueError, match="alphabet"):
        model.score(sequences=reordered)


def test_multichannel_mnhmm_rejects_newdata_with_different_channel_state_order():
    channels = _make_multichannel_seqdata()
    initial_probs, transition_probs, emission_probs = _multichannel_fixed_parameters()
    model = build_mnhmm(
        observations=channels,
        n_states=2,
        n_clusters=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        cluster_probs=np.array([0.55, 0.45]),
    )
    reordered_ch2 = SequenceData(
        channels[1].data[["id", *TIME_COLS]].copy(),
        time=TIME_COLS,
        states=["Y", "X"],
        id_col="id",
    )

    with pytest.raises(ValueError, match="alphabet"):
        model.score(sequences=[channels[0], reordered_ch2])


def test_multichannel_mnhmm_rejects_misaligned_time_grid():
    channels = _make_multichannel_seqdata()
    reversed_ch2 = SequenceData(
        pd.DataFrame(
            [
                ["s1", "X", "X", "X", "X", "X"],
                ["s2", "X", "X", "Y", "X", "X"],
                ["s3", "Y", "Y", "Y", "Y", "Y"],
                ["s4", "Y", "Y", "X", "Y", "Y"],
            ],
            columns=["id", *TIME_COLS],
        ),
        time=list(reversed(TIME_COLS)),
        states=["X", "Y"],
        id_col="id",
    )
    initial_probs, transition_probs, emission_probs = _multichannel_fixed_parameters()

    with pytest.raises(ValueError, match="same time points in the same order"):
        build_mnhmm(
            observations=[channels[0], reversed_ch2],
            n_states=2,
            n_clusters=2,
            initial_probs=initial_probs,
            transition_probs=transition_probs,
            emission_probs=emission_probs,
            cluster_probs=np.array([0.5, 0.5]),
        )


def test_multichannel_mnhmm_rejects_single_channel_emission_shape():
    channels = _make_multichannel_seqdata()
    initial_probs, transition_probs, emission_probs = _fixed_parameters()

    with pytest.raises(ValueError, match="multichannel emission_probs"):
        build_mnhmm(
            observations=channels,
            n_states=2,
            n_clusters=2,
            initial_probs=initial_probs,
            transition_probs=transition_probs,
            emission_probs=emission_probs,
            cluster_probs=np.array([0.5, 0.5]),
        )


def test_mnhmm_rejects_time_varying_cluster_covariates():
    seqdata = _make_seqdata()
    X_cluster = np.ones((len(seqdata.sequences), len(TIME_COLS), 2), dtype=float)
    X_cluster[:, :, 1] = np.linspace(0.0, 1.0, len(TIME_COLS))

    with pytest.raises(ValueError, match="X_cluster must be time-constant"):
        build_mnhmm(
            observations=seqdata,
            n_states=2,
            n_clusters=2,
            X_cluster=X_cluster,
        )


def test_multichannel_mnhmm_scores_timevarying_component_covariates():
    channels = _make_multichannel_seqdata()
    initial_probs, transition_probs, _ = _multichannel_fixed_parameters()
    X_B = np.ones((len(channels[0].sequences), len(TIME_COLS), 2), dtype=float)
    X_B[:, :, 1] = np.linspace(-1.0, 1.0, len(TIME_COLS))

    model = build_mnhmm(
        observations=channels,
        n_states=2,
        n_clusters=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        X_B=X_B,
        eta_B_reduced=[
            [
                np.array([[[0.30, -0.10], [0.20, -0.25]]]),
                np.array([[[0.10, 0.25], [-0.15, 0.05]]]),
            ],
            [
                np.array([[[-0.20, 0.35], [0.15, -0.05]]]),
                np.array([[[0.25, -0.30], [-0.10, 0.20]]]),
            ],
        ],
        cluster_probs=np.array([0.55, 0.45]),
    )

    log_terms = []
    initial, transition, emissions = model._component_probs(0)
    for seq_idx in range(len(channels[0].sequences)):
        obs_by_channel = [
            np.asarray(channel.sequences[seq_idx], dtype=int) - 1
            for channel in channels
        ]
        log_terms.append(
            _brute_force_multichannel_timevarying_loglik(
                initial[seq_idx],
                transition[seq_idx],
                [emission[seq_idx] for emission in emissions],
                obs_by_channel,
            )
            + np.log(0.55)
        )

    initial, transition, emissions = model._component_probs(1)
    for seq_idx in range(len(channels[0].sequences)):
        obs_by_channel = [
            np.asarray(channel.sequences[seq_idx], dtype=int) - 1
            for channel in channels
        ]
        log_terms[seq_idx] = np.logaddexp(
            log_terms[seq_idx],
            _brute_force_multichannel_timevarying_loglik(
                initial[seq_idx],
                transition[seq_idx],
                [emission[seq_idx] for emission in emissions],
                obs_by_channel,
            )
            + np.log(0.45),
        )

    np.testing.assert_allclose(model.score(), np.sum(log_terms), atol=1e-12)


def test_estimate_multichannel_mnhmm_emission_covariates_use_direct_l_bfgs():
    channels = _make_multichannel_seqdata()
    initial_probs, transition_probs, _ = _multichannel_fixed_parameters()
    X_B = np.ones((len(channels[0].sequences), len(TIME_COLS), 2), dtype=float)
    X_B[:, :, 1] = np.linspace(-1.0, 1.0, len(TIME_COLS))
    eta_start = [
        [np.full((1, 2, 2), 0.5), np.full((1, 2, 2), -0.5)],
        [np.full((1, 2, 2), -0.5), np.full((1, 2, 2), 0.5)],
    ]
    start_model = build_mnhmm(
        observations=channels,
        n_states=2,
        n_clusters=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        X_B=X_B,
        eta_B_reduced=eta_start,
        cluster_probs=np.array([0.5, 0.5]),
    )
    start_objective = start_model.objective_and_gradient()["objective"]

    fitted = estimate_mnhmm(
        observations=channels,
        n_states=2,
        n_clusters=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        X_B=X_B,
        eta_B_reduced=eta_start,
        cluster_probs=np.array([0.5, 0.5]),
        n_iter=100,
        tol=1e-9,
    )

    assert fitted.estimation_method == "direct_l_bfgs"
    assert fitted.n_optimized_parameters == 16
    assert fitted.n_iter > 0
    assert fitted.optimization_result.nfev == fitted.optimization_result.njev
    assert fitted.objective_and_gradient()["objective"] < start_objective
    assert fitted.eta_omega is None
    assert fitted.responsibilities.shape == (4, 2)
    np.testing.assert_allclose(fitted.responsibilities.sum(axis=1), np.ones(4))
    assert np.isfinite(fitted.log_likelihood)
    for cluster_eta in fitted.eta_B:
        assert len(cluster_eta) == 2
        for eta in cluster_eta:
            assert eta.shape == (2, 2, 2)
            np.testing.assert_allclose(
                eta.sum(axis=2),
                np.zeros((eta.shape[0], eta.shape[1])),
                atol=1e-10,
            )


def test_multichannel_mnhmm_component_covariate_gradient_matches_finite_difference():
    channels = _make_multichannel_seqdata()
    initial_probs, transition_probs, _ = _multichannel_fixed_parameters()
    X_B = np.ones((len(channels[0].sequences), len(TIME_COLS), 2), dtype=float)
    X_B[:, :, 1] = np.linspace(-1.0, 1.0, len(TIME_COLS))
    model = build_mnhmm(
        observations=channels,
        n_states=2,
        n_clusters=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        X_B=X_B,
        eta_B_reduced=[
            [np.full((1, 2, 2), 0.5), np.full((1, 2, 2), -0.5)],
            [np.full((1, 2, 2), -0.25), np.full((1, 2, 2), 0.25)],
        ],
        cluster_probs=np.array([0.55, 0.45]),
    )

    params, entries = model._pack_covariate_parameters()
    analytic = model.objective_and_gradient()["gradient"]
    eps = 1e-6
    check_indices = [0, len(params) // 2, len(params) - 1]
    for index in check_indices:
        plus = params.copy()
        minus = params.copy()
        plus[index] += eps
        minus[index] -= eps
        model._unpack_covariate_parameters(plus, entries)
        plus_value = model.objective_and_gradient()["objective"]
        model._unpack_covariate_parameters(minus, entries)
        minus_value = model.objective_and_gradient()["objective"]
        numeric = (plus_value - minus_value) / (2.0 * eps)
        np.testing.assert_allclose(analytic[index], numeric, rtol=1e-4, atol=1e-5)

    model._unpack_covariate_parameters(params, entries)


def test_estimate_multichannel_mnhmm_cluster_covariates_are_optimized():
    channels = _make_multichannel_seqdata()
    initial_probs, transition_probs, emission_probs = _multichannel_fixed_parameters()

    fitted = estimate_mnhmm(
        observations=channels,
        n_states=2,
        n_clusters=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        X_cluster=np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        ),
        random_state=123,
        n_iter=100,
        tol=1e-9,
    )

    assert fitted.estimation_method == "fixed_multichannel_cluster_l_bfgs"
    assert fitted.n_iter > 0
    assert fitted.n_optimized_parameters == 2
    np.testing.assert_allclose(
        fitted.eta_omega.sum(axis=1),
        np.zeros(fitted.eta_omega.shape[0]),
        atol=1e-10,
    )
    prior = fitted.compute_cluster_probs()
    assert prior[0, 0] > 0.80
    assert prior[1, 0] > 0.80
    assert prior[2, 1] > 0.80
    assert prior[3, 1] > 0.80


def test_multichannel_fixed_components_fit_updates_unfixed_cluster_probs():
    time_cols = TIME_COLS
    ch1 = SequenceData(
        pd.DataFrame(
            [
                ["s1", "A", "A", "A", "A", "A"],
                ["s2", "A", "A", "A", "A", "A"],
                ["s3", "A", "A", "B", "A", "A"],
                ["s4", "B", "B", "B", "B", "B"],
            ],
            columns=["id", *time_cols],
        ),
        time=time_cols,
        states=["A", "B"],
        id_col="id",
    )
    ch2 = SequenceData(
        pd.DataFrame(
            [
                ["s1", "X", "X", "X", "X", "X"],
                ["s2", "X", "X", "X", "X", "X"],
                ["s3", "X", "X", "Y", "X", "X"],
                ["s4", "Y", "Y", "Y", "Y", "Y"],
            ],
            columns=["id", *time_cols],
        ),
        time=time_cols,
        states=["X", "Y"],
        id_col="id",
    )
    initial_probs, transition_probs, emission_probs = _multichannel_fixed_parameters()
    model = build_mnhmm(
        observations=[ch1, ch2],
        n_states=2,
        n_clusters=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
    )

    fitted = model.fit(n_iter=20, tol=1e-10)

    assert fitted.estimation_method == "fixed_component_cluster_em"
    assert fitted.cluster_probs[0] > 0.60
    assert fitted.cluster_probs[1] < 0.40
    np.testing.assert_allclose(fitted.cluster_probs.sum(), 1.0)
    assert fitted.responsibilities.shape == (4, 2)
    assert np.isfinite(fitted.log_likelihood)


def test_estimate_multichannel_mnhmm_fits_components_with_em():
    channels = _make_multichannel_seqdata()

    fitted = estimate_mnhmm(
        observations=channels,
        n_states=2,
        n_clusters=2,
        random_state=123,
        n_iter=3,
        tol=0.0,
    )

    assert fitted.estimation_method == "multichannel_em"
    assert fitted.n_iter == 3
    assert fitted.responsibilities.shape == (4, 2)
    np.testing.assert_allclose(fitted.responsibilities.sum(axis=1), np.ones(4))
    assert np.isfinite(fitted.log_likelihood)
    assert np.isclose(fitted.cluster_probs.sum(), 1.0)
    for cluster_idx in range(fitted.n_clusters):
        np.testing.assert_allclose(fitted.initial_probs[cluster_idx].sum(), 1.0)
        np.testing.assert_allclose(fitted.transition_probs[cluster_idx].sum(axis=1), np.ones(2))
        for emission in fitted.emission_probs[cluster_idx]:
            np.testing.assert_allclose(emission.sum(axis=1), np.ones(2))


def test_mnhmm_rejects_missing_sequences_to_preserve_time_alignment():
    seqdata = SequenceData(
        pd.DataFrame(
            [
                ["s1", "A", np.nan, "B"],
                ["s2", "B", "B", "B"],
            ],
            columns=["id", "t1", "t2", "t3"],
        ),
        time=["t1", "t2", "t3"],
        states=["A", "B"],
        id_col="id",
    )

    with pytest.raises(ValueError, match="missing values.*MNHMM"):
        build_mnhmm(
            observations=seqdata,
            n_states=2,
            n_clusters=2,
            cluster_probs=np.array([0.5, 0.5]),
        )


def test_mnhmm_compressed_new_data_rejects_missing_sequences():
    seqdata = _make_seqdata()
    initial_probs, transition_probs, emission_probs = _fixed_parameters()
    model = build_mnhmm(
        observations=seqdata,
        n_states=2,
        n_clusters=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        cluster_probs=np.array([0.5, 0.5]),
    )
    new_sequences = SequenceData(
        pd.DataFrame(
            [
                ["s1", "A", np.nan, "B", "B", "B"],
                ["s2", "B", "B", "B", "B", "B"],
                ["s3", "A", "A", "A", "A", "A"],
                ["s4", "B", "A", "B", "A", "B"],
            ],
            columns=["id", *TIME_COLS],
        ),
        time=TIME_COLS,
        states=STATES,
        id_col="id",
    )

    with pytest.raises(ValueError, match="missing values.*MNHMM"):
        model.score(sequences=new_sequences, compress=True)


def test_multichannel_mnhmm_compressed_new_data_rejects_missing_sequences():
    channels = _make_multichannel_seqdata()
    initial_probs, transition_probs, emission_probs = _multichannel_fixed_parameters()
    model = build_mnhmm(
        observations=channels,
        n_states=2,
        n_clusters=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        cluster_probs=np.array([0.55, 0.45]),
    )
    ids = [f"s{i}" for i in range(1, 5)]
    ch1 = pd.DataFrame(
        [
            ["A", np.nan, "A", "A", "A"],
            ["A", "A", "B", "A", "A"],
            ["B", "B", "B", "B", "B"],
            ["B", "B", "A", "B", "B"],
        ],
        columns=TIME_COLS,
    )
    ch1.insert(0, "id", ids)
    ch2 = pd.DataFrame(
        [
            ["X", "X", "X", "X", "X"],
            ["X", "X", "Y", "X", "X"],
            ["Y", "Y", "Y", "Y", "Y"],
            ["Y", "Y", "X", "Y", "Y"],
        ],
        columns=TIME_COLS,
    )
    ch2.insert(0, "id", ids)
    new_channels = [
        SequenceData(ch1, time=TIME_COLS, states=["A", "B"], id_col="id"),
        SequenceData(ch2, time=TIME_COLS, states=["X", "Y"], id_col="id"),
    ]

    with pytest.raises(ValueError, match="missing values.*MNHMM"):
        model.score(sequences=new_channels, compress=True)


def test_simulate_mnhmm_shape_and_reproducibility():
    initial_probs, transition_probs, emission_probs = _fixed_parameters()

    first = simulate_mnhmm(
        n_sequences=6,
        n_clusters=2,
        n_states=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        cluster_probs=np.array([0.6, 0.4]),
        sequence_length=4,
        alphabet=STATES,
        state_names=[["A1", "A2"], ["B1", "B2"]],
        cluster_names=["A-like", "B-like"],
        random_state=123,
    )
    second = simulate_mnhmm(
        n_sequences=6,
        n_clusters=2,
        n_states=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        cluster_probs=np.array([0.6, 0.4]),
        sequence_length=4,
        alphabet=STATES,
        state_names=[["A1", "A2"], ["B1", "B2"]],
        cluster_names=["A-like", "B-like"],
        random_state=123,
    )

    assert len(first["observations"]) == 6
    assert all(len(seq) == 4 for seq in first["observations"])
    assert len(first["states"]) == 6
    assert all(len(seq) == 4 for seq in first["states"])
    assert set(first["clusters"]).issubset({"A-like", "B-like"})
    assert set(first["observations_df"].columns) == {"time_1", "time_2", "time_3", "time_4", "cluster"}
    assert first["observations"] == second["observations"]
    assert first["states"] == second["states"]
    assert first["clusters"] == second["clusters"]
    pd.testing.assert_frame_equal(first["observations_df"], second["observations_df"])


def test_simulate_mnhmm_model_uses_sequence_specific_cluster_probabilities():
    seqdata = _make_seqdata()
    initial_probs, transition_probs, emission_probs = _fixed_parameters()
    model = build_mnhmm(
        observations=seqdata,
        n_states=2,
        n_clusters=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        X_cluster=np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        ),
        eta_omega=np.array([[100.0, -100.0], [-100.0, 100.0]]),
        cluster_names=["A-like", "B-like"],
    )

    simulated = simulate_mnhmm(model=model, random_state=321)

    assert simulated["clusters"] == ["A-like", "A-like", "B-like", "B-like"]


def test_simulate_mnhmm_model_supports_multichannel_fixed_probabilities():
    channels = _make_multichannel_seqdata()
    initial_probs, transition_probs, emission_probs = _multichannel_fixed_parameters()
    model = build_mnhmm(
        observations=channels,
        n_states=2,
        n_clusters=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        cluster_probs=np.array([0.5, 0.5]),
    )

    first = simulate_mnhmm(model=model, random_state=321)
    second = simulate_mnhmm(model=model, random_state=321)

    assert first["channel_names"] == ["Channel 1", "Channel 2"]
    assert set(first["observations"]) == {"Channel 1", "Channel 2"}
    assert len(first["observations"]["Channel 1"]) == model.n_sequences
    assert len(first["observations"]["Channel 2"]) == model.n_sequences
    assert all(len(seq) == 5 for seq in first["observations"]["Channel 1"])
    assert all(len(seq) == 5 for seq in first["observations"]["Channel 2"])
    assert set(first["observations"]["Channel 1"][0]).issubset({"A", "B"})
    assert set(first["observations"]["Channel 2"][0]).issubset({"X", "Y"})
    assert set(first["observations_df"]) == {"Channel 1", "Channel 2"}
    assert list(first["observations_df"]["Channel 1"].columns) == [
        "time_1",
        "time_2",
        "time_3",
        "time_4",
        "time_5",
        "cluster",
    ]
    assert first["observations"] == second["observations"]
    assert first["states"] == second["states"]
    assert first["clusters"] == second["clusters"]
    pd.testing.assert_frame_equal(
        first["observations_df"]["Channel 1"],
        second["observations_df"]["Channel 1"],
    )
    pd.testing.assert_frame_equal(
        first["observations_df"]["Channel 2"],
        second["observations_df"]["Channel 2"],
    )


def test_simulate_mnhmm_model_uses_multichannel_sequence_specific_cluster_probabilities():
    channels = _make_multichannel_seqdata()
    initial_probs, transition_probs, emission_probs = _multichannel_fixed_parameters()
    model = build_mnhmm(
        observations=channels,
        n_states=2,
        n_clusters=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        X_cluster=np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        ),
        eta_omega=np.array([[100.0, -100.0], [-100.0, 100.0]]),
        cluster_names=["A-like", "B-like"],
    )

    simulated = simulate_mnhmm(model=model, random_state=321)

    assert simulated["clusters"] == ["A-like", "A-like", "B-like", "B-like"]
    assert set(simulated["observations"]) == {"Channel 1", "Channel 2"}


def test_simulate_mnhmm_model_rejects_extra_multichannel_emission_blocks():
    channels = _make_multichannel_seqdata()
    initial_probs, transition_probs, emission_probs = _multichannel_fixed_parameters()
    model = build_mnhmm(
        observations=channels,
        n_states=2,
        n_clusters=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        cluster_probs=np.array([0.5, 0.5]),
    )
    model.emission_probs = [*model.emission_probs, model.emission_probs[0]]

    with pytest.raises(ValueError, match="emission_probs length"):
        simulate_mnhmm(model=model, random_state=321)


def test_estimate_mnhmm_intercept_only_em_fits_obvious_groups():
    fitted = estimate_mnhmm(
        observations=_make_seqdata(),
        n_states=2,
        n_clusters=2,
        random_state=11,
        n_iter=12,
        tol=0.0,
    )

    labels = fitted.predict_cluster()

    assert fitted.responsibilities.shape == (4, 2)
    np.testing.assert_allclose(fitted.responsibilities.sum(axis=1), np.ones(4))
    assert np.isfinite(fitted.log_likelihood)
    assert fitted.n_iter == 12
    assert np.isclose(fitted.cluster_probs.sum(), 1.0)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_estimate_mnhmm_verbose_reports_post_mstep_log_likelihood(capsys):
    fitted = estimate_mnhmm(
        observations=_make_seqdata(),
        n_states=2,
        n_clusters=2,
        random_state=11,
        n_iter=1,
        tol=0.0,
        verbose=True,
    )

    captured = capsys.readouterr().out
    assert f"log-likelihood = {fitted.log_likelihood:.4f}" in captured


def test_estimate_mnhmm_cluster_covariate_l_bfgs_learns_mixture_weights():
    seqdata = _make_seqdata()
    initial_probs, transition_probs, emission_probs = _fixed_parameters()
    X_cluster = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )

    fitted = estimate_mnhmm(
        observations=seqdata,
        n_states=2,
        n_clusters=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        X_cluster=X_cluster,
        cluster_names=["A-like", "B-like"],
        random_state=11,
        n_iter=80,
        tol=1e-8,
    )

    cluster_probs = fitted.compute_cluster_probs()

    assert fitted.estimation_method == "direct_l_bfgs"
    assert fitted.cluster_probs is None
    assert fitted.eta_omega.shape == (2, 2)
    assert fitted.optimization_result.nfev == fitted.optimization_result.njev
    assert cluster_probs[0, 0] > 0.80
    assert cluster_probs[1, 0] > 0.80
    assert cluster_probs[2, 1] > 0.80
    assert cluster_probs[3, 1] > 0.80
    assert np.isfinite(fitted.log_likelihood)


def test_estimate_mnhmm_can_use_probability_parameters_as_covariate_starts():
    seqdata = _make_seqdata()
    initial_probs, transition_probs, emission_probs = _fixed_parameters()

    fixed_model = build_mnhmm(
        observations=seqdata,
        n_states=2,
        n_clusters=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        cluster_probs=np.array([0.5, 0.5]),
    )

    fitted = estimate_mnhmm(
        observations=seqdata,
        n_states=2,
        n_clusters=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        cluster_probs=np.array([0.5, 0.5]),
        X_cluster=np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        ),
        probability_parameters_as_starts=True,
        n_iter=0,
        tol=0.0,
    )

    assert fitted.estimation_method == "direct_l_bfgs"
    assert fitted.initial_probs is None
    assert fitted.transition_probs is None
    assert fitted.emission_probs is None
    assert fitted.cluster_probs is None
    assert fitted.n_optimized_parameters == 12
    np.testing.assert_allclose(fitted.score(), fixed_model.score(), atol=1e-12)


def test_build_mnhmm_fit_routes_nonintercept_cluster_covariates_to_l_bfgs():
    seqdata = _make_seqdata()
    initial_probs, transition_probs, emission_probs = _fixed_parameters()
    model = build_mnhmm(
        observations=seqdata,
        n_states=2,
        n_clusters=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        X_cluster=np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        ),
        random_state=11,
    )

    fitted = model.fit(n_iter=80, tol=1e-8)
    cluster_probs = fitted.compute_cluster_probs()

    assert fitted.estimation_method == "direct_l_bfgs"
    assert fitted.cluster_probs is None
    assert cluster_probs[0, 0] > 0.80
    assert cluster_probs[1, 0] > 0.80
    assert cluster_probs[2, 1] > 0.80
    assert cluster_probs[3, 1] > 0.80


def test_estimate_mnhmm_covariate_component_path_runs_without_rejection():
    seqdata = _make_seqdata()
    X_B = np.ones((len(seqdata.sequences), len(TIME_COLS), 2))
    X_B[:, :, 1] = np.arange(len(TIME_COLS))

    fitted = estimate_mnhmm(
        observations=seqdata,
        n_states=2,
        n_clusters=2,
        X_B=X_B,
        random_state=11,
        n_iter=2,
        tol=0.0,
    )

    assert fitted.estimation_method == "direct_l_bfgs"
    assert fitted.eta_B[0].shape == (2, 2, 2)
    assert fitted.eta_B[1].shape == (2, 2, 2)
    assert fitted.eta_omega.shape == (1, 2)
    assert fitted.responsibilities.shape == (4, 2)
    np.testing.assert_allclose(fitted.responsibilities.sum(axis=1), np.ones(4))
    assert fitted.n_optimized_parameters == 15
    for eta in fitted.eta_pi:
        np.testing.assert_allclose(eta.sum(axis=1), np.zeros(eta.shape[0]), atol=1e-10)
    for eta in fitted.eta_A:
        np.testing.assert_allclose(
            eta.sum(axis=2), np.zeros((eta.shape[0], eta.shape[1])), atol=1e-10
        )
    for eta in fitted.eta_B:
        np.testing.assert_allclose(
            eta.sum(axis=2), np.zeros((eta.shape[0], eta.shape[1])), atol=1e-10
        )
    np.testing.assert_allclose(
        fitted.eta_omega.sum(axis=1), np.zeros(fitted.eta_omega.shape[0]), atol=1e-10
    )
    assert np.isfinite(fitted.log_likelihood)


def test_estimate_mnhmm_keeps_explicit_cluster_probs_fixed_in_direct_path():
    seqdata = _make_seqdata()
    X_B = np.ones((len(seqdata.sequences), len(TIME_COLS), 2))
    X_B[:, :, 1] = np.arange(len(TIME_COLS))

    fitted = estimate_mnhmm(
        observations=seqdata,
        n_states=2,
        n_clusters=2,
        X_B=X_B,
        cluster_probs=np.array([0.7, 0.3]),
        random_state=11,
        n_iter=2,
        tol=0.0,
    )

    assert fitted.estimation_method == "direct_l_bfgs"
    assert fitted.eta_omega is None
    np.testing.assert_allclose(fitted.cluster_probs, np.array([0.7, 0.3]))
