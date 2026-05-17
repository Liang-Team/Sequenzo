import os

import numpy as np
import pandas as pd
import pytest

from sequenzo import SequenceData
from sequenzo.seqhmm import (
    build_hmm,
    build_mhmm,
    build_nhmm,
    data_to_stslist,
    get_emission_probs,
    get_initial_probs,
    get_transition_probs,
    gridplot,
    hidden_paths,
    permute_states,
    separate_mhmm,
    ssplot,
    stacked_sequence_plot,
    stslist_to_data,
    trim_model,
)
from sequenzo.seqhmm.utilities import _state_alignment_cost


TIME_COLS = ["t1", "t2", "t3"]
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _seqdata():
    df = pd.DataFrame(
        [
            ["s1", "A", "A", "B"],
            ["s2", "B", "A", "B"],
        ],
        columns=["id", *TIME_COLS],
    )
    return SequenceData(df, time=TIME_COLS, states=["A", "B"], id_col="id")


def _hmm_3_state():
    return build_hmm(
        _seqdata(),
        initial_probs=np.array([0.2, 0.0, 0.8]),
        transition_probs=np.array(
            [
                [0.6, 0.0, 0.4],
                [0.2, 0.6, 0.2],
                [0.1, 0.0, 0.9],
            ]
        ),
        emission_probs=np.array(
            [
                [0.7, 0.3],
                [0.5, 0.5],
                [0.2, 0.8],
            ]
        ),
        state_names=["low", "middle", "high"],
    )


def _hmm_viterbi_config_a():
    time_cols = [f"t{i}" for i in range(1, 9)]
    df = pd.DataFrame(
        [
            ["s0", "A", "A", "B", "B", "C", "A", "A", "B"],
            ["s1", "B", "C", "C", "A", "A", "B", "C", "C"],
            ["s2", "A", "B", "C", "A", "B", "C", "A", "B"],
            ["s3", "C", "C", "B", "A", "A", "A", "B", "C"],
            ["s4", "A", "A", "A", "B", "B", "B", "C", "C"],
        ],
        columns=["id", *time_cols],
    )
    seq = SequenceData(df, time=time_cols, states=["A", "B", "C"], id_col="id")
    return build_hmm(
        seq,
        n_states=2,
        initial_probs=np.array([0.6, 0.4]),
        transition_probs=np.array([[0.7, 0.3], [0.2, 0.8]]),
        emission_probs=np.array([[0.5, 0.3, 0.2], [0.1, 0.4, 0.5]]),
    )


def _mhmm_fixture_model():
    time_cols = [f"t{i}" for i in range(1, 9)]
    raw = [
        ["s0", "A", "A", "B", "B", "C", "A", "A", "B"],
        ["s1", "B", "C", "C", "A", "A", "B", "C", "C"],
        ["s2", "A", "B", "C", "A", "B", "C", "A", "B"],
        ["s3", "C", "C", "B", "A", "A", "A", "B", "C"],
        ["s4", "A", "A", "A", "B", "B", "B", "C", "C"],
        ["s5", "C", "C", "B", "A", "A", "C", "B", "C"],
        ["s6", "C", "B", "C", "A", "B", "C", "A", "B"],
        ["s7", "B", "B", "C", "C", "B", "A", "B", "C"],
        ["s8", "A", "A", "B", "B", "C", "A", "A", "B"],
        ["s9", "C", "B", "A", "A", "B", "C", "A", "B"],
    ]
    seq = SequenceData(
        pd.DataFrame(raw, columns=["id", *time_cols]),
        time=time_cols,
        states=["A", "B", "C"],
        id_col="id",
    )
    return build_mhmm(
        seq,
        n_clusters=2,
        n_states=2,
        initial_probs=[np.array([0.7, 0.3]), np.array([0.4, 0.6])],
        transition_probs=[
            np.array([[0.8, 0.2], [0.3, 0.7]]),
            np.array([[0.6, 0.4], [0.2, 0.8]]),
        ],
        emission_probs=[
            np.array([[0.6, 0.3, 0.1], [0.1, 0.4, 0.5]]),
            np.array([[0.1, 0.3, 0.6], [0.5, 0.4, 0.1]]),
        ],
        cluster_names=["Cluster1", "Cluster2"],
    )


def test_probability_getters_return_defensive_copies():
    model = _hmm_3_state()

    initial = get_initial_probs(model)
    transition = get_transition_probs(model)
    emission = get_emission_probs(model)

    assert np.allclose(initial, model.initial_probs)
    assert np.allclose(transition, model.transition_probs)
    assert np.allclose(emission, model.emission_probs)

    initial[0] = 0.99
    transition[0, 0] = 0.99
    emission[0, 0] = 0.99

    assert not np.isclose(model.initial_probs[0], 0.99)
    assert not np.isclose(model.transition_probs[0, 0], 0.99)
    assert not np.isclose(model.emission_probs[0, 0], 0.99)


def test_probability_getters_support_nhmm_time_varying_shapes():
    seq = _seqdata()
    x = np.ones((len(seq.sequences), len(TIME_COLS), 1))
    model = build_nhmm(
        observations=seq,
        n_states=2,
        X=x,
        eta_pi=np.zeros((1, 2)),
        eta_A=np.zeros((1, 2, 2)),
        eta_B=np.zeros((1, 2, 2)),
    )

    assert get_initial_probs(model).shape == (2, 2)
    assert get_transition_probs(model).shape == (2, 3, 2, 2)
    assert get_emission_probs(model).shape == (2, 3, 2, 2)


def test_hidden_paths_can_return_predict_compatible_array():
    paths = hidden_paths(_hmm_3_state(), output="array")

    assert paths.shape == (2 * len(TIME_COLS),)
    assert set(paths).issubset({0, 1, 2})


def test_hidden_paths_returns_seqhmm_style_dataframe_matching_R_fixture():
    model = _hmm_viterbi_config_a()
    paths = hidden_paths(model)
    ref = pd.read_csv(os.path.join(THIS_DIR, "viterbi", "ref_viterbi_A.csv"))

    ref = ref.sort_values(["seq_idx", "time_idx"]).reset_index(drop=True)
    expected_states = [
        model.state_names[i] for i in ref["state_idx"].astype(int)
    ]

    assert list(paths.columns) == ["id", "time", "state"]
    assert paths["id"].tolist() == [f"s{i}" for i in ref["seq_idx"]]
    assert paths["time"].tolist() == (ref["time_idx"] + 1).tolist()
    assert paths["state"].tolist() == expected_states
    np.testing.assert_allclose(
        paths.attrs["log_prob"],
        ref.groupby("seq_idx")["viterbi_logprob"].first(),
    )


def test_hidden_paths_mhmm_matches_seqhmm_dataframe_fixture():
    paths = hidden_paths(_mhmm_fixture_model())
    ref = pd.read_csv(
        os.path.join(THIS_DIR, "mhmm", "ref_mhmm_hidden_paths.csv")
    )

    assert list(paths.columns) == ["id", "time", "state", "cluster"]
    assert (
        paths[["id", "time", "state", "cluster"]]
        .reset_index(drop=True)
        .equals(ref[["id", "time", "state", "cluster"]].reset_index(drop=True))
    )


def test_hidden_paths_supports_nhmm_dataframe_output():
    seq = _seqdata()
    x = np.ones((len(seq.sequences), len(TIME_COLS), 1))
    model = build_nhmm(
        observations=seq,
        n_states=2,
        X=x,
        eta_pi=np.zeros((1, 2)),
        eta_A=np.zeros((1, 2, 2)),
        eta_B=np.zeros((1, 2, 2)),
    )

    paths = hidden_paths(model)

    assert list(paths.columns) == ["id", "time", "state"]
    assert paths.shape[0] == len(seq.sequences) * len(TIME_COLS)
    assert set(paths["state"]).issubset(set(model.state_names))


def test_hidden_paths_as_stslist_round_trips_state_table():
    model = _hmm_viterbi_config_a()

    seq = hidden_paths(model, as_stslist=True)
    restored = stslist_to_data(seq, id="id", time="time", responses="state")

    assert restored.columns.tolist() == ["id", "time", "state"]
    assert restored.shape[0] == model.n_sequences * model.length_of_sequences
    assert set(restored["state"]).issubset(set(model.state_names))


def test_permute_states_reorders_hmm_parameters_without_mutating_original():
    model = _hmm_3_state()
    permuted = permute_states(model, [2, 0, 1])

    assert permuted is not model
    assert permuted.state_names == ["high", "low", "middle"]
    assert np.allclose(permuted.initial_probs, [0.8, 0.2, 0.0])
    assert np.allclose(
        permuted.transition_probs,
        model.transition_probs[np.ix_([2, 0, 1], [2, 0, 1])],
    )
    assert np.allclose(
        permuted.emission_probs, model.emission_probs[[2, 0, 1]]
    )
    assert model.state_names == ["low", "middle", "high"]


def test_permute_states_matches_seqhmm_gamma_alignment_semantics():
    reference = {
        "gamma_pi": np.array([[0.0], [10.0], [20.0]]),
        "gamma_A": np.arange(27, dtype=float).reshape(3, 3, 3),
        "gamma_B": [np.arange(24, dtype=float).reshape(2, 4, 3)],
    }
    order = [2, 0, 1]
    estimates = {
        "gamma_pi": reference["gamma_pi"][order],
        "gamma_A": reference["gamma_A"][np.ix_(order, range(3), order)],
        "gamma_B": [reference["gamma_B"][0][:, :, order]],
    }

    aligned = permute_states(estimates, reference)

    np.testing.assert_allclose(aligned["gamma_pi"], reference["gamma_pi"])
    np.testing.assert_allclose(aligned["gamma_A"], reference["gamma_A"])
    np.testing.assert_allclose(aligned["gamma_B"][0], reference["gamma_B"][0])
    assert aligned.permutation == [1, 2, 0]


def test_state_alignment_cost_matches_seqhmm_cpp_slice_semantics():
    estimates = {
        "gamma_pi": np.array([[0.0, 0.1], [1.0, 1.1]]),
        "gamma_A": np.arange(12, dtype=float).reshape(2, 3, 2),
        "gamma_B": [np.arange(16, dtype=float).reshape(4, 2, 2)],
    }
    reference = {
        "gamma_pi": np.array([[0.2, 0.3], [0.9, 1.2]]),
        "gamma_A": np.arange(12, dtype=float).reshape(2, 3, 2) + 0.5,
        "gamma_B": [np.arange(16, dtype=float).reshape(4, 2, 2) + 0.25],
    }

    cost = _state_alignment_cost(estimates, reference)

    expected = np.empty((2, 2))
    for k in range(2):
        for j in range(2):
            est_vec = np.concatenate(
                [
                    estimates["gamma_pi"][k].ravel(order="F"),
                    estimates["gamma_A"][:, :, k].ravel(order="F"),
                    estimates["gamma_B"][0][:, :, k].ravel(order="F"),
                ]
            )
            ref_vec = np.concatenate(
                [
                    reference["gamma_pi"][j].ravel(order="F"),
                    reference["gamma_A"][:, :, j].ravel(order="F"),
                    reference["gamma_B"][0][:, :, j].ravel(order="F"),
                ]
            )
            expected[k, j] = np.mean((est_vec - ref_vec) ** 2)
    np.testing.assert_allclose(cost, expected)


def test_permute_states_rejects_non_hmm_explicit_permutation():
    mhmm = _mhmm_fixture_model()

    with pytest.raises(TypeError, match="HMM objects only"):
        permute_states(mhmm, [1, 0])


def test_trim_model_zeroes_small_probabilities_without_dropping_states():
    seq = SequenceData(
        pd.DataFrame(
            [
                ["s1", "A", "A", "B"],
                ["s2", "B", "A", "B"],
            ],
            columns=["id", *TIME_COLS],
        ),
        time=TIME_COLS,
        states=["A", "B", "C"],
        id_col="id",
    )
    model = build_hmm(
        seq,
        initial_probs=np.array([0.8, 0.2]),
        transition_probs=np.array([[0.9, 0.1], [0.2, 0.8]]),
        emission_probs=np.array(
            [[0.5999999999, 0.4, 1e-10], [0.3, 0.7 - 1e-10, 1e-10]]
        ),
        state_names=["low", "high"],
    )
    ll_before = model.score()

    trimmed = trim_model(model, zerotol=1e-8)

    assert trimmed.n_states == 2
    assert trimmed.state_names == ["low", "high"]
    assert np.count_nonzero(trimmed.emission_probs == 0.0) == 2
    np.testing.assert_allclose(trimmed.emission_probs.sum(axis=1), 1.0)
    assert trimmed.score() >= ll_before - 1e-8


def test_trim_model_return_loglik_and_copy_contract():
    model = _hmm_3_state()
    original_emissions = model.emission_probs.copy()

    result = trim_model(model, zerotol=0.31, return_loglik=True)

    assert set(result) == {"model", "loglik"}
    assert result["model"] is not model
    np.testing.assert_allclose(model.emission_probs, original_emissions)
    assert np.isfinite(result["loglik"])


def test_trim_model_processes_every_mhmm_cluster():
    seq = _seqdata()
    mhmm = build_mhmm(
        seq,
        n_clusters=2,
        n_states=2,
        initial_probs=[np.array([0.9999999999, 1e-10]), np.array([0.8, 0.2])],
        transition_probs=[
            np.array([[0.9999999999, 1e-10], [0.2, 0.8]]),
            np.array([[0.7, 0.3], [1e-10, 0.9999999999]]),
        ],
        emission_probs=[
            np.array([[0.6, 0.4], [1e-10, 0.9999999999]]),
            np.array([[0.9999999999, 1e-10], [0.3, 0.7]]),
        ],
    )

    trimmed = trim_model(mhmm, zerotol=1e-8)

    assert trimmed.clusters[0].transition_probs[0, 1] == 0.0
    assert trimmed.clusters[0].emission_probs[1, 0] == 0.0
    assert trimmed.clusters[1].transition_probs[1, 0] == 0.0
    assert trimmed.clusters[1].emission_probs[0, 1] == 0.0
    for cluster in trimmed.clusters:
        np.testing.assert_allclose(cluster.initial_probs.sum(), 1.0)
        np.testing.assert_allclose(cluster.transition_probs.sum(axis=1), 1.0)
        np.testing.assert_allclose(cluster.emission_probs.sum(axis=1), 1.0)


def test_separate_mhmm_returns_independent_cluster_hmms():
    seq = _seqdata()
    mhmm = build_mhmm(
        seq,
        n_clusters=2,
        n_states=2,
        initial_probs=[np.array([0.7, 0.3]), np.array([0.2, 0.8])],
        transition_probs=[
            np.array([[0.8, 0.2], [0.1, 0.9]]),
            np.array([[0.5, 0.5], [0.4, 0.6]]),
        ],
        emission_probs=[
            np.array([[0.9, 0.1], [0.2, 0.8]]),
            np.array([[0.3, 0.7], [0.6, 0.4]]),
        ],
    )

    clusters = separate_mhmm(mhmm)

    assert len(clusters) == 2
    assert np.allclose(get_initial_probs(clusters[1]), [0.2, 0.8])
    clusters[0].initial_probs[0] = 0.01
    assert not np.isclose(mhmm.clusters[0].initial_probs[0], 0.01)


def test_stslist_conversion_round_trips_wide_sequence_data():
    source = pd.DataFrame(
        [
            ["s1", "A", "A", "B"],
            ["s2", "B", "A", "B"],
        ],
        columns=["id", *TIME_COLS],
    )

    seq = data_to_stslist(
        source, time=TIME_COLS, states=["A", "B"], id_col="id"
    )
    restored = stslist_to_data(seq, id_col="id", wide=True)

    assert list(restored.columns) == ["id", *TIME_COLS]
    assert restored.equals(source)


def test_long_format_data_to_stslist_and_back_matches_seqhmm_shape():
    long_data = pd.DataFrame(
        {
            "person": ["s1", "s1", "s1", "s2", "s2", "s2"],
            "year": [1, 2, 3, 1, 2, 3],
            "work": ["A", "A", "B", "B", "A", "B"],
            "family": ["X", "Y", "Y", "Y", "X", "Y"],
        }
    )

    seq = data_to_stslist(
        long_data, id="person", time="year", responses="work"
    )
    restored = stslist_to_data(seq, id="person", time="year", responses="work")

    assert isinstance(seq, SequenceData)
    assert restored.equals(long_data[["person", "year", "work"]])

    multi = data_to_stslist(
        long_data,
        id="person",
        time="year",
        responses=["work", "family"],
    )

    assert set(multi) == {"work", "family"}
    multi_restored = stslist_to_data(
        multi,
        id="person",
        time="year",
        responses=["work", "family"],
    )
    assert multi_restored.equals(long_data)


def test_long_format_data_to_stslist_preserves_first_seen_id_order():
    long_data = pd.DataFrame(
        {
            "person": ["s2", "s2", "s1", "s1"],
            "year": [1, 2, 1, 2],
            "work": ["B", "A", "A", "B"],
        }
    )

    seq = data_to_stslist(
        long_data, id="person", time="year", responses="work"
    )
    restored = stslist_to_data(seq, id="person", time="year", responses="work")

    assert restored["person"].tolist() == ["s2", "s2", "s1", "s1"]
    assert restored["work"].tolist() == ["B", "A", "A", "B"]


def test_long_format_data_to_stslist_uses_natural_time_order():
    long_data = pd.DataFrame(
        {
            "person": ["s1", "s1", "s1"],
            "wave": ["t1", "t2", "t10"],
            "work": ["A", "B", "C"],
        }
    )

    seq = data_to_stslist(
        long_data, id="person", time="wave", responses="work"
    )
    restored = stslist_to_data(seq, id="person", time="wave", responses="work")

    assert list(seq.time) == ["t1", "t2", "t10"]
    assert restored["wave"].tolist() == ["t1", "t2", "t10"]
    assert restored["work"].tolist() == ["A", "B", "C"]


def test_stslist_to_data_long_output_preserves_order_after_vectorization():
    n_sequences = 50
    time_cols = [f"t{i}" for i in range(1, 6)]
    rows = [
        [
            f"s{i}",
            *["A" if (i + t) % 2 == 0 else "B" for t in range(len(time_cols))],
        ]
        for i in range(n_sequences)
    ]
    seq = SequenceData(
        pd.DataFrame(rows, columns=["id", *time_cols]),
        time=time_cols,
        states=["A", "B"],
        id_col="id",
    )

    restored = stslist_to_data(seq, id="id", time="time", responses="state")

    assert restored["id"].tolist() == [
        f"s{i}" for i in range(n_sequences) for _ in time_cols
    ]
    assert restored["time"].tolist() == time_cols * n_sequences
    assert restored["state"].tolist()[:5] == rows[0][1:]


def test_utility_getters_and_hidden_paths_reject_unsupported_objects():
    class Dummy:
        pass

    dummy = Dummy()

    with pytest.raises(TypeError, match="HMM, MHMM, or NHMM"):
        hidden_paths(dummy)
    with pytest.raises(TypeError, match="HMM, MHMM, or NHMM"):
        get_initial_probs(dummy)
    with pytest.raises(TypeError, match="HMM, MHMM, or NHMM"):
        get_transition_probs(dummy)
    with pytest.raises(TypeError, match="HMM, MHMM, or NHMM"):
        get_emission_probs(dummy)
    with pytest.raises(TypeError, match="HMM or MHMM"):
        trim_model(dummy)


def test_seqhmm_plot_compatibility_wrappers_return_figures():
    import matplotlib.pyplot as plt

    seq = _seqdata()

    fig1 = stacked_sequence_plot(seq, include_legend=False)
    fig2 = ssplot(seq, include_legend=False)
    fig3 = gridplot(seq, plots=("stacked", "distribution"))

    assert hasattr(fig1, "savefig")
    assert hasattr(fig2, "savefig")
    assert hasattr(fig3, "savefig")
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
