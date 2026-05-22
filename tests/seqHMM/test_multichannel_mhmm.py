import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from sequenzo import SequenceData
from sequenzo.seqhmm import build_mhmm, posterior_probs_mhmm, predict_mhmm
from sequenzo.seqhmm.utils import sequence_data_to_hmmlearn_format


TIME_COLS = ["t1", "t2", "t3", "t4"]
REF_DIR = "tests/seqhmm_parity/_refs/mhmm_multichannel_fixed"


def _seqdata(rows, states, ids=None):
    df = pd.DataFrame(rows, columns=TIME_COLS)
    df.insert(0, "id", ids or [f"s{i}" for i in range(len(df))])
    return SequenceData(df, time=TIME_COLS, states=states, id_col="id")


def _multichannel_mhmm():
    ch1 = _seqdata(
        [
            ["A", "A", "A", "A"],
            ["A", "A", "B", "A"],
            ["B", "B", "B", "B"],
            ["B", "B", "A", "B"],
        ],
        ["A", "B"],
    )
    ch2 = _seqdata(
        [
            ["X", "X", "X", "X"],
            ["X", "X", "Y", "X"],
            ["Y", "Y", "Y", "Y"],
            ["Y", "Y", "X", "Y"],
        ],
        ["X", "Y"],
    )

    initial_probs = [
        np.array([0.95, 0.05]),
        np.array([0.95, 0.05]),
    ]
    transition_probs = [
        np.array([[0.90, 0.10], [0.10, 0.90]]),
        np.array([[0.90, 0.10], [0.10, 0.90]]),
    ]
    emission_probs = [
        [
            np.array([[0.95, 0.05], [0.10, 0.90]]),
            np.array([[0.88, 0.12], [0.25, 0.75]]),
        ],
        [
            np.array([[0.05, 0.95], [0.90, 0.10]]),
            np.array([[0.12, 0.88], [0.80, 0.20]]),
        ],
    ]

    return build_mhmm(
        [ch1, ch2],
        n_clusters=2,
        n_states=2,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        cluster_probs=np.array([0.5, 0.5]),
        cluster_names=["mostly-AX", "mostly-BY"],
        channel_names=["letter", "marker"],
    )


def test_multichannel_fixed_parameter_mhmm_predicts_without_fit():
    model = _multichannel_mhmm()

    clusters = predict_mhmm(model)

    assert clusters.shape == (4,)
    assert clusters.tolist() == [0, 0, 1, 1]
    np.testing.assert_array_equal(model.predict_cluster(), clusters)


def test_multichannel_fixed_parameter_mhmm_score_matches_manual_likelihood():
    model = _multichannel_mhmm()

    log_likelihoods = model._sequence_log_likelihoods()
    expected = np.sum(
        logsumexp(
            log_likelihoods + np.log(model.cluster_probs)[np.newaxis, :],
            axis=1,
        )
    )

    np.testing.assert_allclose(model.score(), expected, atol=1e-12)


def test_multichannel_fixed_parameter_mhmm_loglik_matches_seqhmm_golden():
    model = _multichannel_mhmm()
    ref = pd.read_csv(f"{REF_DIR}/ref_mhmm_multichannel_loglik.csv")
    expected = float(ref.loc[ref["key"] == "loglik", "value"].iloc[0])

    np.testing.assert_allclose(model.score(), expected, atol=1e-8)


def test_multichannel_fixed_parameter_mhmm_posteriors_sum_to_one_without_fit():
    model = _multichannel_mhmm()

    posterior = posterior_probs_mhmm(model, compress=True)

    assert list(posterior.columns) == ["id", "cluster", "probability"]
    assert len(posterior) == 4 * model.n_clusters
    assert posterior["id"].drop_duplicates().tolist() == ["s0", "s1", "s2", "s3"]

    summed = posterior.groupby("id")["probability"].sum()
    assert np.allclose(summed.to_numpy(), 1.0)

    winners = posterior.loc[
        posterior.groupby("id")["probability"].idxmax()
    ].sort_values("id")
    assert winners["cluster"].tolist() == ["mostly-AX", "mostly-AX", "mostly-BY", "mostly-BY"]


def test_multichannel_fixed_parameter_mhmm_posteriors_match_seqhmm_golden():
    model = _multichannel_mhmm()
    observed = posterior_probs_mhmm(model).sort_values(["id", "cluster"]).reset_index(drop=True)
    reference = pd.read_csv(
        f"{REF_DIR}/ref_mhmm_multichannel_posterior_cluster.csv"
    ).sort_values(["id", "cluster"]).reset_index(drop=True)

    assert observed[["id", "cluster"]].equals(reference[["id", "cluster"]])
    np.testing.assert_allclose(
        observed["probability"].to_numpy(),
        reference["probability"].to_numpy(),
        atol=1e-8,
    )


def test_multichannel_fixed_parameter_mhmm_predict_matches_seqhmm_posterior_golden():
    model = _multichannel_mhmm()
    reference = pd.read_csv(f"{REF_DIR}/ref_mhmm_multichannel_cluster.csv")
    observed = np.asarray(model.cluster_names)[predict_mhmm(model)]

    assert observed.tolist() == reference["cluster"].tolist()


def test_multichannel_mhmm_compressed_score_matches_direct_with_repeated_sequences():
    ch1 = _seqdata(
        [
            ["A", "A", "A", "A"],
            ["A", "A", "A", "A"],
            ["B", "B", "B", "B"],
            ["B", "B", "B", "B"],
            ["A", "A", "B", "A"],
            ["A", "A", "B", "A"],
        ],
        ["A", "B"],
    )
    ch2 = _seqdata(
        [
            ["X", "X", "X", "X"],
            ["X", "X", "X", "X"],
            ["Y", "Y", "Y", "Y"],
            ["Y", "Y", "Y", "Y"],
            ["X", "X", "Y", "X"],
            ["X", "X", "Y", "X"],
        ],
        ["X", "Y"],
    )
    base = _multichannel_mhmm()
    model = build_mhmm(
        [ch1, ch2],
        n_clusters=2,
        n_states=2,
        initial_probs=[c.initial_probs.copy() for c in base.clusters],
        transition_probs=[c.transition_probs.copy() for c in base.clusters],
        emission_probs=[
            [em.copy() for em in c.emission_probs]
            for c in base.clusters
        ],
        cluster_probs=base.cluster_probs.copy(),
    )

    np.testing.assert_allclose(
        model.score(compress=True),
        model.score(compress=False),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        model.compute_responsibilities(compress=True),
        model.compute_responsibilities(compress=False),
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        predict_mhmm(model, compress=True),
        predict_mhmm(model, compress=False),
    )
    np.testing.assert_array_equal(
        model.predict_cluster(compress=True),
        model.predict_cluster(compress=False),
    )


def test_multichannel_mhmm_compressed_fit_matches_direct_for_repeated_sequences():
    ch1 = _seqdata(
        [
            ["A", "A", "A", "A"],
            ["A", "A", "A", "A"],
            ["B", "B", "B", "B"],
            ["B", "B", "B", "B"],
        ],
        ["A", "B"],
    )
    ch2 = _seqdata(
        [
            ["X", "X", "X", "X"],
            ["X", "X", "X", "X"],
            ["Y", "Y", "Y", "Y"],
            ["Y", "Y", "Y", "Y"],
        ],
        ["X", "Y"],
    )
    base = _multichannel_mhmm()
    kwargs = dict(
        observations=[ch1, ch2],
        n_clusters=2,
        n_states=2,
        initial_probs=[c.initial_probs.copy() for c in base.clusters],
        transition_probs=[c.transition_probs.copy() for c in base.clusters],
        emission_probs=[
            [em.copy() for em in c.emission_probs]
            for c in base.clusters
        ],
        cluster_probs=base.cluster_probs.copy(),
    )

    direct = build_mhmm(**kwargs).fit(n_iter=2, tol=0.0, compress=False)
    compressed = build_mhmm(**kwargs).fit(n_iter=2, tol=0.0, compress=True)

    np.testing.assert_allclose(compressed.log_likelihood, direct.log_likelihood, atol=1e-10)
    np.testing.assert_allclose(compressed.cluster_probs, direct.cluster_probs, atol=1e-10)
    np.testing.assert_allclose(compressed.responsibilities, direct.responsibilities, atol=1e-10)
    np.testing.assert_array_equal(compressed.predict_cluster(), direct.predict_cluster())
    for direct_cluster, compressed_cluster in zip(direct.clusters, compressed.clusters):
        np.testing.assert_allclose(
            compressed_cluster.initial_probs,
            direct_cluster.initial_probs,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            compressed_cluster.transition_probs,
            direct_cluster.transition_probs,
            atol=1e-10,
        )
        for direct_emission, compressed_emission in zip(
            direct_cluster.emission_probs,
            compressed_cluster.emission_probs,
        ):
            np.testing.assert_allclose(compressed_emission, direct_emission, atol=1e-10)


def test_multichannel_mhmm_auto_compression_matches_direct_fit():
    ch1 = _seqdata(
        [
            ["A", "A", "A", "A"],
            ["A", "A", "A", "A"],
            ["B", "B", "B", "B"],
            ["B", "B", "B", "B"],
        ],
        ["A", "B"],
    )
    ch2 = _seqdata(
        [
            ["X", "X", "X", "X"],
            ["X", "X", "X", "X"],
            ["Y", "Y", "Y", "Y"],
            ["Y", "Y", "Y", "Y"],
        ],
        ["X", "Y"],
    )
    base = _multichannel_mhmm()
    kwargs = dict(
        observations=[ch1, ch2],
        n_clusters=2,
        n_states=2,
        initial_probs=[c.initial_probs.copy() for c in base.clusters],
        transition_probs=[c.transition_probs.copy() for c in base.clusters],
        emission_probs=[
            [em.copy() for em in c.emission_probs]
            for c in base.clusters
        ],
        cluster_probs=base.cluster_probs.copy(),
    )

    direct = build_mhmm(**kwargs).fit(n_iter=2, tol=0.0, compress=False)
    auto = build_mhmm(**kwargs).fit(n_iter=2, tol=0.0)

    np.testing.assert_allclose(auto.log_likelihood, direct.log_likelihood, atol=1e-10)
    np.testing.assert_allclose(auto.cluster_probs, direct.cluster_probs, atol=1e-10)


def test_multichannel_mhmm_fit_updates_responsibilities_and_normalized_parameters():
    model = _multichannel_mhmm()

    fitted = model.fit(n_iter=3, tol=0.0)

    assert fitted.responsibilities.shape == (4, 2)
    assert np.allclose(fitted.responsibilities.sum(axis=1), 1.0)
    assert np.isfinite(fitted.log_likelihood)
    assert fitted.n_iter == 3
    assert np.isclose(fitted.cluster_probs.sum(), 1.0)
    for cluster in fitted.clusters:
        assert np.isclose(cluster.initial_probs.sum(), 1.0)
        assert np.allclose(cluster.transition_probs.sum(axis=1), 1.0)
        for emission in cluster.emission_probs:
            assert np.allclose(emission.sum(axis=1), 1.0)


def test_multichannel_mhmm_verbose_reports_post_mstep_log_likelihood(capsys):
    model = _multichannel_mhmm()

    fitted = model.fit(n_iter=1, tol=0.0, verbose=True)

    captured = capsys.readouterr().out
    assert f"log-likelihood = {fitted.log_likelihood:.4f}" in captured


def test_multichannel_mhmm_default_fit_breaks_symmetry_on_obvious_groups():
    ch1 = _seqdata(
        [
            ["A", "A", "A", "A"],
            ["A", "A", "A", "B"],
            ["A", "A", "B", "A"],
            ["A", "A", "A", "A"],
            ["B", "B", "B", "B"],
            ["B", "B", "B", "A"],
            ["B", "B", "A", "B"],
            ["B", "B", "B", "B"],
        ],
        ["A", "B"],
    )
    ch2 = _seqdata(
        [
            ["X", "X", "X", "X"],
            ["X", "X", "X", "Y"],
            ["X", "X", "Y", "X"],
            ["X", "X", "X", "X"],
            ["Y", "Y", "Y", "Y"],
            ["Y", "Y", "Y", "X"],
            ["Y", "Y", "X", "Y"],
            ["Y", "Y", "Y", "Y"],
        ],
        ["X", "Y"],
    )

    model = build_mhmm(
        [ch1, ch2],
        n_clusters=2,
        n_states=2,
        random_state=7,
    )

    fitted = model.fit(n_iter=8, tol=0.0)
    labels = fitted.predict_cluster()

    assert len(set(labels[:4])) == 1
    assert len(set(labels[4:])) == 1
    assert labels[0] != labels[4]
    assert np.max(np.abs(fitted.responsibilities - 0.5)) > 0.2
    assert not np.allclose(
        fitted.clusters[0].emission_probs[0],
        fitted.clusters[1].emission_probs[0],
    )


def test_multichannel_mhmm_rejects_shuffled_channel_ids():
    ch1 = _seqdata(
        [["A", "A", "A", "A"], ["B", "B", "B", "B"]],
        ["A", "B"],
        ids=["s1", "s2"],
    )
    ch2 = _seqdata(
        [["Y", "Y", "Y", "Y"], ["X", "X", "X", "X"]],
        ["X", "Y"],
        ids=["s2", "s1"],
    )

    with pytest.raises(ValueError, match="same sequence IDs in the same order"):
        build_mhmm([ch1, ch2], n_clusters=2, n_states=2, random_state=1)


def test_multichannel_mhmm_rejects_newdata_alphabet_mismatch():
    model = _multichannel_mhmm()
    ch1 = _seqdata(
        [["A", "A", "A", "A"], ["B", "B", "B", "B"]],
        ["B", "A"],
    )
    ch2 = _seqdata(
        [["X", "X", "X", "X"], ["Y", "Y", "Y", "Y"]],
        ["X", "Y"],
    )

    with pytest.raises(ValueError, match="alphabet"):
        model.score([ch1, ch2])


def test_multichannel_mhmm_rejects_invalid_fixed_probabilities():
    ch1 = _seqdata([["A", "A", "A", "A"], ["B", "B", "B", "B"]], ["A", "B"])
    ch2 = _seqdata([["X", "X", "X", "X"], ["Y", "Y", "Y", "Y"]], ["X", "Y"])

    with pytest.raises(ValueError, match="initial_probs"):
        build_mhmm(
            [ch1, ch2],
            n_clusters=2,
            n_states=2,
            initial_probs=[np.array([1.2, -0.2]), np.array([0.5, 0.5])],
            transition_probs=[
                np.array([[0.9, 0.1], [0.1, 0.9]]),
                np.array([[0.9, 0.1], [0.1, 0.9]]),
            ],
            emission_probs=[
                [np.eye(2), np.eye(2)],
                [np.eye(2), np.eye(2)],
            ],
        )
    with pytest.raises(ValueError, match="sum to one"):
        build_mhmm(
            [ch1, ch2],
            n_clusters=2,
            n_states=2,
            initial_probs=[np.array([9.0, 1.0]), np.array([0.5, 0.5])],
            transition_probs=[
                np.array([[0.9, 0.1], [0.1, 0.9]]),
                np.array([[0.9, 0.1], [0.1, 0.9]]),
            ],
            emission_probs=[
                [np.eye(2), np.eye(2)],
                [np.eye(2), np.eye(2)],
            ],
        )


def test_multichannel_mhmm_rejects_invalid_cluster_probabilities():
    ch1 = _seqdata([["A", "A", "A", "A"], ["B", "B", "B", "B"]], ["A", "B"])
    ch2 = _seqdata([["X", "X", "X", "X"], ["Y", "Y", "Y", "Y"]], ["X", "Y"])

    with pytest.raises(ValueError, match="cluster_probs"):
        build_mhmm(
            [ch1, ch2],
            n_clusters=2,
            n_states=2,
            cluster_probs=np.array([1.2, -0.2]),
        )
    with pytest.raises(ValueError, match="sum to one"):
        build_mhmm(
            [ch1, ch2],
            n_clusters=2,
            n_states=2,
            cluster_probs=np.array([2.0, 1.0]),
        )


def test_multichannel_mhmm_rejects_invalid_cluster_names_length():
    ch1 = _seqdata([["A", "A", "A", "A"], ["B", "B", "B", "B"]], ["A", "B"])
    ch2 = _seqdata([["X", "X", "X", "X"], ["Y", "Y", "Y", "Y"]], ["X", "Y"])

    with pytest.raises(ValueError, match="cluster_names"):
        build_mhmm(
            [ch1, ch2],
            n_clusters=2,
            n_states=2,
            cluster_names=["only-one"],
        )


def test_single_channel_mhmm_rejects_newdata_alphabet_mismatch():
    seq = _seqdata(
        [["A", "A", "A", "A"], ["B", "B", "B", "B"]],
        ["A", "B"],
    )
    model = build_mhmm(
        seq,
        n_clusters=2,
        n_states=2,
        initial_probs=[np.array([0.9, 0.1]), np.array([0.1, 0.9])],
        transition_probs=[np.eye(2), np.eye(2)],
        emission_probs=[
            np.array([[0.9, 0.1], [0.2, 0.8]]),
            np.array([[0.2, 0.8], [0.9, 0.1]]),
        ],
        cluster_probs=np.array([0.5, 0.5]),
    )
    newdata = _seqdata(
        [["A", "A", "A", "A"], ["B", "B", "B", "B"]],
        ["B", "A"],
    )

    with pytest.raises(ValueError, match="alphabet"):
        model.score(newdata)


def test_multichannel_mhmm_impossible_observation_has_negative_infinite_score():
    ch1 = _seqdata([["A", "A", "A", "A"]], ["A", "B"])
    ch2 = _seqdata([["X", "X", "X", "X"]], ["X", "Y"])
    model = build_mhmm(
        [ch1, ch2],
        n_clusters=2,
        n_states=2,
        initial_probs=[np.array([1.0, 0.0]), np.array([0.0, 1.0])],
        transition_probs=[np.eye(2), np.eye(2)],
        emission_probs=[
            [
                np.array([[0.0, 1.0], [0.0, 1.0]]),
                np.array([[1.0, 0.0], [1.0, 0.0]]),
            ],
            [
                np.array([[0.0, 1.0], [0.0, 1.0]]),
                np.array([[1.0, 0.0], [1.0, 0.0]]),
            ],
        ],
        cluster_probs=np.array([0.5, 0.5]),
    )

    assert model.score() == -np.inf
    with pytest.raises(ValueError, match="zero likelihood"):
        model.compute_responsibilities()
    with pytest.raises(ValueError, match="zero likelihood"):
        predict_mhmm(model)


def test_multichannel_mhmm_zero_cluster_prior_remains_structural_zero():
    model = _multichannel_mhmm()
    model.cluster_probs = np.array([0.0, 1.0])

    responsibilities = model.compute_responsibilities()

    np.testing.assert_allclose(responsibilities[:, 0], 0.0, atol=0.0)
    np.testing.assert_allclose(responsibilities[:, 1], 1.0, atol=0.0)
    expected = np.sum(model._sequence_log_likelihoods()[:, 1])
    np.testing.assert_allclose(model.score(), expected, atol=1e-12)


def test_multichannel_mhmm_responsibilities_use_current_cluster_prior_after_fit():
    model = _multichannel_mhmm().fit(n_iter=1, tol=0.0)
    model.cluster_probs = np.array([0.0, 1.0])

    responsibilities = model.compute_responsibilities()

    np.testing.assert_allclose(responsibilities[:, 0], 0.0, atol=0.0)
    np.testing.assert_allclose(responsibilities[:, 1], 1.0, atol=0.0)


def test_single_channel_mhmm_fit_records_final_parameter_log_likelihood():
    seq = _seqdata(
        [
            ["A", "A", "B", "A"],
            ["A", "B", "B", "A"],
            ["B", "B", "A", "B"],
            ["B", "A", "A", "B"],
        ],
        ["A", "B"],
    )
    model = build_mhmm(
        seq,
        n_clusters=2,
        n_states=2,
        initial_probs=[np.array([0.8, 0.2]), np.array([0.3, 0.7])],
        transition_probs=[
            np.array([[0.9, 0.1], [0.2, 0.8]]),
            np.array([[0.7, 0.3], [0.1, 0.9]]),
        ],
        emission_probs=[
            np.array([[0.9, 0.1], [0.2, 0.8]]),
            np.array([[0.2, 0.8], [0.8, 0.2]]),
        ],
        cluster_probs=np.array([0.55, 0.45]),
    )

    fitted = model.fit(n_iter=1, tol=0.0)
    X, lengths = sequence_data_to_hmmlearn_format(seq)
    log_likelihoods = np.zeros((len(lengths), fitted.n_clusters))
    starts = np.zeros(len(lengths) + 1, dtype=int)
    starts[1:] = np.cumsum(lengths)
    for cluster_idx, cluster in enumerate(fitted.clusters):
        for seq_idx in range(len(lengths)):
            seq_X = X[starts[seq_idx]:starts[seq_idx + 1]]
            seq_len = np.array([lengths[seq_idx]])
            log_likelihoods[seq_idx, cluster_idx] = cluster._hmm_model.score(seq_X, seq_len)
    expected = np.sum(
        logsumexp(
            log_likelihoods + np.log(fitted.cluster_probs)[np.newaxis, :],
            axis=1,
        )
    )

    np.testing.assert_allclose(fitted.log_likelihood, expected, atol=1e-10)
