import numpy as np
import pandas as pd
import pytest

from sequenzo import SequenceData
from sequenzo.seqhmm import build_hmm, posterior_probs, predict


TIME_COLS = ["t1", "t2", "t3", "t4"]


def _seqdata(rows, states, ids=None):
    df = pd.DataFrame(rows, columns=TIME_COLS)
    df.insert(0, "id", ids or [f"s{i}" for i in range(len(df))])
    return SequenceData(df, time=TIME_COLS, states=states, id_col="id")


def _multichannel_model():
    ch1 = _seqdata(
        [
            ["A", "A", "B", "B"],
            ["A", "B", "B", "A"],
        ],
        ["A", "B"],
    )
    ch2 = _seqdata(
        [
            ["X", "X", "Y", "Y"],
            ["X", "Y", "Y", "X"],
        ],
        ["X", "Y"],
    )

    return build_hmm(
        [ch1, ch2],
        n_states=2,
        initial_probs=np.array([0.7, 0.3]),
        transition_probs=np.array(
            [
                [0.85, 0.15],
                [0.20, 0.80],
            ]
        ),
        emission_probs=[
            np.array(
                [
                    [0.90, 0.10],
                    [0.15, 0.85],
                ]
            ),
            np.array(
                [
                    [0.80, 0.20],
                    [0.10, 0.90],
                ]
            ),
        ],
        channel_names=["letter", "marker"],
    )


def test_multichannel_hmm_score_works_without_hmmlearn_backend():
    model = _multichannel_model()

    log_likelihood = model.score()

    assert np.isfinite(log_likelihood)
    assert log_likelihood < 0


def test_multichannel_hmm_predict_and_posterior_work_without_hmmlearn_backend():
    model = _multichannel_model()

    states = predict(model)
    posterior = posterior_probs(model)

    assert states.shape == (2 * len(TIME_COLS),)
    assert set(states).issubset({0, 1})
    assert list(posterior.columns) == ["id", "time", "state", "probability"]
    assert len(posterior) == 2 * len(TIME_COLS) * model.n_states
    summed = posterior.groupby(["id", "time"])["probability"].sum()
    assert np.allclose(summed.to_numpy(), 1.0)


def test_multichannel_hmm_rejects_shuffled_channel_ids():
    ch1 = _seqdata(
        [["A", "A", "B", "B"], ["B", "B", "A", "A"]],
        ["A", "B"],
        ids=["s1", "s2"],
    )
    ch2 = _seqdata(
        [["X", "X", "Y", "Y"], ["Y", "Y", "X", "X"]],
        ["X", "Y"],
        ids=["s2", "s1"],
    )

    with pytest.raises(ValueError, match="same sequence IDs in the same order"):
        build_hmm([ch1, ch2], n_states=2, random_state=1)
