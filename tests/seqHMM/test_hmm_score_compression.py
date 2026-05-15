import numpy as np
import pandas as pd

from sequenzo import SequenceData
from sequenzo.seqhmm import build_hmm


def _seqdata(rows, states):
    time_cols = [f"t{i}" for i in range(1, len(rows[0]) + 1)]
    df = pd.DataFrame(rows, columns=time_cols)
    df.insert(0, "id", [f"s{i}" for i in range(len(df))])
    return SequenceData(df, time=time_cols, states=states, id_col="id")


def test_compressed_score_matches_single_channel_score_with_repeated_sequences():
    seq = _seqdata(
        [
            ["A", "A", "B", "B", "A"],
            ["A", "A", "B", "B", "A"],
            ["B", "B", "A", "A", "B"],
            ["B", "B", "A", "A", "B"],
            ["B", "B", "A", "A", "B"],
        ],
        ["A", "B"],
    )
    model = build_hmm(
        seq,
        initial_probs=np.array([0.55, 0.45]),
        transition_probs=np.array([[0.80, 0.20], [0.25, 0.75]]),
        emission_probs=np.array([[0.85, 0.15], [0.20, 0.80]]),
    )

    assert np.isclose(model.score(compress=True), model.score(), atol=1e-10)


def test_compressed_score_matches_multichannel_score_with_repeated_sequences():
    ch1 = _seqdata(
        [
            ["A", "A", "B", "B"],
            ["A", "A", "B", "B"],
            ["B", "B", "A", "A"],
        ],
        ["A", "B"],
    )
    ch2 = _seqdata(
        [
            ["X", "X", "Y", "Y"],
            ["X", "X", "Y", "Y"],
            ["Y", "Y", "X", "X"],
        ],
        ["X", "Y"],
    )
    model = build_hmm(
        [ch1, ch2],
        n_states=2,
        initial_probs=np.array([0.65, 0.35]),
        transition_probs=np.array([[0.75, 0.25], [0.15, 0.85]]),
        emission_probs=[
            np.array([[0.90, 0.10], [0.25, 0.75]]),
            np.array([[0.80, 0.20], [0.15, 0.85]]),
        ],
    )

    assert np.isclose(model.score(compress=True), model.score(), atol=1e-10)
