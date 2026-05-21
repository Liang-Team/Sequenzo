"""
Parity tests against R MCseqReplic (reference CSVs from generate_r_reference.R).
"""
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sequenzo import SequenceData
from sequenzo.uncertainty import (
    ch_dur,
    get_distance_matrices_per_replicate as mc_disslist,
    get_timing_error_distribution as mc_pj,
    get_timing_perturbed_sequences as mc_seq_replicate,
    get_distance_matrix_stability as mc_seqdist_se,
)
from sequenzo.dissimilarity_measures.utils.seqdur import seqdur

REF_DIR = Path(__file__).resolve().parent / "reference"


def _mini_exdata():
    raw = """
a a b b
a a b b
b b a a
a c c b
b b a c
b b a c
"""
    frame = pd.read_csv(StringIO(raw.strip()), sep=r"\s+", header=None, dtype=str)
    frame.columns = ["t1", "t2", "t3", "t4"]
    states = ["a", "b", "c"]
    return SequenceData(
        frame,
        time=list(frame.columns),
        states=states,
        id_col=None,
        weights=np.ones(len(frame)),
    )


@pytest.fixture
def mini_seq():
    return _mini_exdata()


def test_ch_dur_matches_r_reference():
    ref_path = REF_DIR / "ch_dur_j1.csv"
    if not ref_path.is_file():
        pytest.skip("R reference missing; run tests/mc_replic/generate_r_reference.R")

    seq = _mini_exdata()
    # Same 2-spell sequence as R reference (first row of mini data)
    sub = SequenceData(
        seq.data.iloc[[0]].copy().reset_index(drop=True),
        time=seq.time,
        states=seq.states,
        labels=seq.labels,
    )
    sd = seqdur(sub)[0].astype(np.int64)
    from sequenzo.uncertainty.r_random import r_random_state

    rng = r_random_state(42)
    out = ch_dur(sd, 1, rng=rng)
    ref = pd.read_csv(ref_path)["dur"].to_numpy(dtype=np.int64)
    active = out[out > 0]
    np.testing.assert_array_equal(active, ref)


def test_mc_pj_matches_r_reference():
    ref_path = REF_DIR / "mc_pj.csv"
    if not ref_path.is_file():
        pytest.skip("R reference missing")

    pj, _ = mc_pj(1.2, pzero=0.4)
    ref = pd.read_csv(ref_path)["pj"].to_numpy(dtype=float)
    np.testing.assert_allclose(pj, ref, rtol=1e-4, atol=1e-5)


def test_mc_seq_replicate_matches_r_reference():
    ref_path = REF_DIR / "replicate_r1.csv"
    if not ref_path.is_file():
        pytest.skip("R reference missing")

    seq = _mini_exdata()
    time_cols = list(seq.time)
    rep = mc_seq_replicate(seq, J=1, R=3, model="keep.dss", random_engine="r", rng=25)
    got = rep[0].data[time_cols].to_numpy(dtype=str)
    ref_df = pd.read_csv(ref_path, index_col=0)
    ref_cols = [c for c in ref_df.columns if c.startswith("V")]
    ref = ref_df[ref_cols].to_numpy(dtype=str)
    assert got.shape == ref.shape
    assert (got == ref).all()


def test_mc_seqdist_se_matches_r_reference():
    mean_path = REF_DIR / "mc_mean_ham.csv"
    sd_path = REF_DIR / "mc_sd_ham.csv"
    if not mean_path.is_file() or not sd_path.is_file():
        pytest.skip("R reference missing")

    seq = _mini_exdata()
    alt = mc_seq_replicate(seq, J=1, R=3, include_obs=True, random_engine="r", rng=25)
    disslist = mc_disslist(alt, method="HAM", full_matrix=True)
    res = mc_seqdist_se(disslist, full_matrix=True)

    ref_mean = pd.read_csv(mean_path, index_col=0).to_numpy(dtype=float)
    ref_sd = pd.read_csv(sd_path, index_col=0).to_numpy(dtype=float)
    got_mean = np.asarray(res.mc_mean, dtype=float)
    got_sd = np.asarray(res.mc_sd, dtype=float)
    np.testing.assert_allclose(got_mean, ref_mean, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(got_sd, ref_sd, rtol=1e-6, atol=1e-8)
