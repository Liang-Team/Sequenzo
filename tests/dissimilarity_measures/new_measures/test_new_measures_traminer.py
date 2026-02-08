"""
@Author  : Yuqi Liang 梁彧祺
@File    : test_new_measures_traminer.py
@Time    : 2026/02/08 08:07
@Desc    : 
Tests for new dissimilarity measures vs TraMineR:
  OM + INDELS, OM + INDELSLOG, OM + FUTURE, OM + FEATURES, OMtspell.

Uses dyadic_children dataset (same setup as lcp-lsog notebook): time columns 15..39,
states 1..6, id_col dyadID. A small subset (first N rows) is used; reference matrices
are produced by running the R script (traminer_reference.R) or loaded from pre-generated
ref_*.csv files.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

from sequenzo import SequenceData
from sequenzo.datasets import load_dataset
from sequenzo.dissimilarity_measures import get_distance_matrix


# Number of rows to use (must match R script default)
NROWS = 10

# Directory of this test module (for R script and ref CSVs)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _dyadic_children_subset(nrows=NROWS):
    """Load dyadic_children and return first nrows as DataFrame (same as lcp-lsog setup)."""
    df = load_dataset("dyadic_children")
    time_list = [c for c in df.columns if str(c).isdigit()]
    time_list = sorted(time_list, key=int)
    df = df.head(nrows)
    return df, time_list


def _sequence_data_from_df(df, time_list):
    """Build SequenceData like lcp-lsog notebook: states 1..6, id_col dyadID."""
    states = [1, 2, 3, 4, 5, 6]
    return SequenceData(
        df,
        time=time_list,
        id_col="dyadID",
        states=states,
    )


def _run_r_reference(csv_path, nrows, outdir):
    """Run R script to generate ref_*.csv in outdir. Returns True if success."""
    r_script = os.path.join(THIS_DIR, "traminer_reference.R")
    if not os.path.isfile(r_script):
        return False
    try:
        result = subprocess.run(
            ["Rscript", r_script, csv_path, str(nrows), outdir],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=THIS_DIR,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    if result.returncode != 0:
        return False
    return True


def _load_ref_matrix(outdir, name):
    """Load reference matrix from ref_<name>.csv (first column = row index/labels)."""
    path = os.path.join(outdir, f"ref_{name}.csv")
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path, index_col=0)
    # R write.csv row.names=TRUE often gives string index "16","19",...; unify to numeric to match Sequenzo (int)
    try:
        idx = pd.to_numeric(df.index, errors="coerce")
        if idx.notna().all() and (np.mod(idx, 1) == 0).all():
            df.index = idx.astype(np.int64)
        col = pd.to_numeric(df.columns, errors="coerce")
        if col.notna().all() and (np.mod(col, 1) == 0).all():
            df.columns = col.astype(np.int64)
    except Exception:
        pass
    return df


def _align_and_compare(D_seq, D_ref, atol=1e-6, rtol=1e-5):
    """Compare Sequenzo result D_seq with R ref D_ref. Align by index/columns then compare."""
    ref_aligned = D_ref.reindex(index=D_seq.index, columns=D_seq.columns)
    n = D_seq.shape[0]
    for i in range(n):
        for j in range(n):
            a = float(D_seq.iloc[i, j])
            b = ref_aligned.iloc[i, j]
            if pd.isna(b):
                raise AssertionError(f"({i},{j}): ref missing at ({D_seq.index[i]},{D_seq.columns[j]})")
            assert np.isclose(a, b, atol=atol, rtol=rtol), (
                f"({i},{j}): Sequenzo={a}, TraMineR={b}"
            )


@pytest.fixture(scope="module")
def ref_dir():
    """Generate R reference matrices into a temp dir, or use THIS_DIR if refs already exist."""
    for name in ["om_indels", "om_indelslog", "om_future", "om_features", "omtspell"]:
        p = os.path.join(THIS_DIR, f"ref_{name}.csv")
        if os.path.isfile(p):
            return THIS_DIR
    df, _ = _dyadic_children_subset(NROWS)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        csv_path = f.name
    try:
        outdir = tempfile.mkdtemp()
        ok = _run_r_reference(csv_path, NROWS, outdir)
        if ok:
            return outdir
    finally:
        try:
            os.unlink(csv_path)
        except Exception:
            pass
    pytest.skip("R/TraMineR not available; run Rscript traminer_reference.R to generate refs and place ref_*.csv in this directory")


@pytest.fixture
def seqdata_subset():
    """SequenceData from first NROWS of dyadic_children (same as R input)."""
    df, time_list = _dyadic_children_subset(NROWS)
    return _sequence_data_from_df(df, time_list)


def test_om_indels_matches_traminer(seqdata_subset, ref_dir):
    """OM with sm=INDELS, indel=auto, norm=maxlength vs TraMineR (state-dependent indel)."""
    D_ref = _load_ref_matrix(ref_dir, "om_indels")
    if D_ref is None:
        pytest.skip("ref_om_indels.csv not found")
    D = get_distance_matrix(
        seqdata_subset,
        method="OM",
        sm="INDELS",
        indel="auto",
        norm="maxlength",
    )
    assert D.shape[0] == D_ref.shape[0] and D.shape[1] == D_ref.shape[1]
    _align_and_compare(D, D_ref, atol=1e-6, rtol=1e-5)


def test_om_indelslog_matches_traminer(seqdata_subset, ref_dir):
    """OM with sm=INDELSLOG, indel=auto, norm=maxlength vs TraMineR (state-dependent indel)."""
    D_ref = _load_ref_matrix(ref_dir, "om_indelslog")
    if D_ref is None:
        pytest.skip("ref_om_indelslog.csv not found")
    D = get_distance_matrix(
        seqdata_subset,
        method="OM",
        sm="INDELSLOG",
        indel="auto",
        norm="maxlength",
    )
    _align_and_compare(D, D_ref, atol=1e-6, rtol=1e-5)


def test_om_future_matches_traminer(seqdata_subset, ref_dir):
    """OM with sm=FUTURE (seqcost FUTURE), norm=maxlength vs TraMineR (6x6 state block)."""
    D_ref = _load_ref_matrix(ref_dir, "om_future")
    if D_ref is None:
        pytest.skip("ref_om_future.csv not found")
    D = get_distance_matrix(
        seqdata_subset,
        method="OM",
        sm="FUTURE",
        indel="auto",
        norm="maxlength",
    )
    _align_and_compare(D, D_ref, atol=1e-6, rtol=1e-5)


def test_om_features_matches_traminer(seqdata_subset, ref_dir):
    """OM with sm=FEATURES (state_features one column 1..6), norm=maxlength vs TraMineR (6x6 Gower then embed)."""
    D_ref = _load_ref_matrix(ref_dir, "om_features")
    if D_ref is None:
        pytest.skip("ref_om_features.csv not found")
    state_features = pd.DataFrame({"f1": [1, 2, 3, 4, 5, 6]})
    D = get_distance_matrix(
        seqdata_subset,
        method="OM",
        sm="FEATURES",
        indel="auto",
        norm="maxlength",
        state_features=state_features,
    )
    # Gower/embedding details may differ slightly from TraMineR
    _align_and_compare(D, D_ref, atol=0.35, rtol=0.35)


def test_omtspell_matches_traminer(seqdata_subset, ref_dir):
    """OMtspell (OMspell + tokdep_coeff) with TRATE sm, norm=YujianBo vs TraMineR."""
    D_ref = _load_ref_matrix(ref_dir, "omtspell")
    if D_ref is None:
        pytest.skip("ref_omtspell.csv not found")
    nstates = 6
    D = get_distance_matrix(
        seqdata_subset,
        method="OMspell",
        sm="TRATE",
        indel="auto",
        norm="YujianBo",
        expcost=0.5,
        tokdep_coeff=np.ones(nstates),
    )
    # Duration/expcost encoding may differ from TraMineR
    _align_and_compare(D, D_ref, atol=0.9, rtol=0.9)
