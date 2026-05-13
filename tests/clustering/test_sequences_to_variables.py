"""
Tests for sequences-to-variables (Helske et al. 2024) implementation.
"""
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist, squareform

from sequenzo.clustering.sequences_to_variables.helpers import (
    cluster_labels_to_dummies,
    dummy_column_names,
    max_distance,
    validate_membership_matrix,
)
from sequenzo.clustering.sequences_to_variables import (
    fanny,
    fanny_membership,
    hard_classification_variables,
    pseudoclass_regression,
    representativeness_matrix,
    soft_classification_variables,
)


def _ruspini_diss():
    x = np.array([
        [4, 53], [5, 63], [6, 70], [8, 76], [10, 83], [12, 88], [14, 93],
        [16, 97], [18, 100], [20, 102], [22, 104], [24, 105], [26, 106],
        [30, 107], [34, 108], [38, 108], [42, 107], [46, 106], [50, 104], [54, 101],
    ], dtype=float)
    return squareform(pdist(x))


def _toy_block_diss():
    return np.array([
        [0, 1, 1, 1, 9, 9, 9, 9],
        [1, 0, 1, 1, 9, 9, 9, 9],
        [1, 1, 0, 1, 9, 9, 9, 9],
        [1, 1, 1, 0, 9, 9, 9, 9],
        [9, 9, 9, 9, 0, 1, 1, 1],
        [9, 9, 9, 9, 1, 0, 1, 1],
        [9, 9, 9, 9, 1, 1, 0, 1],
        [9, 9, 9, 9, 1, 1, 1, 0],
    ], dtype=float)


def _r_fanny_reference(diss, k, memb_exp):
    with tempfile.TemporaryDirectory() as tmp:
        diss_path = os.path.join(tmp, "diss.csv")
        memb_path = os.path.join(tmp, "memb.csv")
        clu_path = os.path.join(tmp, "clu.csv")
        obj_path = os.path.join(tmp, "obj.txt")
        pc_path = os.path.join(tmp, "pc.txt")
        pd.DataFrame(diss).to_csv(diss_path)
        script = os.path.join(tmp, "run.R")
        with open(script, "w") as f:
            f.write(
                f"""
suppressPackageStartupMessages(library(cluster))
d <- as.matrix(read.csv("{diss_path}", row.names=1))
fit <- fanny(d, k={k}, diss=TRUE, memb.exp={memb_exp}, maxit=500, tol=1e-15)
write.table(fit$membership, "{memb_path}", row.names=FALSE, col.names=FALSE)
write.table(fit$clustering, "{clu_path}", row.names=FALSE, col.names=FALSE)
writeLines(as.character(fit$objective[["objective"]]), "{obj_path}")
writeLines(as.character(fit$coeff[["dunn_coeff"]]), "{pc_path}")
writeLines(as.character(fit$k.crisp), "{os.path.join(tmp, "kcrisp.txt")}")
"""
            )
        proc = subprocess.run(
            ["Rscript", script], capture_output=True, text=True, timeout=180,
        )
        if proc.returncode != 0:
            pytest.skip(f"R comparison failed: {proc.stderr}")
        memb = np.loadtxt(memb_path)
        clustering = np.loadtxt(clu_path, dtype=int)
        objective = float(open(obj_path).read().strip())
        pc = float(open(pc_path).read().strip())
        k_crisp = int(open(os.path.join(tmp, "kcrisp.txt")).read().strip())
        return memb, objective, clustering - 1, pc, k_crisp


RSCRIPT_AVAILABLE = subprocess.run(["which", "Rscript"], capture_output=True).returncode == 0


def test_max_distance_square_and_condensed():
    diss = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float)
    assert max_distance(diss) == 3.0
    assert max_distance(squareform(diss)) == 3.0
    assert max_distance(np.array([[0.0]])) == 0.0


def test_cluster_labels_to_dummies_0_based():
    labels = np.array([0, 1, 2, 0, 1])
    d = cluster_labels_to_dummies(labels, k=3, reference=0)
    assert d.shape == (5, 2)
    np.testing.assert_array_equal(d[1], [1, 0])
    np.testing.assert_array_equal(d[2], [0, 1])


def test_cluster_labels_to_dummies_1_based():
    labels = np.array([1, 2, 3, 1, 2])
    d = cluster_labels_to_dummies(labels, k=3, reference=0)
    assert d.shape == (5, 2)


def test_dummy_column_names():
    labels = np.array([1, 2, 3, 1, 2])
    assert dummy_column_names(labels, k=3, reference=0) == ["C_2", "C_3"]
    with pytest.raises(ValueError, match="does not match k"):
        dummy_column_names([1, 1, 2], k=3, reference=0)


def test_representativeness_matrix_basic():
    diss = np.array([
        [0, 1, 2, 4],
        [1, 0, 3, 3],
        [2, 3, 0, 2],
        [4, 3, 2, 0],
    ], dtype=float)
    R = representativeness_matrix(diss, [0, 3], d_max=4.0)
    assert R.shape == (4, 2)
    assert R[0, 0] == 1.0
    assert R[3, 1] == 1.0


def test_representativeness_matrix_dataframe_index():
    diss = pd.DataFrame(
        [[0, 1], [1, 0]],
        index=["s1", "s2"],
        columns=["s1", "s2"],
    )
    df = representativeness_matrix(diss, [0], d_max=1.0, as_dataframe=True)
    assert list(df.index) == ["s1", "s2"]


def test_representativeness_dmax_zero():
    diss = np.zeros((3, 3), dtype=float)
    R = representativeness_matrix(diss, [0, 1], d_max=0.0)
    np.testing.assert_allclose(R, np.ones((3, 2)))


def test_representativeness_rejects_invalid_diss():
    diss = np.array([[0, -1], [-1, 0]], dtype=float)
    with pytest.raises(ValueError, match="nonnegative"):
        representativeness_matrix(diss, [0])


def test_soft_classification_variables_reference_omitted():
    U = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    X = soft_classification_variables(U, reference=0)
    assert X.shape == (3, 2)
    np.testing.assert_allclose(X, U[:, 1:])


def test_validate_membership_matrix_rejects_bad_rows():
    with pytest.raises(ValueError, match="Rows of U must sum to 1"):
        validate_membership_matrix(np.array([[0.5, 0.3, 0.1]] * 3))


def test_pseudoclass_regression_ols_runs():
    rng = np.random.default_rng(0)
    n = 40
    U = np.array([[0.6, 0.3, 0.1]] * n, dtype=float)
    y = rng.normal(size=n)
    result = pseudoclass_regression(y, U, M=10, random_state=0, model_type="ols")
    assert result["m_eff"] > 0
    assert result["failed"] == 10 - result["m_eff"]


def test_pseudoclass_regression_logit_runs():
    rng = np.random.default_rng(1)
    n = 60
    U = np.array([[0.5, 0.3, 0.2]] * n, dtype=float)
    y = (rng.normal(size=n) > 0).astype(float)
    result = pseudoclass_regression(y, U, M=10, random_state=1, model_type="logit")
    assert result["m_eff"] > 0


def test_pseudoclass_regression_validates_m_and_y():
    U = np.array([[0.6, 0.4]] * 5)
    y = np.zeros(3)
    with pytest.raises(ValueError, match="M must be at least 1"):
        pseudoclass_regression(y, U, M=0)
    with pytest.raises(ValueError, match="same number of rows"):
        pseudoclass_regression(y, U, M=5)


def test_fanny_rows_sum_to_one():
    diss = _ruspini_diss()
    U, _ = fanny_membership(diss, k=4, m=1.5, max_iter=500)
    np.testing.assert_allclose(U.sum(axis=1), np.ones(20), atol=1e-10)


def test_fanny_k_one():
    diss = np.array([[0, 1], [1, 0]], dtype=float)
    result = fanny(diss, k=1, memb_exp=1.4)
    np.testing.assert_allclose(result.membership, np.ones((2, 1)))


@pytest.mark.skipif(not RSCRIPT_AVAILABLE, reason="Rscript not available")
def test_fanny_against_R_cluster_fanny_small_matrix():
    diss = _toy_block_diss()
    rmemb, r_obj, r_clu, r_pc, r_k_crisp = _r_fanny_reference(diss, k=2, memb_exp=1.4)
    py = fanny(diss, k=2, memb_exp=1.4, max_iter=500, tol=1e-15)
    np.testing.assert_allclose(py.membership, rmemb, atol=1e-12)
    assert abs(py.objective - r_obj) < 1e-5
    np.testing.assert_array_equal(py.clustering, r_clu)
    assert py.k_crisp == r_k_crisp
    assert abs(py.partition_coefficient - r_pc) < 1e-10


@pytest.mark.parametrize("memb_exp", [1.4, 1.5, 2.0])
@pytest.mark.skipif(not RSCRIPT_AVAILABLE, reason="Rscript not available")
def test_fanny_against_R_ruspini(memb_exp):
    diss = _ruspini_diss()
    rmemb, r_obj, _, r_pc, _ = _r_fanny_reference(diss, k=4, memb_exp=memb_exp)
    py = fanny(diss, k=4, memb_exp=memb_exp, max_iter=500, tol=1e-15)
    np.testing.assert_allclose(py.membership, rmemb, atol=1e-12)
    assert abs(py.objective - r_obj) < 1e-5
    if memb_exp == 2.0:
        assert abs(py.partition_coefficient - r_pc) < 1e-10


def test_hard_classification_variables_dataframe():
    labels = np.array([1, 2, 3, 1, 2])
    df = hard_classification_variables(
        labels, k=3, reference=0, ids=["i", "j", "k", "l", "m"], as_dataframe=True,
    )
    assert list(df.columns) == ["C_2", "C_3"]
