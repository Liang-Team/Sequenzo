import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import squareform

from sequenzo import SequenceData
from sequenzo.dissimilarity_measures import get_distance_matrix
from sequenzo.dissimilarity_measures.__init__ import _import_c_code


TWED_SM_2_STATE = np.array([[0.0, 2.0], [2.0, 0.0]], dtype=np.float64)


def _sequence_data_from_numeric_rows(rows, *, ids=None, states=(1, 2)):
    width = max(len(row) for row in rows)
    padded = [list(row) + [1] * (width - len(row)) for row in rows]
    time_cols = [f"T{i + 1}" for i in range(width)]
    if ids is None:
        ids = [f"s{i}" for i in range(len(rows))]
    df = pd.DataFrame(padded, columns=time_cols)
    df.insert(0, "id", ids)
    return SequenceData(df, time=time_cols, states=list(states), id_col="id")


def _set_effective_lengths(seqdata, lengths):
    values = seqdata.seqdata.to_numpy(copy=True)
    for row_idx, length in enumerate(lengths):
        if length < values.shape[1]:
            values[row_idx, length:] = 0
    seqdata.seqdata.loc[:, :] = values
    return seqdata


def _make_twed_reference_seqdata(order=None):
    rows = [
        [1, 1, 2, 2, 1],
        [1, 2, 2, 1, 1],
        [2, 1, 1, 2, 2],
        [1, 1, 1, 2, 2],
        [2, 2, 1, 1, 2],
    ]
    ids = [f"s{i}" for i in range(len(rows))]
    if order is not None:
        rows = [rows[i] for i in order]
        ids = [ids[i] for i in order]
    return _sequence_data_from_numeric_rows(rows, ids=ids)


def _twed_kwargs(**overrides):
    kwargs = {
        "method": "TWED",
        "norm": "none",
        "nu": 0.5,
        "h": 0.5,
        "sm": TWED_SM_2_STATE,
        "indel": 2.0,
    }
    kwargs.update(overrides)
    return kwargs


def _augment_twed_sm(sm, indel):
    sm = np.asarray(sm, dtype=np.float64)
    out = np.full((sm.shape[0] + 1, sm.shape[1] + 1), np.nan, dtype=np.float64)
    out[1:, 1:] = sm
    out[0, 0] = 0.0
    out[0, 1:] = indel
    out[1:, 0] = indel
    return out


def _python_twed_true_infinity(seq_a, seq_b, sm, *, indel, nu, h):
    m = len(seq_a)
    n = len(seq_b)
    prev = np.empty(n + 1, dtype=np.float64)
    curr = np.empty(n + 1, dtype=np.float64)
    prev[0] = 0.0
    for j in range(1, n + 1):
        prev[j] = j * indel

    for i in range(1, m + 1):
        i_state = seq_a[i - 1]
        i_prev_state = 0 if i == 1 else seq_a[i - 2]
        curr[0] = i * indel
        for j in range(1, n + 1):
            j_state = seq_b[j - 1]
            j_prev_state = 0 if j == 1 else seq_b[j - 2]
            if i_state == j_state and i_prev_state == j_prev_state:
                cost = 0.0
            else:
                cost = sm[i_state, j_state] + sm[i_prev_state, j_prev_state]
            substitute = prev[j - 1] + cost + 2.0 * nu * abs(i - j)

            if j > 1:
                insert = curr[j - 1] + sm[j_state, j_prev_state] + nu + h
            else:
                insert = np.inf
                if i > 1:
                    substitute = np.inf

            if i > 1:
                delete = prev[j] + sm[i_state, i_prev_state] + nu + h
            else:
                delete = np.inf
                if j > 1:
                    substitute = np.inf

            curr[j] = min(substitute, insert, delete)
        prev, curr = curr, prev

    return float(prev[n])


def _normalize_distance(rawdist, maxdist, l1, l2, norm):
    if abs(rawdist) < 1e-10:
        return 0.0
    if norm == "none":
        return rawdist
    if norm == "maxlength":
        return rawdist / max(l1, l2) if max(l1, l2) > 0.0 else 0.0
    if norm == "gmean":
        return 0.0 if abs(l1 - l2) < 1e-10 else 1.0 if abs(l1 * l2) < 1e-10 else 1.0 - (
            (maxdist - rawdist) / (2.0 * np.sqrt(l1) * np.sqrt(l2))
        )
    if norm == "maxdist":
        return 1.0 if abs(maxdist) < 1e-10 else rawdist / maxdist
    if norm == "YujianBo":
        return 1.0 if abs(maxdist) < 1e-10 else (2.0 * rawdist) / (rawdist + maxdist)
    raise ValueError(f"unsupported test norm: {norm}")


def _python_twed_reference(seq_a, seq_b, sm, *, indel, nu, h, norm):
    raw = _python_twed_true_infinity(seq_a, seq_b, sm, indel=indel, nu=nu, h=h)
    if norm == "YujianBo":
        maxscost = 2.0 * indel
    else:
        finite_upper = [
            sm[i, j]
            for i in range(sm.shape[0])
            for j in range(i + 1, sm.shape[1])
            if np.isfinite(sm[i, j])
        ]
        maxscost = min(max(finite_upper, default=0.0), 2.0 * indel)

    m = len(seq_a)
    n = len(seq_b)
    maxdist = abs(n - m) * (nu + h + maxscost) + 2.0 * (maxscost + nu) * min(m, n)
    return _normalize_distance(raw, maxdist, m * indel, n * indel, norm)


def test_twed_long_unbalanced_matches_true_infinity_reference():
    """Long unbalanced TWED paths should not be capped by a finite sentinel."""
    length = 1100
    seqdata = _sequence_data_from_numeric_rows(
        [[1] * length, [1] * length],
        ids=["short", "long"],
        states=(1,),
    )
    _set_effective_lengths(seqdata, [1, length])

    sm = np.array([[0.0]], dtype=np.float64)
    kwargs = _twed_kwargs(sm=sm, indel=0.01, nu=1.0, h=1.0)
    observed = get_distance_matrix(seqdata, **kwargs)
    expected = _python_twed_true_infinity(
        [1],
        [1] * length,
        _augment_twed_sm(sm, indel=0.01),
        indel=0.01,
        nu=1.0,
        h=1.0,
    )

    assert expected == 2198.0
    assert observed.loc["short", "long"] == pytest.approx(expected, abs=1e-9)


def test_twed_long_unbalanced_zero_state_does_not_hit_finite_sentinel():
    """A one-state sequence against a long sequence should use true infinity boundaries."""
    length = 3002
    seqdata = _sequence_data_from_numeric_rows(
        [[1] * length, [1] * length],
        ids=["short", "long"],
        states=(1,),
    )
    _set_effective_lengths(seqdata, [1, length])

    observed = get_distance_matrix(
        seqdata,
        method="TWED",
        norm="none",
        nu=0.5,
        h=0.5,
        sm=np.array([[0.0]], dtype=np.float64),
        indel=0.0,
    )

    assert observed.loc["short", "long"] == pytest.approx(length - 1, abs=1e-9)


def test_twed_matches_independent_small_dp_oracle_for_multiple_norms():
    """The optimized 2-row recurrence should match an independent TWED oracle."""
    rng = np.random.default_rng(20260517)
    lengths = [0, 1, 2, 3, 5, 7, 8, 9]
    rows = [rng.integers(1, 4, size=length).tolist() for length in lengths]
    seqdata = _sequence_data_from_numeric_rows(rows, states=(1, 2, 3))
    _set_effective_lengths(seqdata, lengths)

    sm = np.array(
        [
            [0.0, 1.5, 3.0],
            [1.5, 0.0, 2.25],
            [3.0, 2.25, 0.0],
        ],
        dtype=np.float64,
    )
    indel = 1.75
    nu = 0.3
    h = 0.8
    augmented_sm = _augment_twed_sm(sm, indel=indel)

    for norm in ("none", "YujianBo"):
        observed = get_distance_matrix(
            seqdata,
            method="TWED",
            norm=norm,
            nu=nu,
            h=h,
            sm=sm,
            indel=indel,
        ).to_numpy(dtype=np.float64)
        expected = np.zeros_like(observed)
        for i, seq_a in enumerate(rows):
            for j, seq_b in enumerate(rows):
                expected[i, j] = _python_twed_reference(
                    seq_a,
                    seq_b,
                    augmented_sm,
                    indel=indel,
                    nu=nu,
                    h=h,
                    norm=norm,
                )

        np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-9)


def test_twed_refseq_single_int_returns_series_matching_full_matrix_column():
    """refseq=int should return the corresponding full-matrix column."""
    seqdata = _make_twed_reference_seqdata()
    full = get_distance_matrix(seqdata, **_twed_kwargs())

    ref = get_distance_matrix(seqdata, refseq=0, **_twed_kwargs())

    assert isinstance(ref, pd.Series)
    assert list(ref.index) == list(seqdata.ids)
    np.testing.assert_allclose(ref.to_numpy(), full.iloc[:, 0].to_numpy(), rtol=0, atol=1e-9)


def test_twed_accepts_numpy_scalar_parameters_and_refseq_index():
    """Common NumPy scalar inputs should be accepted by the TWED API."""
    seqdata = _make_twed_reference_seqdata()
    kwargs = _twed_kwargs(indel=np.int64(2), nu=np.float64(0.5), h=np.float64(0.5))
    full = get_distance_matrix(seqdata, **kwargs)

    ref = get_distance_matrix(seqdata, refseq=np.int64(0), **kwargs)

    assert isinstance(ref, pd.Series)
    np.testing.assert_allclose(ref.to_numpy(), full.iloc[:, 0].to_numpy(), rtol=0, atol=1e-9)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"nu": np.nan}, "'nu'"),
        ({"nu": np.inf}, "'nu'"),
        ({"h": np.nan}, "'h'"),
        ({"h": np.inf}, "'h'"),
        ({"indel": np.nan}, "'indel'"),
        ({"indel": -0.1}, "'indel'"),
        ({"indel": [1.0, np.inf]}, "'indel'"),
        ({"indel": [-1.0, 2.0]}, "'indel'"),
    ],
)
def test_twed_rejects_nonfinite_or_negative_cost_parameters(overrides, match):
    """Invalid TWED costs should fail before producing NaN/inf output."""
    seqdata = _make_twed_reference_seqdata()

    with pytest.raises(ValueError, match=match):
        get_distance_matrix(seqdata, **_twed_kwargs(**overrides))


def test_twed_elzinga_studer_refseq_index_is_normalized_to_reference_unit_distances():
    """ElzingaStuder refseq output should be normalized to reference unit distances."""
    seqdata = _make_twed_reference_seqdata()

    ref = get_distance_matrix(
        seqdata,
        refseq=0,
        **_twed_kwargs(norm="ElzingaStuder"),
    )

    assert isinstance(ref, pd.Series)
    assert ref.iloc[0] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(ref.iloc[1:].to_numpy(), np.ones(len(ref) - 1), rtol=0, atol=1e-12)


def test_twed_elzinga_studer_refseq_index_rejects_different_normalization_reference():
    """refseq output needs a full matrix when using a different normalization reference."""
    seqdata = _make_twed_reference_seqdata()

    with pytest.raises(ValueError, match="requires full_matrix=True"):
        get_distance_matrix(
            seqdata,
            refseq=0,
            **_twed_kwargs(norm="ElzingaStuder", normalization_reference_index=1),
        )


def test_twed_matrix_sm_indel_auto_uses_traminer_empty_sequence_formula():
    """TWED matrix sm with indel='auto' should use the TraMineR formula."""
    sm = np.array([[0.0, 8.0], [8.0, 0.0]], dtype=np.float64)
    nu = 0.25
    h = 0.75
    seqdata = _sequence_data_from_numeric_rows(
        [[1, 1, 1], [2, 1, 2]],
        ids=["empty", "nonempty"],
    )
    _set_effective_lengths(seqdata, [0, 3])

    observed = get_distance_matrix(
        seqdata,
        method="TWED",
        norm="none",
        nu=nu,
        h=h,
        sm=sm,
        indel="auto",
    )

    expected_indel = 2.0 * np.max(sm) + nu + h
    assert expected_indel == 17.0
    assert observed.loc["empty", "nonempty"] == pytest.approx(3 * expected_indel, abs=1e-9)


def test_twed_generated_constant_sm_indel_auto_uses_traminer_formula():
    """Generated substitution costs should use the TWED indel='auto' formula."""
    nu = 0.25
    h = 0.75
    seqdata = _sequence_data_from_numeric_rows(
        [[1, 1, 1], [2, 1, 2]],
        ids=["empty", "nonempty"],
    )
    _set_effective_lengths(seqdata, [0, 3])

    observed = get_distance_matrix(
        seqdata,
        method="TWED",
        norm="none",
        nu=nu,
        h=h,
        sm="CONSTANT",
        indel="auto",
    )

    expected_indel = 2.0 * 2.0 + nu + h
    assert expected_indel == 5.0
    assert observed.loc["empty", "nonempty"] == pytest.approx(3 * expected_indel, abs=1e-9)


def test_twed_accepts_list_indel_by_using_max_cost():
    """Python list indel should use TraMineR max-vector semantics for TWED."""
    seqdata = _make_twed_reference_seqdata()

    observed = get_distance_matrix(seqdata, **_twed_kwargs(indel=[2.0, 2.0]))
    expected = get_distance_matrix(seqdata, **_twed_kwargs(indel=2.0))

    np.testing.assert_allclose(observed.to_numpy(), expected.to_numpy(), rtol=0, atol=1e-9)


def test_twed_full_matrix_false_matches_squareform_of_full_matrix():
    """Condensed TWED output should match squareform(full_matrix)."""
    seqdata = _make_twed_reference_seqdata()

    full = get_distance_matrix(seqdata, **_twed_kwargs())
    condensed = get_distance_matrix(seqdata, full_matrix=False, **_twed_kwargs())

    assert isinstance(condensed, np.ndarray)
    assert condensed.shape == (len(seqdata.ids) * (len(seqdata.ids) - 1) // 2,)
    np.testing.assert_allclose(squareform(condensed), full.to_numpy(), rtol=0, atol=1e-9)


def test_twed_elzinga_studer_full_matrix_false_matches_squareform_of_full_matrix():
    """ElzingaStuder normalization should support condensed TWED output."""
    seqdata = _make_twed_reference_seqdata()
    kwargs = _twed_kwargs(norm="ElzingaStuder", normalization_reference_index=0)

    full = get_distance_matrix(seqdata, **kwargs)
    condensed = get_distance_matrix(seqdata, full_matrix=False, **kwargs)

    assert isinstance(condensed, np.ndarray)
    assert condensed.shape == (len(seqdata.ids) * (len(seqdata.ids) - 1) // 2,)
    np.testing.assert_allclose(squareform(condensed), full.to_numpy(), rtol=0, atol=1e-9)


def test_twed_elzinga_studer_full_matrix_false_matches_full_matrix_with_duplicates():
    """Condensed normalization should preserve duplicate-to-unique mapping."""
    seqdata = _sequence_data_from_numeric_rows(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [2, 1, 1, 2],
            [2, 2, 1, 1],
        ],
        ids=["s0", "s1", "s2", "s3"],
    )
    kwargs = _twed_kwargs(norm="ElzingaStuder", normalization_reference_index=0)

    full = get_distance_matrix(seqdata, **kwargs)
    condensed = get_distance_matrix(seqdata, full_matrix=False, **kwargs)

    np.testing.assert_allclose(squareform(condensed), full.to_numpy(), rtol=0, atol=1e-9)
    assert condensed[0] == pytest.approx(0.0, abs=1e-12)


def test_twed_full_matrix_false_does_not_instantiate_full_matrix_expander(monkeypatch):
    """full_matrix=False should avoid the full-matrix expansion path."""
    c_code = _import_c_code()
    if c_code is None:
        pytest.skip("C++ extension is required for TWED distance tests")

    seqdata = _make_twed_reference_seqdata()
    expected = squareform(get_distance_matrix(seqdata, **_twed_kwargs()).to_numpy())

    class ForbiddenDist2Matrix:
        def __init__(self, *args, **kwargs):
            raise AssertionError("full_matrix=False must not instantiate c_code.dist2matrix")

    monkeypatch.setattr(c_code, "dist2matrix", ForbiddenDist2Matrix)

    condensed = get_distance_matrix(seqdata, full_matrix=False, **_twed_kwargs())

    assert isinstance(condensed, np.ndarray)
    np.testing.assert_allclose(condensed, expected, rtol=0, atol=1e-9)


def test_twed_distances_are_stable_under_input_order():
    """Ordering and deduplication should not change labeled TWED distances."""
    seqdata = _make_twed_reference_seqdata()
    permuted = _make_twed_reference_seqdata(order=[3, 1, 4, 0, 2])

    baseline = get_distance_matrix(seqdata, **_twed_kwargs())
    reordered = get_distance_matrix(permuted, **_twed_kwargs()).reindex(
        index=baseline.index,
        columns=baseline.columns,
    )

    np.testing.assert_allclose(reordered.to_numpy(), baseline.to_numpy(), rtol=0, atol=1e-9)


def test_twed_distances_are_stable_across_openmp_thread_counts():
    """OpenMP thread count should not affect TWED distances."""
    repo_root = Path(__file__).resolve().parents[2]
    script = r"""
import json
import numpy as np
import pandas as pd
from sequenzo import SequenceData
from sequenzo.dissimilarity_measures import get_distance_matrix

rows = np.array([
    [1, 1, 2, 2, 1, 2],
    [1, 2, 2, 1, 1, 2],
    [2, 1, 1, 2, 2, 1],
    [1, 1, 1, 2, 2, 2],
    [2, 2, 1, 1, 2, 1],
], dtype=int)
time_cols = [f"T{i + 1}" for i in range(rows.shape[1])]
df = pd.DataFrame(rows, columns=time_cols)
df.insert(0, "id", [f"s{i}" for i in range(rows.shape[0])])
seqdata = SequenceData(df, time=time_cols, states=[1, 2], id_col="id")
sm = np.array([[0.0, 2.0], [2.0, 0.0]], dtype=np.float64)
D = get_distance_matrix(seqdata, method="TWED", norm="none", nu=0.5, h=0.5, sm=sm, indel=2.0)
print("RESULT=" + json.dumps(D.to_numpy().tolist()))
"""

    outputs = []
    for threads in ("1", "4"):
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = threads
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repo_root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        result_line = [line for line in result.stdout.splitlines() if line.startswith("RESULT=")][-1]
        outputs.append(np.asarray(json.loads(result_line.removeprefix("RESULT=")), dtype=np.float64))

    np.testing.assert_allclose(outputs[0], outputs[1], rtol=0, atol=1e-9)


def test_openmp_runtime_info_smoke():
    """Smoke test for the C++ extension and OpenMP metadata hook."""
    c_code = _import_c_code()
    assert c_code is not None
    assert hasattr(c_code, "_openmp_runtime_info")

    info = c_code._openmp_runtime_info(1)

    assert isinstance(info, dict)
    assert "openmp_version_string" in info
    assert int(info["actual_threads"]) >= 1
