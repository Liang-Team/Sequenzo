import importlib
import sysconfig
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sequenzo import SequenceData, get_distance_matrix


def _seqdata_with_duplicate_reference():
    time_cols = ["T1", "T2", "T3", "T4"]
    raw = pd.DataFrame(
        [
            ["A", "A", "B", "B"],
            ["A", "A", "B", "B"],
            ["B", "B", "A", "A"],
            ["A", "B", "A", "B"],
        ],
        columns=time_cols,
    )
    raw.insert(0, "id", ["s0", "s1", "s2", "s3"])
    return SequenceData(raw, time=time_cols, states=["A", "B"], id_col="id")


def _twed_seqdata():
    time_cols = ["T1", "T2", "T3", "T4", "T5"]
    raw = pd.DataFrame(
        [
            [1, 1, 2, 2, 1],
            [1, 2, 2, 1, 1],
            [2, 1, 1, 2, 2],
            [1, 1, 1, 2, 2],
        ],
        columns=time_cols,
    )
    raw.insert(0, "id", ["s0", "s1", "s2", "s3"])
    return SequenceData(raw, time=time_cols, states=[1, 2], id_col="id")


def test_refseq_int_maps_original_index_to_unique_sequence_with_duplicates():
    seqdata = _seqdata_with_duplicate_reference()
    kwargs = dict(method="OM", sm="CONSTANT", indel=1.0, norm="none")

    full = get_distance_matrix(seqdata, **kwargs)
    ref = get_distance_matrix(seqdata, refseq=0, **kwargs)

    assert isinstance(ref, pd.Series)
    assert list(ref.index) == list(seqdata.ids)
    np.testing.assert_allclose(ref.to_numpy(dtype=np.float64), full.iloc[:, 0].to_numpy(dtype=np.float64))
    assert ref.iloc[0] == 0.0
    assert ref.iloc[1] == 0.0


def test_refseq_sets_still_returns_requested_block():
    seqdata = _seqdata_with_duplicate_reference()
    kwargs = dict(method="OM", sm="CONSTANT", indel=1.0, norm="none")

    full = get_distance_matrix(seqdata, **kwargs)
    block = get_distance_matrix(seqdata, refseq=[[0, 1], [2, 3]], **kwargs)

    assert isinstance(block, pd.DataFrame)
    assert block.shape == (2, 2)
    expected = full.iloc[[0, 1], [2, 3]]
    np.testing.assert_allclose(block.to_numpy(dtype=np.float64), expected.to_numpy(dtype=np.float64))


def test_refseq_sets_out_of_range_is_rejected_before_indexing():
    seqdata = _seqdata_with_duplicate_reference()

    with pytest.raises(ValueError, match="out of range"):
        get_distance_matrix(seqdata, refseq=[[0], [4]], method="OM", sm="CONSTANT", indel=1.0, norm="none")


def test_full_matrix_false_expands_condensed_without_dist2matrix_allocation(monkeypatch):
    import sequenzo.dissimilarity_measures.c_code as c_code

    seqdata = _seqdata_with_duplicate_reference()
    kwargs = dict(method="OM", sm="CONSTANT", indel=1.0, norm="none")
    full = get_distance_matrix(seqdata, **kwargs)

    def forbidden_dist2matrix(*args, **kwargs):
        raise AssertionError("full_matrix=False must not instantiate c_code.dist2matrix")

    monkeypatch.setattr(c_code, "dist2matrix", forbidden_dist2matrix)

    condensed = get_distance_matrix(seqdata, full_matrix=False, **kwargs)
    expected = full.to_numpy(dtype=np.float64)[np.triu_indices(len(seqdata.ids), k=1)]
    assert condensed.shape == (len(seqdata.ids) * (len(seqdata.ids) - 1) // 2,)
    np.testing.assert_allclose(condensed, expected)


def test_numeric_list_indel_is_accepted():
    seqdata = _seqdata_with_duplicate_reference()
    dist = get_distance_matrix(
        seqdata,
        method="OM",
        sm="CONSTANT",
        indel=[1.0, 2.0],
        norm="none",
    )
    assert dist.shape == (4, 4)


def test_twed_matrix_indel_auto_uses_traminer_compatible_formula(monkeypatch):
    import sequenzo.dissimilarity_measures.c_code as c_code

    seqdata = _twed_seqdata()
    sm = np.array([[0.0, 2.0], [2.0, 0.0]])
    nu = 0.5
    h = 0.5
    traminer_compatible_indel = 2 * np.nanmax(sm) + nu + h
    captured = {}

    class FakeTWEDDistance:
        def __init__(self, sequences, sm, indel, norm_num, nu, h_twed, lengths, refseq_id):
            captured["indel"] = indel
            self.nseqs = sequences.shape[0]

        def compute_all_distances(self):
            return np.zeros((self.nseqs, self.nseqs), dtype=np.float64)

    monkeypatch.setattr(c_code, "TWEDdistance", FakeTWEDDistance)

    get_distance_matrix(
        seqdata,
        method="TWED",
        sm=sm,
        indel="auto",
        nu=nu,
        h=h,
        norm="none",
    )

    assert captured["indel"] == traminer_compatible_indel


def test_twed_full_matrix_false_uses_condensed_kernel_for_duplicate_mapping(monkeypatch):
    import sequenzo.dissimilarity_measures.c_code as c_code

    seqdata = _seqdata_with_duplicate_reference()
    captured = {}

    class FakeTWEDDistance:
        def __init__(self, sequences, sm, indel, norm_num, nu, h_twed, lengths, refseq_id):
            captured["n_unique"] = sequences.shape[0]

        def compute_all_distances(self):
            raise AssertionError("TWED full_matrix=False must not allocate the unique U x U matrix")

        def compute_condensed_distances(self):
            captured["condensed_called"] = True
            assert captured["n_unique"] == 3
            # unique pairs: (0,1), (0,2), (1,2)
            return np.array([10.0, 20.0, 30.0], dtype=np.float64)

    monkeypatch.setattr(c_code, "TWEDdistance", FakeTWEDDistance)

    condensed = get_distance_matrix(
        seqdata,
        method="TWED",
        sm=np.array([[0.0, 2.0], [2.0, 0.0]], dtype=np.float64),
        indel=2.0,
        nu=0.5,
        h=0.5,
        norm="none",
        full_matrix=False,
    )

    assert captured["condensed_called"] is True
    np.testing.assert_allclose(
        condensed,
        np.array([0.0, 10.0, 20.0, 10.0, 20.0, 30.0], dtype=np.float64),
    )


def test_twed_rejects_nonfinite_state_substitution_costs():
    seqdata = _twed_seqdata()
    sm = np.array([[0.0, np.nan], [np.nan, 0.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="finite state-to-state costs"):
        get_distance_matrix(
            seqdata,
            method="TWED",
            sm=sm,
            indel=2.0,
            nu=0.5,
            h=0.5,
            norm="none",
        )


def test_twed_rejects_negative_state_substitution_costs():
    seqdata = _twed_seqdata()
    sm = np.array([[0.0, -1.0], [-1.0, 0.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="non-negative state-to-state costs"):
        get_distance_matrix(
            seqdata,
            method="TWED",
            sm=sm,
            indel=2.0,
            nu=0.5,
            h=0.5,
            norm="none",
        )


def test_get_distance_matrix_does_not_force_gc_by_default(monkeypatch):
    gdm_module = importlib.import_module("sequenzo.dissimilarity_measures.get_distance_matrix")

    seqdata = _seqdata_with_duplicate_reference()
    calls = []

    monkeypatch.setattr(gdm_module.gc, "collect", lambda: calls.append(True) or 0)

    get_distance_matrix(seqdata, method="OM", sm="CONSTANT", indel=1.0, norm="none")

    assert calls == []


def test_get_distance_matrix_can_opt_in_to_gc(monkeypatch):
    gdm_module = importlib.import_module("sequenzo.dissimilarity_measures.get_distance_matrix")

    seqdata = _seqdata_with_duplicate_reference()
    calls = []

    monkeypatch.setattr(gdm_module.gc, "collect", lambda: calls.append(True) or 0)

    get_distance_matrix(
        seqdata,
        method="OM",
        sm="CONSTANT",
        indel=1.0,
        norm="none",
        collect_garbage=True,
    )

    assert calls == [True]


def test_dissimilarity_extension_builds_without_fast_math_for_twed_ieee_semantics():
    setup_py = Path(__file__).resolve().parents[2] / "setup.py"
    source = setup_py.read_text()

    assert "def get_compile_args_for_file(filename, *, fast_math=True):" in source
    assert 'dissimilarity_compile_args = get_compile_args_for_file("dummy.cpp", fast_math=False)' in source
    assert (
        "sources=['sequenzo/dissimilarity_measures/src/module.cpp'],\n"
        "            include_dirs=get_dissimilarity_measures_include_dirs(),\n"
        "            extra_compile_args=dissimilarity_compile_args,"
    ) in source


def test_loaded_dissimilarity_extension_exposes_runtime_metadata():
    import sequenzo.dissimilarity_measures.c_code as c_code

    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    module_path = Path(c_code.__file__)
    info = c_code._openmp_runtime_info(1)

    assert module_path.exists()
    assert suffix is None or module_path.name.endswith(suffix)
    assert isinstance(info, dict)
    assert "_OPENMP" in info
    assert "compiler" in info
