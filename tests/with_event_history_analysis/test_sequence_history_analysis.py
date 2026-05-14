import numpy as np
import pandas as pd
import pytest

from sequenzo.with_event_history_analysis import (
    get_sequence_history_data,
    person_level_to_person_period,
)


def _toy_seqdata():
    return pd.DataFrame(
        [[1, 2, 3, 4], [5, 6, 7, 8]],
        columns=["t1", "t2", "t3", "t4"],
    )


def _toy_inputs():
    seqdata = _toy_seqdata()
    time = np.array([3, 2])
    event = np.array([1, 0])
    return seqdata, time, event


def _history_columns(df):
    return [col for col in df.columns if col not in {"id", "time", "event"}]


def test_package_imports_expose_callable_api():
    import sequenzo
    from sequenzo import (
        get_sequence_history_data as top_level_fn,
        person_level_to_person_period as top_level_plpp,
    )
    from sequenzo.with_event_history_analysis import (
        get_sequence_history_data as module_fn,
        person_level_to_person_period as module_plpp,
    )

    assert callable(top_level_fn)
    assert callable(top_level_plpp)
    assert top_level_fn is module_fn
    assert top_level_plpp is module_plpp
    assert "get_sequence_history_data" in sequenzo.__all__
    assert "person_level_to_person_period" in sequenzo.__all__
    assert "seqsha" not in sequenzo.__all__


@pytest.mark.parametrize(
    "include_present,align_end",
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_get_sequence_history_data_four_mode_combinations(include_present, align_end):
    seqdata, time, event = _toy_inputs()
    result = get_sequence_history_data(
        seqdata,
        time,
        event,
        include_present=include_present,
        align_end=align_end,
    )

    assert len(result) == int(time.sum())
    assert set(result["event"]) <= {False, True}
    assert result.loc[result["time"] == time[0], "event"].iloc[-1] is np.True_
    assert not result.loc[(result["id"] == 1) & (result["time"] < 3), "event"].any()


def test_include_present_false_excludes_current_state_left_aligned():
    seqdata, time, event = _toy_inputs()
    result = get_sequence_history_data(
        seqdata, time, event, include_present=False, align_end=False
    )
    history_cols = _history_columns(result)

    row_t3 = result[(result["id"] == 1) & (result["time"] == 3)].iloc[0]
    assert row_t3["t1"] == "1"
    assert row_t3["t2"] == "2"
    assert pd.isna(row_t3["t3"])

    row_t1 = result[(result["id"] == 1) & (result["time"] == 1)].iloc[0]
    assert all(pd.isna(row_t1[col]) for col in history_cols)


def test_include_present_true_includes_current_state_left_aligned():
    seqdata, time, event = _toy_inputs()
    result = get_sequence_history_data(
        seqdata, time, event, include_present=True, align_end=False
    )

    row_t3 = result[(result["id"] == 1) & (result["time"] == 3)].iloc[0]
    assert row_t3["t1"] == "1"
    assert row_t3["t2"] == "2"
    assert row_t3["t3"] == "3"

    row_t1 = result[(result["id"] == 1) & (result["time"] == 1)].iloc[0]
    assert row_t1["t1"] == "1"
    assert pd.isna(row_t1["t2"])
    assert pd.isna(row_t1["t3"])


def test_align_end_false_keeps_only_first_ma_columns():
    seqdata, time, event = _toy_inputs()
    result = get_sequence_history_data(
        seqdata, time, event, include_present=False, align_end=False
    )
    assert _history_columns(result) == ["t1", "t2", "t3"]


def test_align_end_true_keeps_full_sequence_width():
    seqdata, time, event = _toy_inputs()
    result = get_sequence_history_data(
        seqdata, time, event, include_present=False, align_end=True
    )
    assert _history_columns(result) == ["Tm4", "Tm3", "Tm2", "Tm1"]


def test_align_end_include_present_semantics():
    seqdata, time, event = _toy_inputs()
    result_excl = get_sequence_history_data(
        seqdata, time, event, include_present=False, align_end=True
    )
    result_incl = get_sequence_history_data(
        seqdata, time, event, include_present=True, align_end=True
    )

    row_excl = result_excl[(result_excl["id"] == 1) & (result_excl["time"] == 3)].iloc[0]
    row_incl = result_incl[(result_incl["id"] == 1) & (result_incl["time"] == 3)].iloc[0]

    assert pd.isna(row_excl["Tm4"])
    assert pd.isna(row_excl["Tm3"])
    assert row_excl["Tm2"] == "1"
    assert row_excl["Tm1"] == "2"
    assert row_incl["Tm3"] == "1"
    assert row_incl["Tm2"] == "2"
    assert row_incl["Tm1"] == "3"

    row_t1 = result_excl[(result_excl["id"] == 1) & (result_excl["time"] == 1)].iloc[0]
    assert all(pd.isna(row_t1[col]) for col in _history_columns(result_excl))


def test_missing_states_are_converted_to_na_orig():
    seqdata = pd.DataFrame([[1, np.nan, 3]], columns=["a", "b", "c"])
    result = get_sequence_history_data(
        seqdata, [3], [0], include_present=True, align_end=False
    )
    row = result.iloc[2]
    assert row["a"] == "1"
    assert row["b"] == "NA_orig"
    assert row["c"] == "3"


def test_align_end_right_aligns_recent_states_toward_last_column():
    seqdata, time, event = _toy_inputs()
    result = get_sequence_history_data(
        seqdata, time, event, include_present=True, align_end=True
    )

    row_t3 = result[(result["id"] == 1) & (result["time"] == 3)].iloc[0]
    assert row_t3["Tm1"] == "3"
    assert row_t3["Tm2"] == "2"
    assert row_t3["Tm3"] == "1"
    assert pd.isna(row_t3["Tm4"])


def test_empty_seqdata_is_rejected():
    with pytest.raises(ValueError, match="at least one sequence"):
        get_sequence_history_data(pd.DataFrame(), [1], [0])

    with pytest.raises(ValueError, match="at least one time column"):
        get_sequence_history_data(
            pd.DataFrame(index=[0, 1]), np.array([1, 1]), np.array([0, 0])
        )


def test_sequence_column_names_must_not_overlap_base_columns():
    seqdata = pd.DataFrame([[1, 2]], columns=["time", "t2"])
    with pytest.raises(ValueError, match="duplicate base columns"):
        get_sequence_history_data(seqdata, [2], [0])


def test_person_level_to_person_period_rejects_boolean_time():
    with pytest.raises(ValueError, match="not boolean values"):
        person_level_to_person_period(
            pd.DataFrame({"id": [1], "time": [True], "event": [0]})
        )


def test_person_level_to_person_period_validates_time_and_event():
    with pytest.raises(ValueError, match="integer durations"):
        person_level_to_person_period(
            pd.DataFrame({"id": [1], "time": ["3"], "event": [0]})
        )

    with pytest.raises(ValueError, match="at least 1"):
        person_level_to_person_period(
            pd.DataFrame({"id": [1], "time": [0], "event": [0]})
        )

    with pytest.raises(ValueError, match="boolean or 0/1"):
        person_level_to_person_period(
            pd.DataFrame({"id": [1], "time": [2], "event": [2]})
        )


def test_covar_row_count_and_column_overlap_checks():
    seqdata, time, event = _toy_inputs()

    with pytest.raises(ValueError, match="Number of rows in 'covar'"):
        get_sequence_history_data(
            seqdata, time, event, covar=pd.DataFrame({"sex": [1]})
        )

    with pytest.raises(ValueError, match="Number of rows in 'covar'"):
        get_sequence_history_data(
            seqdata, time, event, covar=np.array([[1], [2], [3]])
        )

    with pytest.raises(ValueError, match="duplicate existing columns"):
        get_sequence_history_data(
            seqdata,
            time,
            event,
            covar=pd.DataFrame({"time": [1, 2], "sex": [0, 1]}),
        )
