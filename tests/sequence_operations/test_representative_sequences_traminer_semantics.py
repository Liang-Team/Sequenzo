import numpy as np
import pandas as pd

from sequenzo import (
    SequenceData,
    get_distance_center,
    get_relative_frequency_groups,
    get_relative_frequency_representatives,
    get_representative_objects,
    get_representative_sequences,
)
from sequenzo.dissimilarity_measures import get_distance_matrix


def _toy_seqdata() -> SequenceData:
    raw = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "T1": ["A", "A", "B", "B"],
            "T2": ["A", "B", "B", "A"],
            "T3": ["A", "B", "A", "A"],
        }
    )
    return SequenceData(data=raw, time=["T1", "T2", "T3"], states=["A", "B"], id_col="id")


def test_get_distance_center_returns_group_medoid_index():
    D = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ]
    )
    med = get_distance_center(D, medoids_index="first")
    assert med.shape == (1,)
    assert int(med[0]) == 1


def test_get_relative_frequency_groups_and_get_relative_frequency_representatives_return_valid_medoids():
    s = _toy_seqdata()
    D = get_distance_matrix(s, method="OM", indel=1, sm="CONSTANT")
    rf = get_relative_frequency_groups(D, k=2, grp_meth="prop")
    assert len(rf["medoids"]) == 2
    assert all(0 <= int(i) < s.n_sequences for i in rf["medoids"])

    srf = get_relative_frequency_representatives(s, D, k=2, grp_meth="prop")
    assert srf["seqtoplot"].shape[0] == 2
    assert srf["seqtoplot"].shape[1] == s.n_steps


def test_get_representative_objects_and_sequences_return_representatives():
    s = _toy_seqdata()
    D = get_distance_matrix(s, method="OM", indel=1, sm="CONSTANT")
    dr = get_representative_objects(D, criterion="density", coverage=0.5, pradius=0.3)
    assert dr["indices"].size >= 1
    assert 0.0 <= dr["quality"] <= 1.0

    sr = get_representative_sequences(s, diss=D, criterion="density", coverage=0.5, pradius=0.3)
    assert sr["sequences"].shape[0] == sr["indices"].size
    assert 0.0 <= sr["quality"] <= 1.0
