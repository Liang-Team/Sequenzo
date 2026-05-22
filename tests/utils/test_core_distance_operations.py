import importlib
from unittest import mock

import numpy as np
import pytest


def test_weighted_inertia_contrib_matches_reference():
    from sequenzo.utils.core_distance_operations import weighted_inertia_contrib

    dist = np.array(
        [
            [0.0, 2.0, 4.0],
            [2.0, 0.0, 6.0],
            [4.0, 6.0, 0.0],
        ],
        dtype=np.float64,
    )
    indices = np.array([0, 2], dtype=np.int32)
    weights = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    result = weighted_inertia_contrib(dist, indices, weights)
    expected = np.array([3.0, 1.0], dtype=np.float64)
    np.testing.assert_allclose(result, expected)


def test_weighted_inertia_contrib_raises_when_extension_missing():
    from sequenzo.utils import core_distance_operations as cdo

    cdo._c_extension = None

    original_import_module = importlib.import_module

    def _fail_core_distance_import(name, package=None):
        if name.endswith("core_distance_c_code"):
            raise ImportError("simulated Windows DLL load failure")
        return original_import_module(name, package=package)

    dist = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    indices = np.array([0, 1], dtype=np.int32)
    weights = np.array([1.0, 1.0], dtype=np.float64)

    with pytest.raises(ImportError, match="core distance operations"):
        with mock.patch(
            "sequenzo.utils.core_distance_operations.importlib.import_module",
            side_effect=_fail_core_distance_import,
        ):
            cdo.weighted_inertia_contrib(dist, indices, weights)
