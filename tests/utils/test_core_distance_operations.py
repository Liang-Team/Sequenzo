import importlib
import sys
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


def test_weighted_inertia_contrib_python_fallback_when_extension_missing():
    from sequenzo.utils import core_distance_operations as cdo

    cdo._c_extension = None
    cdo._c_extension_unavailable = False

    original_import_module = importlib.import_module

    def _fail_core_distance_import(name, package=None):
        if name.endswith("core_distance_c_code"):
            raise ImportError("simulated Windows DLL load failure")
        return original_import_module(name, package=package)

    dist = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    indices = np.array([0, 1], dtype=np.int32)
    weights = np.array([1.0, 1.0], dtype=np.float64)

    with mock.patch("importlib.import_module", side_effect=_fail_core_distance_import):
        with pytest.warns(RuntimeWarning, match="Python fallback"):
            result = cdo.weighted_inertia_contrib(dist, indices, weights)

    np.testing.assert_allclose(result, np.array([0.5, 0.5]))


def test_cluster_import_path_does_not_require_eager_core_distance_extension():
    from sequenzo.utils import core_distance_operations as cdo

    dist = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    weights = np.array([1.0, 1.0], dtype=np.float64)

    if "sequenzo.clustering.utils.disscenter" in sys.modules:
        del sys.modules["sequenzo.clustering.utils.disscenter"]

    with mock.patch.object(cdo, "_get_c_extension", return_value=None):
        from sequenzo.clustering.utils.disscenter import disscentertrim
        center = disscentertrim(dist, weights=weights)

    assert center.shape == (2,)
    np.testing.assert_allclose(center, np.array([0.25, 0.25]))
