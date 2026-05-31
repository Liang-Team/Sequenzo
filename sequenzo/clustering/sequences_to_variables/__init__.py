"""
@Author  : Yuqi Liang 梁彧祺; Yapeng Wei 卫亚鹏
@File    : __init__.py
@Time    : 10/05/2025 22:11
@Desc    :
Helske-style sequence-to-regression-variable helpers.
"""
from __future__ import annotations

import importlib
import inspect
import types
from typing import Any

from .fanny import (
    FannyResult,
    fanny_membership,
    highest_membership_indices_from_membership,
    medoid_membership_approximation,
)
from .helske_regression_variables import (
    cluster_labels_from_kmedoids_result,
    hard_classification_variables,
    medoid_indices_from_kmedoids_result,
    pseudoclass_regression,
    representativeness_matrix,
    soft_classification_variables,
)
from .helpers import cluster_labels_to_dummies, dummy_column_names, max_distance


class _CallableExportModule(types.ModuleType):
    """Submodule that can also be called like its same-named exported function."""

    _sequenzo_export_attr: str

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return getattr(self, self._sequenzo_export_attr)(*args, **kwargs)


def _callable_submodule(module: types.ModuleType, attr: str) -> types.ModuleType:
    if not isinstance(module, _CallableExportModule):
        module.__class__ = _CallableExportModule
    module._sequenzo_export_attr = attr
    exported = getattr(module, attr)
    module.__signature__ = inspect.signature(exported)
    module.__wrapped__ = exported
    return module


fanny = _callable_submodule(
    importlib.import_module(f"{__name__}.fanny"),
    "fanny",
)

__all__ = [
    "representativeness_matrix",
    "medoid_indices_from_kmedoids_result",
    "cluster_labels_from_kmedoids_result",
    "hard_classification_variables",
    "fanny",
    "FannyResult",
    "fanny_membership",
    "medoid_membership_approximation",
    "highest_membership_indices_from_membership",
    "soft_classification_variables",
    "pseudoclass_regression",
    "max_distance",
    "cluster_labels_to_dummies",
    "dummy_column_names",
]
