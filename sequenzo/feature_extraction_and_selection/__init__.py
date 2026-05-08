"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 18/03/2026 07:35
@Desc: Feature extraction and selection toolkit for Sequenzo.
"""

import importlib
from typing import Any

_LAZY: dict[str, tuple[str, str]] = {
    "run_feature_extraction_and_selection_pipeline": (
        "sequenzo.feature_extraction_and_selection.feature_extraction_and_selection_pipeline",
        "run_feature_extraction_and_selection_pipeline",
    ),
    "FeatureExtractionAndSelectionConfig": (
        "sequenzo.feature_extraction_and_selection.feature_extraction_and_selection_pipeline",
        "FeatureExtractionAndSelectionConfig",
    ),
    "get_feature_extraction_and_selection_config_preset": (
        "sequenzo.feature_extraction_and_selection.feature_extraction_and_selection_pipeline",
        "get_feature_extraction_and_selection_config_preset",
    ),
    "clustassoc_like_typology_validation": (
        "sequenzo.feature_extraction_and_selection.clustassoc_typology_validation",
        "clustassoc_like_typology_validation",
    ),
    "extract_sequence_features": (
        "sequenzo.feature_extraction_and_selection.feature_extraction",
        "extract_sequence_features",
    ),
    "select_relevant_features": (
        "sequenzo.feature_extraction_and_selection.selection",
        "select_relevant_features",
    ),
    "interpret_selected_features": (
        "sequenzo.feature_extraction_and_selection.interpretation",
        "interpret_selected_features",
    ),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attr = _LAZY[name]
    mod = importlib.import_module(module_path)
    return getattr(mod, attr)


__all__ = list(_LAZY.keys())

