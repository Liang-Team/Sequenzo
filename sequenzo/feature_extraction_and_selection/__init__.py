"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 18/03/2026 07:35
@Desc    :
    Feature Extraction & Selection toolkit for Sequenzo.
"""

from .feature_extraction_and_selection_pipeline import (
    run_feature_extraction_and_selection_pipeline,
    FeatureExtractionAndSelectionConfig,
    run_fes_pipeline,
    FESConfig,
)
from .clustassoc_typology_validation import clustassoc_like_typology_validation

__all__ = [
    "run_feature_extraction_and_selection_pipeline",
    "FeatureExtractionAndSelectionConfig",
    "clustassoc_like_typology_validation",
    # Backward-compatible aliases
    "run_fes_pipeline",
    "FESConfig",
]

