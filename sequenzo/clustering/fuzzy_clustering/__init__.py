"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 08/05/2025 20:59
@Desc    :
Fuzzy clustering utilities aligned with Studer (2018) and WeightedCluster.

- ``get_fuzzy_clusters``: FANNY for Studer (2018)-style fuzzy sequence clustering;
  optional ``wfcmdd`` for WeightedCluster-style distance-based fuzzy C-medoids
- ``most_typical_members``: highest-membership sequence per cluster (Studer 4.2.1)
- ``membership_summary``: descriptive statistics of membership strengths
- ``crispness``: partition crispness scores
- ``fuzzy_sequence_plot``: membership-weighted index and distribution plots
- ``dirichlet_regression`` / ``beta_regression``: regressions on membership
"""
from .fuzzy_helpers import (
    FuzzyClusterResult,
    get_fuzzy_clusters,
    membership_summary,
    most_typical_members,
)
from .fuzzy_regression import (
    DirichletRegData,
    DirichletRegResult,
    beta_regression,
    dirichlet_regression,
    prepare_dirichlet_data,
)
from .fuzzy_sequence_plots import fuzzy_sequence_plot, fuzzy_sequence_plot_single
from .wfcmdd_fuzzy_clustering import WfcmddResult, crispness, wfcmdd

__all__ = [
    "get_fuzzy_clusters",
    "FuzzyClusterResult",
    "membership_summary",
    "most_typical_members",
    "wfcmdd",
    "crispness",
    "WfcmddResult",
    "fuzzy_sequence_plot",
    "fuzzy_sequence_plot_single",
    "prepare_dirichlet_data",
    "DirichletRegData",
    "DirichletRegResult",
    "dirichlet_regression",
    "beta_regression",
]
