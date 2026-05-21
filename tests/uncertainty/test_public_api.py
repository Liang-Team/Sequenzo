"""
@Author  : Yuqi Liang 梁彧祺
@File    : test_public_api.py
@Time    : 21/05/2026 19:40
@Desc    : Public API aliases for sequenzo.uncertainty.
"""
from sequenzo.uncertainty import (
    get_distance_timing_uncertainty,
    get_timing_perturbed_sequences,
    plot_distance_uncertainty_heatmap,
    print_distance_uncertainty,
    summarize_distance_uncertainty,
)
from sequenzo.uncertainty.seqdist_mcse import seqdist_mcse


def test_primary_aliases_point_to_implementation():
    assert get_timing_perturbed_sequences is not None
    assert get_distance_timing_uncertainty is seqdist_mcse
    assert callable(print_distance_uncertainty)
    assert callable(summarize_distance_uncertainty)
    assert callable(plot_distance_uncertainty_heatmap)


def test_import_from_sequenzo_top_level():
    import sequenzo as sz

    assert sz.get_distance_timing_uncertainty is seqdist_mcse
    assert hasattr(sz, "uncertainty")
    assert sz.uncertainty.get_distance_timing_uncertainty is seqdist_mcse
