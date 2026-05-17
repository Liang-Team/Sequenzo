"""Shared fixtures for hierarchical tests."""

from __future__ import annotations

import pytest


def pytest_configure(config):
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
    except ImportError:
        pass


@pytest.fixture
def close_mpl_figures():
    """Close any matplotlib figures opened during a test."""
    import matplotlib.pyplot as plt

    before = set(plt.get_fignums())
    yield
    for num in plt.get_fignums():
        if num not in before:
            plt.close(num)
