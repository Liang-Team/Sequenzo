"""
@Author  : 李欣怡
@File    : __init__.py
@Time    : 2025/2/26 23:19
@Desc    :
"""
from __future__ import annotations

import importlib
from typing import Any


_LAZY: dict[str, tuple[str, str]] = {
    "get_distance_matrix": (
        "sequenzo.dissimilarity_measures.get_distance_matrix",
        "get_distance_matrix",
    ),
    "get_substitution_cost_matrix": (
        "sequenzo.dissimilarity_measures.get_substitution_cost_matrix",
        "get_substitution_cost_matrix",
    ),
    "get_LCP_length_for_2_seq": (
        "sequenzo.dissimilarity_measures.utils.get_LCP_length_for_2_seq",
        "get_LCP_length_for_2_seq",
    ),
    "get_sm_trate_substitution_cost_matrix": (
        "sequenzo.dissimilarity_measures.utils.get_sm_trate_substitution_cost_matrix",
        "get_sm_trate_substitution_cost_matrix",
    ),
    "seqconc": ("sequenzo.dissimilarity_measures.utils.seqconc", "seqconc"),
    "seqdss": ("sequenzo.dissimilarity_measures.utils.seqdss", "seqdss"),
    "seqdur": ("sequenzo.dissimilarity_measures.utils.seqdur", "seqdur"),
    "seqlength": ("sequenzo.dissimilarity_measures.utils.seqlength", "seqlength"),
}

_loaded: dict[str, Any] = {}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    mod_path, attr = _LAZY[name]
    if mod_path not in _loaded:
        _loaded[mod_path] = importlib.import_module(mod_path)
    value = getattr(_loaded[mod_path], attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


def _import_c_code():
    """Lazily import the c_code module to avoid circular dependencies during installation."""
    try:
        return importlib.import_module("sequenzo.dissimilarity_measures.c_code")
    except ImportError:
        print(
            "Warning: The C++ extension (c_code) could not be imported. "
            "Please ensure the extension module is compiled correctly."
        )
        return None


__all__ = [
    "get_distance_matrix",
    "get_substitution_cost_matrix",
    "get_LCP_length_for_2_seq",
]
