"""
@Author  : 李欣怡
@File    : __init__.py
@Time    : 2025/2/26 23:19
@Desc    :
"""
from __future__ import annotations

import importlib
import inspect
import sys
import types
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


class _CallableExportModule(types.ModuleType):
    """Submodule that can also be called like its same-named exported function."""

    _sequenzo_export_attr: str

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return getattr(self, self._sequenzo_export_attr)(*args, **kwargs)


def _is_same_named_submodule(name: str, mod_path: str, attr: str) -> bool:
    return attr == name and mod_path.rsplit(".", 1)[-1] == name


def _callable_submodule(module: types.ModuleType, attr: str) -> types.ModuleType:
    if not isinstance(module, _CallableExportModule):
        module.__class__ = _CallableExportModule
    module._sequenzo_export_attr = attr
    exported = getattr(module, attr)
    module.__signature__ = inspect.signature(exported)
    module.__wrapped__ = exported
    return module


def _load_export(name: str) -> Any:
    mod_path, attr = _LAZY[name]
    module = _loaded.get(mod_path)
    if module is None:
        module = importlib.import_module(mod_path)
        _loaded[mod_path] = module
    if _is_same_named_submodule(name, mod_path, attr):
        return _callable_submodule(module, attr)
    return getattr(module, attr)


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = _load_export(name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


class _LazyExportModule(types.ModuleType):
    """Resolve lazy exports even after Python attaches same-named submodules."""

    def __getattribute__(self, name: str) -> Any:
        value = super().__getattribute__(name)
        lazy = super().__getattribute__("_LAZY")
        if name in lazy:
            mod_path, attr = lazy[name]
            if isinstance(value, types.ModuleType) and getattr(value, "__name__", None) == mod_path:
                if _is_same_named_submodule(name, mod_path, attr):
                    return _callable_submodule(value, attr)
                return getattr(value, attr)
        return value


sys.modules[__name__].__class__ = _LazyExportModule


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
