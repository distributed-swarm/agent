"""
Basic op registry for Base Agent v2.

This file defines a simple registry that other modules can use to
register operation handlers. For now we also eagerly register the
built-in map_classify op.
"""

from typing import Any, Callable, Dict

# op name -> handler(task: dict) -> dict
OPS_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}


def register_op(name: str, handler: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
    """Register a handler callable under a given op name."""
    OPS_REGISTRY[name] = handler


def get_op(name: str):
    """Get a handler by op name, or None if not registered."""
    return OPS_REGISTRY.get(name)


def list_ops():
    """Return a list of registered op names."""
    return list(OPS_REGISTRY.keys())


# ------------------------------------------------------------------
# Built-in ops
# ------------------------------------------------------------------

# Import map_classify so we can register it in the global registry.
try:
    from . import map_classify

    register_op(map_classify.OP_NAME, map_classify.handle)
except Exception:
    # During early wiring / tests we don't want the whole agent to explode
    # just because one op module has an import error.
    pass
