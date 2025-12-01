"""
Basic op registry for Base Agent v2.

This file defines a simple registry that other modules can use to
register operation handlers. For now we also eagerly register the
built-in ops like map_classify and map_summarize.
"""

from typing import Any, Callable, Dict, Optional

# op name -> handler(task: dict) -> dict
OPS_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}


def register_op(name: str, handler: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None):
    """
    Register an op handler.

    Supports two usage styles:

      1) Direct registration (what older code used):
         register_op("map_classify", handler_fn)

      2) Decorator style (what newer plugins can use):
         @register_op("map_summarize")
         def handle(task: dict) -> dict:
             ...

    In both cases, OPS_REGISTRY[name] will point to the handler.
    """

    # Direct call: register_op("name", handler_fn)
    if handler is not None:
        OPS_REGISTRY[name] = handler
        return handler

    # Decorator usage: @register_op("name")
    def decorator(fn: Callable[[Dict[str, Any]], Dict[str, Any]]):
        OPS_REGISTRY[name] = fn
        return fn

    return decorator


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

    # If it exposes OP_NAME/handle, register that explicitly.
    if hasattr(map_classify, "OP_NAME") and hasattr(map_classify, "handle"):
        register_op(map_classify.OP_NAME, map_classify.handle)
except Exception:
    # During early wiring / tests we don't want the whole agent to explode
    # just because one op module has an import error.
    pass


# Import map_summarize so it is also registered at startup.
try:
    from . import map_summarize

    # If the module defines OP_NAME/handle, register explicitly.
    # If instead it uses @register_op("map_summarize") decorator,
    # importing it is enough for the decorator to run.
    if hasattr(map_summarize, "OP_NAME") and hasattr(map_summarize, "handle"):
        register_op(map_summarize.OP_NAME, map_summarize.handle)
except Exception:
    # Same idea: if summarization wiring is broken, don't kill the whole agent.
    # The op just won't be available until it's fixed.
    pass
