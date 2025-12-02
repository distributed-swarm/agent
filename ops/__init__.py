"""
Basic op registry for Base Agent v2.

This file defines a simple registry that other modules can use to
register operation handlers. For now we also eagerly register the
built-in ops like map_classify and map_summarize.
"""

from typing import Any, Callable, Dict, Optional

# op name -> handler(task: dict) -> dict
OPS_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}


def register_op(
    name: str,
    handler: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
):
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


def _safe_register_builtins():
    """
    Try to register built-ins in a backward-compatible way.

    We support:
      - New style: module has OP_NAME + handle(task: dict)
      - Old style: module has map_classify(task) / map_summarize(task)
    """
    # ---- map_classify ----
    try:
        from . import map_classify as mc  # type: ignore

        if hasattr(mc, "OP_NAME") and hasattr(mc, "handle"):
            register_op(mc.OP_NAME, mc.handle)
        elif hasattr(mc, "map_classify"):
            register_op("map_classify", mc.map_classify)  # old-style
    except Exception:
        # Don't kill the agent just because one op is broken.
        pass

    # ---- map_summarize ----
    try:
        from . import map_summarize as ms  # type: ignore

        if hasattr(ms, "OP_NAME") and hasattr(ms, "handle"):
            register_op(ms.OP_NAME, ms.handle)
        elif hasattr(ms, "map_summarize"):
            register_op("map_summarize", ms.map_summarize)  # old-style
    except Exception:
        # Same idea: summarization wiring issues shouldn't kill the agent.
        pass


_safe_register_builtins()
