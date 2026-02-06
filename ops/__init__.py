# agent/ops/__init__.py
from __future__ import annotations

from typing import Any, Callable, Dict, List

# Global op registry: op_name -> handler(payload_dict) -> Any
OPS: Dict[str, Callable[[Dict[str, Any]], Any]] = {}


def register_op(op_name: str, fn: Callable[[Dict[str, Any]], Any]) -> None:
    """
    Register an op handler.

    Rules:
      - op_name must be a non-empty string
      - fn must be callable and accept a single dict payload argument
      - last registration wins (idempotent overwrite)
    """
    if not isinstance(op_name, str) or not op_name.strip():
        raise ValueError("op_name must be a non-empty string")
    if not callable(fn):
        raise TypeError(f"handler for '{op_name}' is not callable: {fn!r}")

    OPS[op_name] = fn


def list_ops() -> List[str]:
    """Return sorted list of registered op names."""
    return sorted(OPS.keys())
