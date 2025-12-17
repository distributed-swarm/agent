# ops/__init__.py
from __future__ import annotations

from typing import Callable, Dict, Any, Optional

# Global registry of ops
OPS_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_op(name: str):
    """
    Decorator to register an op handler function.

    New-logic expectation:
      - ops should be importable (so decorators run)
      - op names should be unique
    """
    def decorator(fn: Callable[..., Any]):
        prev = OPS_REGISTRY.get(name)
        if prev is not None and prev is not fn:
            # Keep last one (explicit override), but make it obvious in logs.
            try:
                prev_name = getattr(prev, "__name__", str(prev))
                fn_name = getattr(fn, "__name__", str(fn))
            except Exception:
                prev_name, fn_name = "<?>", "<?>"
            print(f"[ops] WARNING: op '{name}' re-registered ({prev_name} -> {fn_name})", flush=True)

        OPS_REGISTRY[name] = fn
        return fn

    return decorator


def list_ops():
    """Return sorted list of registered op names."""
    return sorted(OPS_REGISTRY.keys())


def get_op(name: str) -> Callable[..., Any]:
    """
    Return the handler function for a given op name.
    Raises ValueError (kept for compatibility with existing agent code paths).
    """
    fn = OPS_REGISTRY.get(name)
    if fn is None:
        raise ValueError(f"Unknown op {name!r}. Registered ops: {list_ops()}")
    return fn


def try_get_op(name: str) -> Optional[Callable[..., Any]]:
    """New helper: return op or None (no exception)."""
    return OPS_REGISTRY.get(name)


# Import op modules so their @register_op decorators run.
# Core ops (should always exist in the image)
from . import echo            # noqa: F401
from . import map_tokenize    # noqa: F401
from . import map_summarize   # noqa: F401
from . import csv_shard       # noqa: F401
from . import map_classify    # noqa: F401
from . import risk_accumulate # noqa: F401

# If you later add truly optional ops, do it like this:
# try:
#     from . import some_optional_op  # noqa: F401
# except Exception as e:
#     print(f"[ops] optional op some_optional_op not loaded: {type(e).__name__}: {e}", flush=True)
