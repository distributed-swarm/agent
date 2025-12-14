from typing import Callable, Dict, Any

# Global registry of ops
OPS_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_op(name: str):
    """Decorator to register an op handler function."""
    def decorator(fn: Callable[..., Any]):
        OPS_REGISTRY[name] = fn
        return fn
    return decorator


def list_ops():
    """Return sorted list of registered op names."""
    return sorted(OPS_REGISTRY.keys())


def get_op(name: str) -> Callable[..., Any]:
    """Return the handler function for a given op name."""
    try:
        return OPS_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown op {name!r}. Registered ops: {sorted(OPS_REGISTRY.keys())}"
        )


# Import op modules so their @register_op decorators run.

from . import echo          # noqa: F401
from . import map_classify  # noqa: F401
from . import map_summarize  # noqa: F401

# Optional op, don't die if it's missing
try:
    from . import map_tokenize  # noqa: F401
except ImportError:
    pass

# New CSV shard op
from . import csv_shard  # noqa: F401
