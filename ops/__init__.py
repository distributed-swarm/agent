from typing import Callable, Dict, Any

# Global registry of ops
OPS_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_op(name: str):
    """
    Decorator to register an op handler function.

        @register_op("map_summarize")
        def handle(task):
            ...
    """
    def decorator(fn: Callable[..., Any]):
        OPS_REGISTRY[name] = fn
        return fn
    return decorator


def list_ops():
    """Return list of registered op names."""
    return sorted(OPS_REGISTRY.keys())


def get_op(name: str) -> Callable[..., Any]:
    """Return the handler function for a given op name."""
    handler = OPS_REGISTRY.get(name)
    if handler is None:
        raise ValueError(f"Unknown op {name!r}. Registered ops: {list(OPS_REGISTRY.keys())}")
    return handler


# Import op modules so their @register_op decorators run.
# These imports MUST come after register_op is defined to avoid circular imports.
from . import map_classify  # noqa: F401
from . import map_summarize  # noqa: F401
from . import map_tokenize  # noqa: F401
EOF
