# ops/__init__.py

from .map_classify import handle as map_classify_handle
from .map_summarize import handle as map_summarize_handle

OPS_REGISTRY = {
    "map_classify": map_classify_handle,
    "map_summarize": map_summarize_handle,
}


def list_ops():
    return sorted(OPS_REGISTRY.keys())


def get_op(name: str):
    handler = OPS_REGISTRY.get(name)
    if handler is None:
        raise ValueError(f"Unknown op {name!r}")
    return handler
