"""
Basic op registry for Base Agent v2.

Right now this is just a placeholder.
Later we'll register real ops like map_classify, map_tokenize, etc.
"""

OPS_REGISTRY = {}


def register_op(name, handler):
    """Register a handler callable under a given op name."""
    OPS_REGISTRY[name] = handler


def get_op(name):
    """Get a handler by op name, or None if not registered."""
    return OPS_REGISTRY.get(name)


def list_ops():
    """Return a list of registered op names."""
    return list(OPS_REGISTRY.keys())
