from typing import Any, Dict

from . import register_op


@register_op("echo")
def echo_op(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Trivial diagnostic op.

    It just returns the payload back wrapped in a dict so we can
    test controller ↔ agent plumbing without touching any models.
    """
    return {
        "echo": payload
    }
