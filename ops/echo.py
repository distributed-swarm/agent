# ops/echo.py
from typing import Any, Dict, Optional

from . import register_op


@register_op("echo")
def echo_op(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Trivial diagnostic op.

    Returns payload back so we can test controller ↔ agent plumbing.
    Contract:
      - always returns {"ok": true, "echo": <payload>}
      - if payload is None, returns empty dict
    """
    if payload is None:
        payload = {}

    # If something non-dict slips through, still echo it safely
    if not isinstance(payload, dict):
        return {"ok": True, "echo": payload, "note": "payload_was_not_dict"}

    return {"ok": True, "echo": payload}
