# ops/risk_accumulate.py
from __future__ import annotations

import time
from typing import Any, Dict, List

from . import register_op


def _to_float(x: Any) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        return float(x.strip())
    raise ValueError("value must be numeric")


@register_op("risk_accumulate")
def risk_accumulate(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accumulate risk metrics.

    Expected payload shapes:
      A) {"values":[...]} where values are numeric
      B) {"items":[{"risk":...}, ...], "field":"risk"} (field optional, defaults 'risk')

    Returns:
      - count, sum, mean, min, max
    """
    start = time.time()

    values: List[float] = []

    if "values" in payload:
        raw = payload.get("values")
        if not isinstance(raw, list):
            raise ValueError("payload.values must be a list")
        values = [_to_float(v) for v in raw]

    elif "items" in payload:
        items = payload.get("items")
        if not isinstance(items, list):
            raise ValueError("payload.items must be a list")
        field = payload.get("field", "risk")
        for it in items:
            if not isinstance(it, dict):
                raise ValueError("payload.items must contain dict objects")
            if field not in it:
                continue
            values.append(_to_float(it[field]))

    else:
        raise ValueError("payload must include either 'values' or 'items'")

    if not values:
        return {
            "count": 0,
            "sum": 0.0,
            "mean": 0.0,
            "min": None,
            "max": None,
            "compute_time_ms": (time.time() - start) * 1000.0,
        }

    total = sum(values)
    mn = min(values)
    mx = max(values)
    mean = total / len(values)

    return {
        "count": len(values),
        "sum": total,
        "mean": mean,
        "min": mn,
        "max": mx,
        "compute_time_ms": (time.time() - start) * 1000.0,
    }
