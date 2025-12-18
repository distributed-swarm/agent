# ops/fibonacci.py
from __future__ import annotations

import time
from typing import Any, Dict

from . import register_op


def _fib_iter(n: int) -> int:
    # Fast + safe for moderately sized n
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


@register_op("fibonacci")
def map_fibonacci(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate Fibonacci number at position n."""
    n_raw = payload.get("n", 30)
    try:
        n = int(n_raw)
    except (TypeError, ValueError):
        raise ValueError("payload.n must be an int")

    if n < 0:
        raise ValueError("payload.n must be >= 0")

    # Prevent accidental “n=999999” nuking an agent
    if n > 50000:
        raise ValueError("payload.n too large (max 50000)")

    start = time.time()
    result = _fib_iter(n)
    elapsed_ms = (time.time() - start) * 1000.0

    return {
        "n": n,
        "result": result,
        "compute_time_ms": elapsed_ms,
    }
