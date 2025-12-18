# ops/subset_sum.py
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from . import register_op


def _subset_sum_dp(nums: List[int], target: int) -> Dict[str, Any]:
    """
    Pseudo-polynomial DP subset sum solver.
    Returns whether solvable and (optionally) one witness subset.

    This is a *solver*, not just a verifier. Good for demos/smoke tests.
    """
    # dp[t] = index of number used to reach sum t, or None if unreachable.
    # parent[t] = previous sum before adding nums[idx]
    dp: List[Optional[int]] = [None] * (target + 1)
    parent: List[Optional[int]] = [None] * (target + 1)
    dp[0] = -1

    for i, x in enumerate(nums):
        if x < 0:
            raise ValueError("nums must be non-negative for this DP implementation")
        if x > target:
            continue
        # iterate backwards to avoid reuse
        for t in range(target, x - 1, -1):
            if dp[t] is None and dp[t - x] is not None:
                dp[t] = i
                parent[t] = t - x

    sat = dp[target] is not None

    witness: List[int] = []
    if sat:
        t = target
        while t != 0:
            idx = dp[t]
            if idx is None or idx < 0:
                break
            witness.append(nums[idx])
            t_prev = parent[t]
            if t_prev is None:
                break
            t = t_prev
        witness.reverse()

    return {"solvable": sat, "witness": witness}


@register_op("subset_sum")
def subset_sum(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Subset Sum op.
    payload:
      - nums: list[int]
      - target: int

    Returns:
      - solvable: bool
      - witness: list[int] (one solution subset, may be empty if unsolved)
    """
    nums = payload.get("nums")
    target_raw = payload.get("target")

    if not isinstance(nums, list) or any(not isinstance(x, (int, float, str)) for x in nums):
        raise ValueError("payload.nums must be a list of numbers (ints preferred)")

    if target_raw is None:
        raise ValueError("payload.target is required")

    try:
        target = int(target_raw)
    except (TypeError, ValueError):
        raise ValueError("payload.target must be an int")

    # normalize nums to ints
    nums_i: List[int] = []
    for x in nums:
        try:
            nums_i.append(int(x))
        except (TypeError, ValueError):
            raise ValueError("payload.nums must contain only int-coercible values")

    if target < 0:
        raise ValueError("payload.target must be >= 0")

    if any(x < 0 for x in nums_i):
        raise ValueError("payload.nums must be non-negative for this DP implementation")

    # Safety limits: DP is O(n*target)
    if target > 200000:
        raise ValueError("payload.target too large (max 200000)")

    if len(nums_i) > 20000:
        raise ValueError("payload.nums too long (max 20000 items)")

    start = time.time()
    out = _subset_sum_dp(nums_i, target)
    elapsed_ms = (time.time() - start) * 1000.0

    out.update({
        "target": target,
        "n": len(nums_i),
        "compute_time_ms": elapsed_ms,
    })
    return out
