import math
import time
from typing import Dict, Any, List, Tuple

# In-memory state: per entity
_STATE: Dict[str, Dict[str, float]] = {}

def _exp_decay(dt: float, tau: float) -> float:
    if tau <= 0:
        return 0.0
    return math.exp(-dt / tau)

def run(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    payload:
      {
        "now_ts": 1765997000.0,   # optional; default=time.time()
        "tau_sec": 3600.0,
        "weights": {"food": 1.0, "air": 0.3, "contact": 0.6},
        "thresholds": {"warn": 10.0, "act": 20.0},
        "events": [
          {"entity_id":"jack", "mode":"air", "dose": 0.8, "ts": 1765996990.0},
          ...
        ]
      }
    """
    now_ts = float(payload.get("now_ts") or time.time())
    tau = float(payload.get("tau_sec", 3600.0))
    weights = payload.get("weights") or {}
    thr = payload.get("thresholds") or {"warn": 10.0, "act": 20.0}
    events: List[Dict[str, Any]] = payload.get("events") or []

    # Group events per entity; sum dose per mode
    grouped: Dict[str, Dict[str, float]] = {}
    latest_ts: Dict[str, float] = {}

    for e in events:
        ent = str(e["entity_id"])
        mode = str(e.get("mode", "unknown"))
        dose = float(e.get("dose", 0.0))
        ts = float(e.get("ts", now_ts))

        grouped.setdefault(ent, {})
        grouped[ent][mode] = grouped[ent].get(mode, 0.0) + dose
        latest_ts[ent] = max(latest_ts.get(ent, 0.0), ts)

    results = []
    for ent, doses_by_mode in grouped.items():
        st = _STATE.get(ent) or {"B": 0.0, "last_ts": latest_ts.get(ent, now_ts)}
        last_ts = float(st.get("last_ts", now_ts))
        dt = max(0.0, now_ts - last_ts)

        B = float(st.get("B", 0.0))
        B *= _exp_decay(dt, tau)

        # Add new doses
        for mode, dose in doses_by_mode.items():
            w = float(weights.get(mode, 0.0))
            B += w * dose

        # Save state
        _STATE[ent] = {"B": B, "last_ts": now_ts}

        # Flags
        warn = B >= float(thr.get("warn", 10.0))
        act = B >= float(thr.get("act", 20.0))

        results.append({
            "entity_id": ent,
            "B": B,
            "warn": warn,
            "act": act,
            "now_ts": now_ts,
            "dt": dt,
        })

    return {"ok": True, "results": results}
