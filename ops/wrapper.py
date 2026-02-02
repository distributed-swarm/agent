# ops/wrapper.py
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


Json = Dict[str, Any]
# Ops return RAW results (any JSON-ish type). Wrapper owns the envelope.
OpFn = Callable[[Dict[str, Any]], Any]  # most ops accept dict payloads


@dataclass
class OpError(Exception):
    code: str
    message: str
    retryable: bool = False

    def to_dict(self, *, trace: Optional[str] = None) -> Dict[str, Any]:
        d = {"code": self.code, "message": self.message, "retryable": self.retryable}
        if trace:
            d["trace"] = trace
        return d


def _now_ms() -> float:
    return time.time() * 1000.0


def _coerce_payload(payload: Any) -> Any:
    """
    Normalize raw payload forms:
      - None -> {}
      - str/int/float/bool -> {"value": <scalar>}
      - dict -> dict
      - list -> list
    """
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        return payload
    if isinstance(payload, (str, int, float, bool)):
        return {"value": payload}
    # last resort: stringify (keeps wrapper from exploding)
    return {"value": str(payload)}


def _is_map_op(op_name: str) -> bool:
    return op_name.startswith("map_")


def _normalize_map_input(payload: Any) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    For map_* ops:
      - Accept dict -> single item
      - Accept list[dict|scalar] -> batch items (each coerced to dict)
    Returns (is_batch, items_as_dicts)
    """
    if isinstance(payload, list):
        items: List[Dict[str, Any]] = []
        for it in payload:
            it2 = _coerce_payload(it)
            if not isinstance(it2, dict):
                # should never happen after _coerce_payload, but keep it tight
                raise OpError("INVALID_ARGUMENT", "map input items must be object-like", retryable=False)
            items.append(it2)
        return True, items

    payload2 = _coerce_payload(payload)
    if not isinstance(payload2, dict):
        raise OpError("INVALID_ARGUMENT", "payload must be an object", retryable=False)
    return False, [payload2]


def _wrap_success(result: Any, metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {"ok": True, "result": result, "metrics": metrics}


def _wrap_error(err: OpError, metrics: Dict[str, Any], *, trace: Optional[str] = None) -> Dict[str, Any]:
    return {"ok": False, "error": err.to_dict(trace=trace), "metrics": metrics}


def _classify_exception(e: Exception) -> OpError:
    """
    Map raw exceptions to stable error codes.
    Tune this over time; keep it conservative.
    """
    if isinstance(e, OpError):
        return e

    # Common “bad input” paths
    if isinstance(e, (ValueError, TypeError)):
        return OpError("INVALID_ARGUMENT", str(e) or "Invalid argument", retryable=False)

    # Torch/CUDA OOM often contains "out of memory"
    msg = str(e) or e.__class__.__name__
    if "out of memory" in msg.lower():
        return OpError("OOM", msg, retryable=True)

    return OpError("INTERNAL", msg, retryable=True)


def _get_op_spec(op_fn: Callable[..., Any]) -> Dict[str, Any]:
    """
    Optional op metadata.
    Convention: attach a dict to the function object:
        op_fn.OP_SPEC = {"accepts_batch": True, "batch_key": "items", "result_key": "items"}
    """
    spec = getattr(op_fn, "OP_SPEC", None)
    return spec if isinstance(spec, dict) else {}


def run_op(
    op_name: str,
    op_fn: OpFn,
    payload: Any,
    *,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute an op under a strict envelope.

    Contract:
      - Ops return RAW results (any JSON-ish type). They do NOT return {"ok": ...}.
      - Wrapper ALWAYS returns {"ok": bool, "result"/"error": ..., "metrics": ...}

    - For non-map ops: passes a dict payload to op_fn and returns raw output under result.
    - For map_* ops:
        * Accepts dict OR list input.
        * If op advertises OP_SPEC.accepts_batch and input is a list, passes the whole batch
          in a single call (GPU-friendly).
        * Otherwise falls back to scalar-per-item execution (correct, slower).
    """
    t0 = _now_ms()
    meta = meta or {}

    metrics: Dict[str, Any] = {
        "op": op_name,
        "started_ms": t0,
        "meta": meta,
    }

    # Only include traceback when explicitly requested (keeps UI clean by default)
    want_trace = bool(meta.get("debug"))

    try:
        p = _coerce_payload(payload)

        if _is_map_op(op_name):
            is_batch, items = _normalize_map_input(p)

            spec = _get_op_spec(op_fn)
            accepts_batch = bool(spec.get("accepts_batch"))
            batch_key = spec.get("batch_key", "items")
            result_key = spec.get("result_key", "items")

            # Fast path: pass batch through in one call (op handles true batching)
            if is_batch and accepts_batch:
                raw = op_fn({batch_key: items})

                # Compatibility: if an op returns {"items":[...]} we can unwrap
                # to keep map results stable. Raw ops may also return a list directly.
                if isinstance(raw, dict) and result_key in raw and isinstance(raw[result_key], list):
                    result = raw[result_key]
                else:
                    result = raw

                metrics["duration_ms"] = int(_now_ms() - t0)
                metrics["batched"] = True
                metrics["batch_size"] = len(items)
                return _wrap_success(result=result, metrics=metrics)

            # Fallback: scalar-per-item (correct, slower)
            out_items: List[Any] = []
            for item in items:
                out_items.append(op_fn(item))

            metrics["duration_ms"] = int(_now_ms() - t0)
            metrics["batched"] = False
            metrics["batch_size"] = len(items)
            result = out_items if is_batch else out_items[0]
            return _wrap_success(result=result, metrics=metrics)

        # non-map op: single dict payload
        if not isinstance(p, dict):
            raise OpError("INVALID_ARGUMENT", "payload must be an object", retryable=False)

        raw = op_fn(p)
        metrics["duration_ms"] = int(_now_ms() - t0)
        return _wrap_success(result=raw, metrics=metrics)

    except Exception as e:
        err = _classify_exception(e)
        metrics["duration_ms"] = int(_now_ms() - t0)
        trace = traceback.format_exc() if want_trace else None
        return _wrap_error(err=err, metrics=metrics, trace=trace)
