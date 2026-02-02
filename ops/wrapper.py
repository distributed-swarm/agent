# ops/wrapper.py
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


Json = Dict[str, Any]
OpFn = Callable[[Dict[str, Any]], Dict[str, Any]]  # existing ops mostly accept dict payloads


@dataclass
class OpError(Exception):
    code: str
    message: str
    retryable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "message": self.message, "retryable": self.retryable}


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


def _wrap_error(err: OpError, metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {"ok": False, "error": err.to_dict(), "metrics": metrics}


def _classify_exception(e: Exception) -> OpError:
    """
    Map raw exceptions to stable error codes.
    Tune this over time; keep it conservative.
    """
    if isinstance(e, OpError):
        return e

    # Common “bad input” paths (map_image_gen currently raises ValueError) :contentReference[oaicite:2]{index=2}
    if isinstance(e, (ValueError, TypeError)):
        return OpError("INVALID_ARGUMENT", str(e) or "Invalid argument", retryable=False)

    # Torch/CUDA OOM often contains "out of memory"
    msg = str(e) or e.__class__.__name__
    if "out of memory" in msg.lower():
        return OpError("OOM", msg, retryable=True)

    return OpError("INTERNAL", msg, retryable=True)


def run_op(
    op_name: str,
    op_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    payload: Any,
    *,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute an op under a strict envelope.

    - For non-map ops: passes a dict payload to op_fn and returns op output under result.
    - For map_* ops: supports dict OR list input; calls op_fn per-item; returns list results.
      (If you later build true batching in the op itself, you can special-case here.)
    """
    t0 = _now_ms()
    meta = meta or {}

    metrics: Dict[str, Any] = {
        "op": op_name,
        "started_ms": t0,
        "meta": meta,
    }

    try:
        p = _coerce_payload(payload)

        if _is_map_op(op_name):
            is_batch, items = _normalize_map_input(p)

            out_items: List[Any] = []
            for item in items:
                raw = op_fn(item)

                # Allow legacy “ok” shape to pass through as result, but normalize
                # map_classify returns {"ok": True, "label": ..., ...} :contentReference[oaicite:3]{index=3}
                # map_image_gen returns {"count":..., "items":..., ...} and no ok :contentReference[oaicite:4]{index=4}
                out_items.append(raw)

            metrics["duration_ms"] = _now_ms() - t0
            result = out_items if is_batch else out_items[0]
            return _wrap_success(result=result, metrics=metrics)

        # non-map op: single dict payload
        if not isinstance(p, dict):
            raise OpError("INVALID_ARGUMENT", "payload must be an object", retryable=False)

        raw = op_fn(p)
        metrics["duration_ms"] = _now_ms() - t0
        return _wrap_success(result=raw, metrics=metrics)

    except Exception as e:
        err = _classify_exception(e)
        metrics["duration_ms"] = _now_ms() - t0

        # Optional: include traceback only when debugging, never by default.
        # You can gate this behind an env var if you want.
        # metrics["traceback"] = traceback.format_exc(limit=20)

        return _wrap_error(err=err, metrics=metrics)
