from __future__ import annotations

import base64
import io
import time
from typing import Any, Dict, List, Optional, Tuple

from . import register_op

try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None  # pillow optional

# Import OpError from wrapper so we can return stable error codes.
# This is safe because wrapper.py does not import ops modules.
from .wrapper import OpError  # type: ignore


_BACKEND = "placeholder_fake_png"


def _parse_size(size: str) -> Tuple[int, int]:
    """
    Parse WxH like '256x256'. Clamp to sane bounds.
    """
    w, h = 256, 256
    s = (size or "").strip().lower()
    if "x" in s:
        a, b = s.split("x", 1)
        try:
            w = int(a)
            h = int(b)
        except Exception:
            raise OpError("INVALID_ARGUMENT", "size must be like '256x256'", retryable=False)

    # Clamp to keep placeholder safe
    w = max(1, min(2048, w))
    h = max(1, min(2048, h))
    return w, h


def _fake_image_bytes(prompt: str, size: str) -> bytes:
    """
    Placeholder generator for environments without a real image model.
    Produces a tiny PNG with deterministic content.
    Replace later with your actual backend.
    """
    if Image is None:
        # minimal 1x1 PNG bytes fallback
        return base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
        )

    w, h = _parse_size(size)
    # Note: we ignore prompt in placeholder mode, but keep it in outputs for contract stability
    img = Image.new("RGB", (w, h), (255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _one(prompt: str, size: str) -> Dict[str, Any]:
    png = _fake_image_bytes(prompt, size)
    b64 = base64.b64encode(png).decode("ascii")
    return {
        "prompt": prompt,
        "size": size,
        "image_base64_png": b64,
        "bytes": len(png),
    }


def _require_nonempty_str(val: Any, field: str) -> str:
    if not isinstance(val, str) or not val.strip():
        raise OpError("INVALID_ARGUMENT", f"{field} must be a non-empty string", retryable=False)
    return val.strip()


def _require_int(val: Any, field: str) -> int:
    try:
        return int(val)
    except Exception:
        raise OpError("INVALID_ARGUMENT", f"{field} must be an int", retryable=False)


@register_op("map_image_gen")
def map_image_gen(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Image generation mapping op.

    Placeholder backend: returns a valid PNG in base64.
    Swap _fake_image_bytes() with a real generator later.

    Supported payloads:

    Single:
      {
        "prompt": "a cat",
        "n": 1,                 # optional, default 1, [1..8]
        "size": "256x256"       # optional, default "256x256"
      }

    Batch:
      {
        "items": [
          {"prompt":"a cat", "size":"256x256"},
          {"prompt":"a dog"}              # size inherited from top-level size/default
        ],
        "size":"256x256"                 # optional default for items
      }

    Returns (op-level envelope):
      {
        "ok": true,
        "result": {"count": N, "items": [...], "note": "..."},
        "metrics": {"compute_time_ms": ..., "backend": "..."}
      }
    """
    start = time.time()

    if not isinstance(payload, dict):
        raise OpError("INVALID_ARGUMENT", "payload must be an object", retryable=False)

    default_size = str(payload.get("size", "256x256")).strip() or "256x256"

    # Batch mode
    if "items" in payload:
        items = payload.get("items")
        if not isinstance(items, list):
            raise OpError("INVALID_ARGUMENT", "payload.items must be a list", retryable=False)

        out: List[Dict[str, Any]] = []
        for idx, it in enumerate(items):
            if not isinstance(it, dict):
                raise OpError("INVALID_ARGUMENT", f"payload.items[{idx}] must be an object", retryable=False)

            prompt = _require_nonempty_str(it.get("prompt"), f"payload.items[{idx}].prompt")
            it_size = str(it.get("size", default_size)).strip() or default_size

            out.append(_one(prompt, it_size))

        compute_ms = (time.time() - start) * 1000.0
        return {
            "ok": True,
            "result": {
                "count": len(out),
                "items": out,
                "note": "placeholder generator; replace backend when ready",
            },
            "metrics": {
                "compute_time_ms": compute_ms,
                "backend": _BACKEND,
                "batched": True,
                "batch_size": len(out),
            },
        }

    # Single mode
    prompt = _require_nonempty_str(payload.get("prompt"), "payload.prompt")

    n_int = _require_int(payload.get("n", 1), "payload.n")
    if n_int < 1 or n_int > 8:
        raise OpError("INVALID_ARGUMENT", "payload.n must be between 1 and 8", retryable=False)

    out = [_one(prompt, default_size) for _ in range(n_int)]
    compute_ms = (time.time() - start) * 1000.0
    return {
        "ok": True,
        "result": {
            "count": len(out),
            "items": out,
            "note": "placeholder generator; replace backend when ready",
        },
        "metrics": {
            "compute_time_ms": compute_ms,
            "backend": _BACKEND,
            "batched": False,
            "batch_size": len(out),
        },
    }


# Tell the wrapper we support batch payloads in one call.
# Even though placeholder mode doesn't speed up, the contract is correct and future-proof.
map_image_gen.OP_SPEC = {"accepts_batch": True, "batch_key": "items", "result_key": "items"}
