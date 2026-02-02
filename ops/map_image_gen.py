from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Tuple

from . import register_op
from .wrapper import OpError  # structured errors -> wrapper produces envelope

try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None  # pillow optional

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

    w = max(1, min(2048, w))
    h = max(1, min(2048, h))
    return w, h


def _fake_image_bytes(prompt: str, size: str) -> bytes:
    """
    Placeholder generator for environments without a real image model.
    Produces a valid PNG.
    Replace later with your actual backend.
    """
    if Image is None:
        # minimal 1x1 PNG bytes fallback
        return base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
        )

    w, h = _parse_size(size)
    img = Image.new("RGB", (w, h), (255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _require_dict(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise OpError("INVALID_ARGUMENT", "payload must be an object", retryable=False)
    return payload


def _require_nonempty_str(val: Any, field: str) -> str:
    if not isinstance(val, str) or not val.strip():
        raise OpError("INVALID_ARGUMENT", f"{field} must be a non-empty string", retryable=False)
    return val.strip()


def _require_int(val: Any, field: str) -> int:
    try:
        return int(val)
    except Exception:
        raise OpError("INVALID_ARGUMENT", f"{field} must be an int", retryable=False)


def _one_image_raw(prompt: str, size: str) -> Dict[str, Any]:
    png = _fake_image_bytes(prompt, size)
    b64 = base64.b64encode(png).decode("ascii")
    return {
        "prompt": prompt,
        "size": size,
        "mime": "image/png",
        "image_base64": b64,
        "bytes": len(png),
        "backend": _BACKEND,
    }


@register_op("map_image_gen")
def map_image_gen(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    RAW-only op.

    Semantics:
      - Scalar payload generates N images for a single prompt.
      - Batched payload maps each item -> (possibly N images) for that item.

    Scalar payload:
      {
        "prompt": "a cat",
        "n": 1,                 # optional, default 1, [1..8]
        "size": "256x256"       # optional, default "256x256"
      }

    Returns RAW dict:
      {
        "count": N,
        "items": [ {image...}, ... ],
        "note": "..."
      }

    Batch payload:
      {
        "items": [
          {"prompt":"a cat", "n": 1, "size":"256x256"},
          {"prompt":"a dog"}              # size inherited from top-level/default
        ],
        "size":"256x256"                 # optional default for items
      }

    Returns RAW dict:
      {"items": [ <RAW scalar dict>, <RAW scalar dict>, ... ]}
      (wrapper unwraps to list result)
    """
    payload = _require_dict(payload)

    default_size = str(payload.get("size", "256x256")).strip() or "256x256"

    # Batch mode: items -> list of RAW scalar outputs
    if isinstance(payload.get("items"), list):
        items = payload["items"]
        out_list: List[Dict[str, Any]] = []

        for idx, it in enumerate(items):
            if not isinstance(it, dict):
                raise OpError("INVALID_ARGUMENT", f"payload.items[{idx}] must be an object", retryable=False)

            prompt = _require_nonempty_str(it.get("prompt"), f"payload.items[{idx}].prompt")
            size = str(it.get("size", default_size)).strip() or default_size

            n = _require_int(it.get("n", 1), f"payload.items[{idx}].n")
            if n < 1 or n > 8:
                raise OpError(
                    "INVALID_ARGUMENT",
                    f"payload.items[{idx}].n must be between 1 and 8",
                    retryable=False,
                )

            imgs = [_one_image_raw(prompt, size) for _ in range(n)]
            out_list.append(
                {
                    "count": len(imgs),
                    "items": imgs,
                    "note": "placeholder generator; replace backend when ready",
                }
            )

        return {"items": out_list}

    # Scalar mode
    prompt = _require_nonempty_str(payload.get("prompt"), "payload.prompt")
    n = _require_int(payload.get("n", 1), "payload.n")
    if n < 1 or n > 8:
        raise OpError("INVALID_ARGUMENT", "payload.n must be between 1 and 8", retryable=False)

    imgs = [_one_image_raw(prompt, default_size) for _ in range(n)]
    return {
        "count": len(imgs),
        "items": imgs,
        "note": "placeholder generator; replace backend when ready",
    }


# Wrapper fast-path: if input is a list, wrapper passes {"items":[...]} through in one call.
map_image_gen.OP_SPEC = {"accepts_batch": True, "batch_key": "items", "result_key": "items"}
