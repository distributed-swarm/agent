# ops/map_image_gen.py
from __future__ import annotations

import base64
import io
import time
from typing import Any, Dict, List, Optional

from . import register_op

try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None  # pillow optional


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

    w, h = 256, 256
    try:
        if "x" in size:
            a, b = size.lower().split("x", 1)
            w = max(1, min(2048, int(a)))
            h = max(1, min(2048, int(b)))
    except Exception:
        pass

    img = Image.new("RGB", (w, h), (255, 255, 255))
    # keep it simple; no text rendering dependency
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


@register_op("map_image_gen")
def map_image_gen(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Image generation mapping op.

    NOTE: This file ships with a placeholder generator to keep the op usable
    immediately. Swap _fake_image_bytes() with your actual image backend.

    payload:
      - prompt: str (single)
      - n: int (optional, default 1)
      - size: str (optional, default '256x256')
    OR
      - items: list[{"prompt": str, "size": str?}] (batch)
    """
    start = time.time()

    size = str(payload.get("size", "256x256"))
    n = payload.get("n", 1)

    if "items" in payload:
        items = payload.get("items")
        if not isinstance(items, list):
            raise ValueError("payload.items must be a list")
        out: List[Dict[str, Any]] = []
        for it in items:
            if not isinstance(it, dict):
                raise ValueError("payload.items elements must be dicts")
            prompt = it.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("each item must include non-empty prompt")
            it_size = str(it.get("size", size))
            out.append(_one(prompt, it_size))
        return {
            "count": len(out),
            "items": out,
            "compute_time_ms": (time.time() - start) * 1000.0,
            "note": "placeholder generator; replace backend when ready",
        }

    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("payload.prompt must be a non-empty string")

    try:
        n_int = int(n)
    except Exception:
        raise ValueError("payload.n must be an int")

    if n_int < 1 or n_int > 8:
        raise ValueError("payload.n must be between 1 and 8")

    out = [_one(prompt, size) for _ in range(n_int)]
    return {
        "count": len(out),
        "items": out,
        "compute_time_ms": (time.time() - start) * 1000.0,
        "note": "placeholder generator; replace backend when ready",
    }
