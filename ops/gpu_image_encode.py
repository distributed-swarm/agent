# ops/gpu_image_encode.py
from __future__ import annotations

from typing import Any, Dict

from . import register_op
from .gpu_state import TENSOR_CACHE
from . import artifacts


@register_op("gpu_image_encode")
def handle(payload: Dict[str, Any]) -> Dict[str, Any]:
    tensor_ref = str(payload.get("tensor_ref", "")).strip()
    if not tensor_ref:
        raise ValueError("tensor_ref is required")

    fmt = str(payload.get("format", "jpg")).lower()
    quality = int(payload.get("quality", 90))

    import torch
    import io
    from PIL import Image

    t = TENSOR_CACHE.get(tensor_ref)
    if t is None:
        raise ValueError(f"unknown tensor_ref: {tensor_ref}")

    # Expect [1,C,H,W] in 0..1 range
    x = t.detach().float().clamp(0, 1).cpu()
    x = x[0].permute(1, 2, 0).numpy()  # HWC float
    img = Image.fromarray((x * 255).astype("uint8"))

    buf = io.BytesIO()
    save_fmt = "JPEG" if fmt in ("jpg", "jpeg") else "PNG"
    img.save(buf, format=save_fmt, quality=quality if save_fmt == "JPEG" else None)
    art = artifacts.put_bytes(buf.getvalue(), ext=fmt)

    return {"image_ref": art.ref, "format": fmt, "bytes": art.size_bytes}

