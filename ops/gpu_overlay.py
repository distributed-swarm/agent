# ops/gpu_overlay.py
from __future__ import annotations

from typing import Any, Dict, List

from . import register_op
from . import artifacts


@register_op("gpu_overlay")
def handle(payload: Dict[str, Any]) -> Dict[str, Any]:
    image_ref = str(payload.get("image_ref", "")).strip()
    if not image_ref:
        raise ValueError("image_ref is required")

    boxes = payload.get("boxes", [])
    if boxes is None:
        boxes = []
    if not isinstance(boxes, list):
        raise ValueError("boxes must be a list")

    out_format = str(payload.get("out_format", "jpg")).lower()

    from PIL import Image, ImageDraw, ImageFont
    import io

    path = artifacts.get_path(image_ref)
    img = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Font optional; PIL default is fine
    for b in boxes:
        if not isinstance(b, dict):
            continue
        bbox = b.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        x, y, w, h = [float(v) for v in bbox]
        x2, y2 = x + w, y + h
        draw.rectangle([x, y, x2, y2], width=3)
        label = str(b.get("label", ""))
        conf = b.get("conf", None)
        txt = label if conf is None else f"{label} {float(conf):.2f}"
        if txt.strip():
            draw.text((x + 3, y + 3), txt)

    buf = io.BytesIO()
    save_fmt = "JPEG" if out_format in ("jpg", "jpeg") else "PNG"
    img.save(buf, format=save_fmt, quality=90 if save_fmt == "JPEG" else None)
    art = artifacts.put_bytes(buf.getvalue(), ext=out_format)

    return {"image_ref": art.ref, "format": out_format}

