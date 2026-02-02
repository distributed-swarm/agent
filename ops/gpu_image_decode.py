# ops/gpu_image_decode.py
from __future__ import annotations

from typing import Any, Dict

from . import register_op
from . import artifacts
from .gpu_state import TENSOR_CACHE


@register_op("gpu_image_decode")
def handle(payload: Dict[str, Any]) -> Dict[str, Any]:
    image_ref = str(payload.get("image_ref", "")).strip()
    if not image_ref:
        raise ValueError("image_ref is required")

    device = int(payload.get("device", 0))
    dtype = str(payload.get("dtype", "fp16")).lower()

    import torch
    from PIL import Image
    import torchvision.transforms as T

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available on this agent")

    path = artifacts.get_path(image_ref)
    img = Image.open(path).convert("RGB")

    t = T.ToTensor()(img)  # float32 CHW
    t = t.unsqueeze(0)     # [1,C,H,W]

    torch.cuda.set_device(device)
    t = t.to(f"cuda:{device}")

    if dtype in ("fp16", "half"):
        t = t.half()

    # simple ref
    key = f"tensor://{abs(hash((image_ref, device, dtype)))}"
    TENSOR_CACHE[key] = t

    _, c, h, w = t.shape
    return {"tensor_ref": key, "shape": [int(h), int(w), int(c)], "dtype": dtype, "device": device}

