# ops/gpu_embed_image.py
from __future__ import annotations

from typing import Any, Dict, List

from . import register_op
from . import artifacts
from .gpu_state import MODEL_CACHE


@register_op("gpu_embed_image")
def handle(payload: Dict[str, Any]) -> Dict[str, Any]:
    model_key = str(payload.get("model_cache_key", "")).strip()
    if not model_key:
        raise ValueError("model_cache_key is required (use gpu_model_load first)")

    images = payload.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("images must be a non-empty list")

    import torch
    from PIL import Image
    import torchvision.transforms as T

    model = MODEL_CACHE.get(model_key)
    if model is None:
        raise ValueError(f"model not loaded: {model_key}")

    device = next(model.parameters()).device

    # simple, deterministic preproc
    tfm = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    batch_tensors: List[torch.Tensor] = []
    for item in images:
        if not isinstance(item, dict):
            raise ValueError("each images[] item must be an object")

        if "image_ref" in item:
            path = artifacts.get_path(str(item["image_ref"]))
            img = Image.open(path).convert("RGB")
        elif "bytes_b64" in item:
            b = artifacts.b64_to_bytes(str(item["bytes_b64"]))
            import io
            img = Image.open(io.BytesIO(b)).convert("RGB")
        else:
            raise ValueError("image item must contain image_ref or bytes_b64")

        t = tfm(img)  # float32 CHW
        batch_tensors.append(t)

    x = torch.stack(batch_tensors, dim=0).to(device)
    # match model precision
    try:
        p = next(model.parameters())
        if p.dtype == torch.float16:
            x = x.half()
    except StopIteration:
        pass

    with torch.no_grad():
        vec = model(x)  # [N, 512]
        vec = vec.float().cpu()

    vectors = vec.numpy().tolist()
    dim = int(vec.shape[1])

    return {"count": len(vectors), "dim": dim, "vectors": vectors, "model_cache_key": model_key}

