# ops/gpu_model_load.py
from __future__ import annotations

from typing import Any, Dict

from . import register_op
from .gpu_state import MODEL_CACHE


@register_op("gpu_model_load")
def handle(payload: Dict[str, Any]) -> Dict[str, Any]:
    model_id = str(payload.get("model_id", "")).strip()
    if not model_id:
        raise ValueError("model_id is required")

    device = int(payload.get("device", 0))
    precision = str(payload.get("precision", "fp16")).lower()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available on this agent")

    torch.cuda.set_device(device)
    dev = torch.device(f"cuda:{device}")

    cache_key = f"{model_id}@{precision}@cuda:{device}"
    if cache_key in MODEL_CACHE:
        return {"model_id": model_id, "loaded": True, "device": device, "precision": precision, "cache_key": cache_key}

    # Load model
    model = None

    if model_id == "resnet18_embed":
        import torchvision.models as models
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Use penultimate layer as embedding: remove final classifier
        m.fc = torch.nn.Identity()
        model = m.eval()

    elif model_id == "fasterrcnn_resnet50_fpn":
        import torchvision.models.detection as det
        m = det.fasterrcnn_resnet50_fpn(weights=det.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        model = m.eval()

    else:
        raise ValueError(
            f"Unknown model_id={model_id!r}. Supported: "
            f"'resnet18_embed', 'fasterrcnn_resnet50_fpn'"
        )

    if precision in ("fp16", "half"):
        model = model.half()

    model = model.to(dev)
    MODEL_CACHE[cache_key] = model

    return {"model_id": model_id, "loaded": True, "device": device, "precision": precision, "cache_key": cache_key}

