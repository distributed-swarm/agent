# ops/gpu_detect.py
from __future__ import annotations

from typing import Any, Dict, List

from . import register_op
from . import artifacts
from .gpu_state import MODEL_CACHE


# COCO labels used by torchvision detection models
COCO_LABELS = [
    "__background__",
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","N/A","stop sign","parking meter","bench","bird","cat","dog",
    "horse","sheep","cow","elephant","bear","zebra","giraffe","N/A","backpack","umbrella","N/A","N/A",
    "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","N/A","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","N/A","dining table","N/A","N/A","toilet","N/A","tv","laptop","mouse","remote","keyboard",
    "cell phone","microwave","oven","toaster","sink","refrigerator","N/A","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]


@register_op("gpu_detect")
def handle(payload: Dict[str, Any]) -> Dict[str, Any]:
    model_key = str(payload.get("model_cache_key", "")).strip()
    if not model_key:
        raise ValueError("model_cache_key is required (use gpu_model_load first)")

    image_ref = str(payload.get("image_ref", "")).strip()
    if not image_ref:
        raise ValueError("image_ref is required")

    conf_thres = float(payload.get("conf_thres", 0.25))
    max_det = int(payload.get("max_det", 300))

    import torch
    from PIL import Image
    import torchvision.transforms as T

    model = MODEL_CACHE.get(model_key)
    if model is None:
        raise ValueError(f"model not loaded: {model_key}")

    device = next(model.parameters()).device

    path = artifacts.get_path(image_ref)
    img = Image.open(path).convert("RGB")

    tfm = T.ToTensor()
    x = tfm(img).to(device)  # float32 CHW

    # match fp16 model if loaded that way
    try:
        p = next(model.parameters())
        if p.dtype == torch.float16:
            x = x.half()
    except StopIteration:
        pass

    with torch.no_grad():
        out = model([x])[0]

    # out: boxes [N,4] xyxy, labels [N], scores [N]
    boxes = out["boxes"].detach().float().cpu().numpy()
    labels = out["labels"].detach().cpu().numpy()
    scores = out["scores"].detach().float().cpu().numpy()

    items: List[Dict[str, Any]] = []
    for b, lab, sc in zip(boxes, labels, scores):
        if sc < conf_thres:
            continue
        x1, y1, x2, y2 = [float(v) for v in b.tolist()]
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        lid = int(lab)
        name = COCO_LABELS[lid] if 0 <= lid < len(COCO_LABELS) else str(lid)
        items.append(
            {
                "label": name,
                "label_id": lid,
                "conf": float(sc),
                "bbox": [x1, y1, w, h],
            }
        )
        if len(items) >= max_det:
            break

    return {
        "count": len(items),
        "items": items,
        "model_cache_key": model_key,
        "image_ref": image_ref,
    }

