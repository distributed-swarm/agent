# ops/gpu_warmup.py
from __future__ import annotations

from typing import Any, Dict

from . import register_op


@register_op("gpu_warmup")
def handle(payload: Dict[str, Any]) -> Dict[str, Any]:
    device = int(payload.get("device", 0))
    iters = int(payload.get("iters", 10))
    size = int(payload.get("size", 1024))

    import time
    try:
        import torch
    except Exception as e:
        raise RuntimeError(f"torch_import_failed: {e}")

    if not torch.cuda.is_available():
        return {"device": device, "kind": "torch", "iters": iters, "elapsed_ms": 0, "cuda_available": False}

    torch.cuda.set_device(device)

    a = torch.randn((size, size), device="cuda")
    b = torch.randn((size, size), device="cuda")

    t0 = time.time()
    for _ in range(iters):
        c = a @ b
        _ = c.sum().item()
    torch.cuda.synchronize()
    elapsed_ms = int((time.time() - t0) * 1000)

    return {"device": device, "kind": "torch", "iters": iters, "size": size, "elapsed_ms": elapsed_ms}

