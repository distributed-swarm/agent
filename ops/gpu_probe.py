# ops/gpu_probe.py
from __future__ import annotations

from typing import Any, Dict, List

from . import register_op


@register_op("gpu_probe")
def handle(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        import torch
    except Exception as e:
        return {
            "cuda_available": False,
            "device_count": 0,
            "devices": [],
            "error": f"torch_import_failed: {e}",
        }

    cuda_ok = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_ok else 0

    devices: List[Dict[str, Any]] = []
    if cuda_ok:
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            cc = f"{props.major}.{props.minor}"
            total_mb = int(props.total_memory // (1024 * 1024))
            devices.append(
                {
                    "id": i,
                    "name": props.name,
                    "compute_capability": cc,
                    "total_vram_mb": total_mb,
                }
            )

    cudnn = None
    try:
        cudnn = getattr(torch.backends, "cudnn", None)
        cudnn = cudnn.version() if cudnn is not None else None
    except Exception:
        cudnn = None

    return {
        "cuda_available": cuda_ok,
        "device_count": device_count,
        "devices": devices,
        "torch_version": getattr(torch, "__version__", None),
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "cudnn": cudnn,
    }
