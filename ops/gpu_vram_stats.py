# ops/gpu_vram_stats.py
from __future__ import annotations

from typing import Any, Dict, Optional

from . import register_op


def _try_nvml(device: int) -> Optional[Dict[str, Any]]:
    try:
        import pynvml  # type: ignore
    except Exception:
        return None

    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(device)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        total_mb = int(mem.total // (1024 * 1024))
        free_mb = int(mem.free // (1024 * 1024))
        used_mb = int(mem.used // (1024 * 1024))
        return {"total_mb": total_mb, "free_mb": free_mb, "used_mb": used_mb}
    except Exception:
        return None


@register_op("gpu_vram_stats")
def handle(payload: Dict[str, Any]) -> Dict[str, Any]:
    device = int(payload.get("device", 0))

    try:
        import torch
    except Exception as e:
        return {"device": device, "error": f"torch_import_failed: {e}"}

    nvml_stats = _try_nvml(device)

    torch_alloc = None
    torch_reserved = None
    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
            torch_alloc = int(torch.cuda.memory_allocated(device) // (1024 * 1024))
            torch_reserved = int(torch.cuda.memory_reserved(device) // (1024 * 1024))
    except Exception:
        pass

    out = {"device": device}
    if nvml_stats:
        out.update(nvml_stats)
    else:
        out.update({"total_mb": None, "free_mb": None, "used_mb": None, "note": "nvml_unavailable"})

    out.update(
        {
            "torch_allocated_mb": torch_alloc,
            "torch_reserved_mb": torch_reserved,
        }
    )
    return out

