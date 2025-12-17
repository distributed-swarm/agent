import os
import math
import subprocess
from typing import Dict, Any, List

try:
    import psutil
except ImportError:
    psutil = None


def _detect_cpu() -> Dict[str, Any]:
    """
    Basic CPU sizing using psutil if available, otherwise os.cpu_count().
    """
    if psutil is not None:
        try:
            total_cores = psutil.cpu_count(logical=True) or 1
        except Exception:
            total_cores = os.cpu_count() or 1
    else:
        total_cores = os.cpu_count() or 1

    # Reserve some cores for the system / Docker overhead
    reserved_cores = min(4, max(1, total_cores // 4))
    usable_cores = max(1, total_cores - reserved_cores)

    # Rough heuristic: up to 1 worker per usable core
    max_cpu_workers = max(1, usable_cores)
    min_cpu_workers = 1

    return {
        "total_cores": int(total_cores),
        "reserved_cores": int(reserved_cores),
        "usable_cores": int(usable_cores),
        "min_cpu_workers": int(min_cpu_workers),
        "max_cpu_workers": int(max_cpu_workers),
    }


def _nvidia_visible_devices_allows_gpu() -> bool:
    """
    NVIDIA container runtime convention:
      - NVIDIA_VISIBLE_DEVICES=none  -> no GPUs exposed
      - empty / unset               -> depends, but assume allowed
      - "all" or list of ids        -> allowed
    """
    v = os.getenv("NVIDIA_VISIBLE_DEVICES")
    if v is None:
        return True
    v = str(v).strip().lower()
    if v in ("", "void"):
        # Treat empty/void as "not explicitly blocked"
        return True
    if v == "none":
        return False
    return True


def _parse_nvidia_smi() -> List[Dict[str, Any]]:
    """
    Use `nvidia-smi` to detect GPUs and their memory.

    Returns a list of devices:
      [
        {"index": 0, "name": "...", "total_memory_bytes": 123},
        ...
      ]

    If anything fails, returns [].
    """
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except Exception:
        return []

    devices: List[Dict[str, Any]] = []
    for idx, line in enumerate(out.splitlines()):
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        name, mem_str = parts
        try:
            mem_mib = float(mem_str)
        except ValueError:
            continue
        total_bytes = int(mem_mib * 1024 * 1024)
        devices.append(
            {
                "index": idx,
                "name": name,
                "total_memory_bytes": total_bytes,
            }
        )
    return devices


def _detect_gpu() -> Dict[str, Any]:
    """
    GPU sizing based on nvidia-smi only.

    Reports:
      - gpu_present
      - gpu_count
      - vram_gb (largest device VRAM in GiB, rounded)
      - devices[]
      - max_gpu_workers

    If no GPUs are visible (or explicitly blocked), gpu_present=False.
    """
    if not _nvidia_visible_devices_allows_gpu():
        return {
            "gpu_present": False,
            "gpu_count": 0,
            "vram_gb": None,
            "devices": [],
            "max_gpu_workers": 0,
        }

    devices = _parse_nvidia_smi()
    if not devices:
        return {
            "gpu_present": False,
            "gpu_count": 0,
            "vram_gb": None,
            "devices": [],
            "max_gpu_workers": 0,
        }

    gpu_count = len(devices)

    max_bytes = max(int(d.get("total_memory_bytes", 0) or 0) for d in devices)
    vram_gb = (max_bytes / float(1024 ** 3)) if max_bytes > 0 else None

    # Simple heuristic: 1 worker per visible GPU device
    max_gpu_workers = gpu_count

    return {
        "gpu_present": True,
        "gpu_count": int(gpu_count),
        "vram_gb": (round(vram_gb, 2) if vram_gb is not None else None),
        "devices": devices,
        "max_gpu_workers": int(max_gpu_workers),
    }


def build_worker_profile() -> Dict[str, Any]:
    """
    Combined CPU + GPU + worker sizing.

    Shape is what the controller & UI already expect:

      {
        "cpu": {...},
        "gpu": {...},
        "workers": {"max_total_workers": int, "current_workers": 0}
      }
    """
    cpu_info = _detect_cpu()
    gpu_info = _detect_gpu()

    cpu_max = cpu_info.get("max_cpu_workers", 1)
    if isinstance(cpu_max, float):
        cpu_max = int(math.floor(cpu_max))
    cpu_max = max(1, int(cpu_max))

    gpu_max = int(gpu_info.get("max_gpu_workers", 0) or 0)

    # Total worker limit: allow CPU + GPU workers (controller can still cap/reroute).
    max_total_workers = max(1, cpu_max + gpu_max)

    return {
        "cpu": cpu_info,
        "gpu": gpu_info,
        "workers": {
            "max_total_workers": int(max_total_workers),
            "current_workers": 0,
        },
    }
