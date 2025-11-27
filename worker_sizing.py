import os
import math
import subprocess
from typing import Dict, Any, List, Optional

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


def _parse_nvidia_smi() -> List[Dict[str, Any]]:
    """
    Use `nvidia-smi` to detect GPUs and their memory.

    Returns a list of devices:
      [
        {
          "index": 0,
          "name": "NVIDIA GeForce RTX 3060",
          "total_memory_bytes": 12486246400
        },
        ...
      ]

    If anything fails, returns [].
    """
    try:
        # Query: name + total memory (MiB), no units, no header
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
        # Example line: "NVIDIA GeForce RTX 3060, 12288"
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

    We don't care if PyTorch has CUDA or not here; this is purely for:
      - worker_profile.gpu_present
      - gpu_count
      - vram_gb
      - devices[...]

    If no GPUs are visible to nvidia-smi, we report gpu_present = False.
    """
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
    # Use the largest single-device memory as "vram_gb" for routing thresholds
    max_bytes = max(d.get("total_memory_bytes", 0) for d in devices) or 0
    vram_gb = max_bytes / float(1024 ** 3) if max_bytes > 0 else 0.0

    # Simple heuristic: assume up to 1 worker per device by default
    max_gpu_workers = gpu_count

    return {
        "gpu_present": True,
        "gpu_count": gpu_count,
        "vram_gb": round(vram_gb, 2),
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
        "workers": {
          "max_total_workers": int,
          "current_workers": 0
        }
      }
    """
    cpu_info = _detect_cpu()
    gpu_info = _detect_gpu()

    # Total worker limit: use CPU max workers as upper bound for now.
    max_total_workers = cpu_info.get("max_cpu_workers", 1)
    if isinstance(max_total_workers, float):
        max_total_workers = int(math.floor(max_total_workers))

    if max_total_workers < 1:
        max_total_workers = 1

    return {
        "cpu": cpu_info,
        "gpu": gpu_info,
        "workers": {
            "max_total_workers": int(max_total_workers),
            "current_workers": 0,
        },
    }
