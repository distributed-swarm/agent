import os 
from math import floor


def _to_int(value, default=None):
    """
    Safe int converter for env variables.
    """
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def calculate_cpu_workers(total_cores: int | None = None) -> dict:
    """
    Decide how many CPU workers this agent should use.

    Rules:
    - Detect cores if not provided.
    - Reserve ~25% of cores for the OS / other stuff (min 1).
    - Use the rest, minus 1, for workers (min 1).
    - Allow overrides via:
        AGENT_MAX_CPU_WORKERS
        AGENT_MIN_CPU_WORKERS
    """
    if total_cores is None:
        total_cores = os.cpu_count() or 1

    max_env = _to_int(os.getenv("AGENT_MAX_CPU_WORKERS"))
    min_env = _to_int(os.getenv("AGENT_MIN_CPU_WORKERS"))

    reserved_cores = max(1, floor(total_cores * 0.25))
    usable_cores = max(1, total_cores - reserved_cores)

    # automatic max (before env overrides)
    auto_max = max(1, usable_cores - 1)

    # start from automatic values
    max_cpu_workers = auto_max
    min_cpu_workers = 1

    # apply env overrides if present
    if max_env is not None:
        max_cpu_workers = max(1, min(auto_max, max_env))

    if min_env is not None:
        min_cpu_workers = max(1, min_env)

    # ensure consistency
    if max_cpu_workers < min_cpu_workers:
        max_cpu_workers = min_cpu_workers

    return {
        "total_cores": total_cores,
        "reserved_cores": reserved_cores,
        "usable_cores": usable_cores,
        "min_cpu_workers": min_cpu_workers,
        "max_cpu_workers": max_cpu_workers,
    }


def calculate_gpu_workers(
    gpu_present: bool = False,
    gpu_count: int = 0,
    vram_gb: int | None = None,
) -> dict:
    """
    Decide how many GPU workers this agent should use.

    Rules:
    - If no GPU: 0 workers.
    - Base: 1 worker per GPU.
    - If vram_gb >= 16, allow up to 2 workers per GPU.
    - Allow override via AGENT_MAX_GPU_WORKERS.
    """
    max_gpu_workers = 0

    if gpu_present and gpu_count > 0:
        # base
        max_gpu_workers = gpu_count

        # vram-based bump (simple rule for now)
        if vram_gb is not None and vram_gb >= 16:
            max_gpu_workers = gpu_count * 2

    max_env = _to_int(os.getenv("AGENT_MAX_GPU_WORKERS"))
    if max_env is not None:
        max_gpu_workers = max(0, max_env)

    return {
        "gpu_present": bool(gpu_present),
        "gpu_count": int(gpu_count),
        "vram_gb": vram_gb,
        "max_gpu_workers": max_gpu_workers,
    }


def build_worker_profile(
    total_cores: int | None = None,
    gpu_present: bool = False,
    gpu_count: int = 0,
    vram_gb: int | None = None,
) -> dict:
    """
    Build the combined worker profile for this agent.

    This is what we will expose to the controller/UI.
    """
    cpu_info = calculate_cpu_workers(total_cores=total_cores)
    gpu_info = calculate_gpu_workers(
        gpu_present=gpu_present,
        gpu_count=gpu_count,
        vram_gb=vram_gb,
    )

    max_total_workers = cpu_info["max_cpu_workers"] + gpu_info["max_gpu_workers"]

    return {
        "cpu": cpu_info,
        "gpu": gpu_info,
        "workers": {
            "max_total_workers": max_total_workers,
            # this will be updated by the agent's worker manager later
            "current_workers": 0,
        },
    }
