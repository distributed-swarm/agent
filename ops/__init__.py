# agent/ops/__init__.py
from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, List

OPS: Dict[str, Callable[[dict], Any]] = {}

# Keep this list authoritative (matches what you want available on this agent image)
_OP_MODULES: List[str] = [
    "echo",
    "map_tokenize",
    "map_summarize",
    "csv_shard",
    "map_classify",
    "risk_accumulate",
    "sat_verify",
    "fibonacci",
    "prime_factor",
    "subset_sum",
    "map_image_gen",
    "artifacts",
    "gpu_state",
    "gpu_probe",
    "gpu_vram_stats",
    "gpu_warmup",
    "gpu_model_load",
    "gpu_embed_image",
    "gpu_detect",
    "gpu_overlay",
    "gpu_image_decode",
    "gpu_image_encode",
    "gpu_video_encode",
]

def _register(op_name: str, fn: Callable[[dict], Any]) -> None:
    if not callable(fn):
        raise TypeError(f"handler for {op_name} is not callable: {fn!r}")
    OPS[op_name] = fn

def _load() -> None:
    # Deterministic: if module has .handle, use it as op name == module basename
    # Special case echo: use echo_op if present, and key it as "echo"
    for base in _OP_MODULES:
        modname = f"ops.{base}"
        print(f"[ops] importing {modname}", flush=True)
        mod = importlib.import_module(modname)

        fn = getattr(mod, "handle", None)
        if fn is None and base == "echo":
            fn = getattr(mod, "echo_op", None)
            if fn is not None:
                _register("echo", fn)
                continue

        if fn is None:
            # Optional: allow modules that export OPS dict
            mod_ops = getattr(mod, "OPS", None)
            if isinstance(mod_ops, dict):
                for k, v in mod_ops.items():
                    if isinstance(k, str) and callable(v):
                        _register(k, v)
                continue

            raise RuntimeError(f"{modname} exports no handler (expected .handle or echo.echo_op or module.OPS)")
        _register(base, fn)

_load()

CAPABILITIES_LIST = sorted(OPS.keys())
