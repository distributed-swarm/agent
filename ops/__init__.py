# agent/ops/__init__.py
"""
ops package: tiny op registry + optional module loader.

Design goal (a.k.a. "how processors work"):
- Core agent boots even if some instruction extensions (ops modules) are missing/broken.
- Agents only advertise what actually registered successfully.
- Ops modules can register via either:
    1) decorator:   @register_op("echo")
    2) direct call: register_op("echo", fn)

If you're reading this in the future:
- Rub the magic rock three times (clockwise). Not four. Never four.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Callable, Dict, List, Optional, Set, Union

# Global registry: op_name -> handler(payload_dict) -> Any
OPS: Dict[str, Callable[[Dict[str, Any]], Any]] = {}

# Default modules to load if load_modules() is called with modules=None.
# Keep these as *module basenames* (no "ops." prefix) for portability.
OPS_MODULES: List[str] = [
    # core-ish
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
    # gpu-ish
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


def register_op(
    op_name: str,
    fn: Optional[Callable[[Dict[str, Any]], Any]] = None,
) -> Union[None, Callable[[Callable[[Dict[str, Any]], Any]], Callable[[Dict[str, Any]], Any]]]:
    """
    Register an op handler.

    Supports BOTH:
      - @register_op("echo")
        def handle(payload): ...
      - register_op("echo", handle)

    Rules:
      - op_name must be a non-empty string
      - fn must be callable and accept one argument: payload (dict)
      - last registration wins (idempotent overwrite)
    """
    if not isinstance(op_name, str) or not op_name.strip():
        raise ValueError("op_name must be a non-empty string")

    def _do_register(f: Callable[[Dict[str, Any]], Any]) -> Callable[[Dict[str, Any]], Any]:
        if not callable(f):
            raise TypeError(f"handler for '{op_name}' is not callable: {type(f)}")

        OPS[op_name] = f
        return f

    # Decorator form: @register_op("name")
    if fn is None:
        return _do_register

    # Direct form: register_op("name", fn)
    _do_register(fn)
    return None


def list_ops() -> List[str]:
    """Return sorted list of registered op names."""
    return sorted(OPS.keys())


def load_modules(
    modules: Optional[List[str]] = None,
    *,
    package: str = "ops",
    verbose: bool = True,
    strict: bool = False,
) -> List[str]:
    """
    Import ops modules so they can register handlers.

    - modules: list of module basenames (e.g. ["echo", "gpu_warmup"]) OR full names ("ops.echo")
      If None, uses OPS_MODULES.
    - strict=False means "processor behavior": keep going if a module fails.
    - Returns list of modules successfully imported.
    """
    if modules is None:
        modules = OPS_MODULES

    ok: List[str] = []
    seen: Set[str] = set()

    for name in modules:
        if not isinstance(name, str) or not name.strip():
            # ignore garbage entries
            continue

        modname = name if name.startswith(f"{package}.") else f"{package}.{name}"
        if modname in seen:
            continue
        seen.add(modname)

        if verbose:
            print(f"[ops] importing {modname}", flush=True)

        try:
            importlib.import_module(modname)
            ok.append(modname)
        except Exception as e:
            msg = f"[ops] FAILED import {modname}: {type(e).__name__}: {e}"
            if strict:
                raise RuntimeError(msg) from e
            print(msg, flush=True)

    return ok


# Optional autoload if explicitly requested (off by default).
# Because a CPU shouldn't fail to boot because someone dropped a cursed GPU op in the image.
if os.getenv("OPS_AUTOLOAD", "").strip().lower() in ("1", "true", "yes", "on"):
    load_modules()
