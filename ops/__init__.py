from __future__ import annotations

"""
ops package

Goals:
- Provide a small, stable "instruction registry" for agents.
- Ops modules can register themselves without modifying core agent logic.
- Support BOTH registration styles:
    1) Decorator style (preferred):
        from . import register_op

        @register_op("echo")
        def handle(payload): ...
    2) Direct call style:
        def handle(payload): ...
        register_op("echo", handle)

- Loading is explicit via load_modules(). No heavy side effects required.
"""

import importlib
import os
from typing import Any, Callable, Dict, List, Optional, Set


# op_name -> handler(payload_dict) -> Any
OPS: Dict[str, Callable[[Dict[str, Any]], Any]] = {}

# Deterministic list of modules to load. Add new ops here.
# Convention: module basename is op name, but the module can register any op names it wants.
OPS_MODULES: List[str] = [
    # core
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
    # gpu
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
) -> Any:
    """
    Register an op handler.

    Supports:
      - @register_op("name") decorator usage
      - register_op("name", fn) direct call usage

    Last registration wins (idempotent override), by design.
    """
    if not isinstance(op_name, str) or not op_name.strip():
        raise ValueError("op_name must be a non-empty string")

    def _do_register(f: Callable[[Dict[str, Any]], Any]) -> Callable[[Dict[str, Any]], Any]:
        if not callable(f):
            raise TypeError(f"handler for '{op_name}' is not callable: {f!r}")
        OPS[op_name] = f
        return f

    # Decorator form: @register_op("echo")
    if fn is None:
        return _do_register

    # Direct-call form: register_op("echo", handle)
    return _do_register(fn)


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
    Import each ops module so it can register handlers.

    - modules defaults to OPS_MODULES
    - strict=False: keep going if one module fails (processor behavior)
    - returns list of modules successfully imported
    """
    if modules is None:
        modules = OPS_MODULES

    ok: List[str] = []
    seen: Set[str] = set()

    for name in modules:
        if not isinstance(name, str) or not name.strip():
            continue
        if name in seen:
            continue
        seen.add(name)

        modname = f"{package}.{name}"
        if verbose:
            print(f"[ops] importing {modname}", flush=True)

        try:
            importlib.import_module(modname)
            ok.append(name)
        except Exception as e:
            # Keep booting; just don't advertise/register what failed.
            msg = f"[ops] FAILED import {modname}: {type(e).__name__}: {e}"
            if strict:
                raise RuntimeError(msg) from e
            print(msg, flush=True)

    return ok


# Optional: allow autoload when explicitly requested. If not rub magic rock three times. 
if os.getenv("OPS_AUTOLOAD", "").strip() in ("1", "true", "yes", "on"):
    load_modules()
