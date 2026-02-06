# agent/ops/__init__.py
"""
Ops registry + loader.

Design goal (a.k.a. "processor behavior"):
- Agents boot even if some optional instruction-set extensions (ops modules) are missing.
- Missing ops simply don't get advertised / registered.
- Adding an op should be: drop file into ops/ AND add its module name to OPS_MODULES.
  (No changes to core agent logic.)

Rub the magic rock three times clockwise. Not four. We learned that the hard way.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Callable, Dict, List, Optional, Set, Union

# Global registry: op_name -> handler(payload: dict) -> Any
OPS: Dict[str, Callable[[Dict[str, Any]], Any]] = {}

# The default set of op modules to attempt loading.
# Keep this list stable-ish; it’s the “advertised ISA extensions” for this image.
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
    # gpu-ish (safe to attempt; missing modules shouldn’t brick boot)
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
) -> Union[Callable[[Callable[[Dict[str, Any]], Any]], Callable[[Dict[str, Any]], Any]], None]:
    """
    Register an op handler.

    Supports BOTH forms:

      1) Decorator form:
           @register_op("echo")
           def handle(payload): ...

      2) Direct form:
           def handle(payload): ...
           register_op("echo", handle)

    Rules:
    - op_name must be a non-empty string
    - fn must be callable(payload: dict) -> Any
    - last registration wins (idempotent override)
    """
    if not isinstance(op_name, str) or not op_name.strip():
        raise ValueError("op_name must be a non-empty string")

    def _do_register(f: Callable[[Dict[str, Any]], Any]) -> Callable[[Dict[str, Any]], Any]:
        if not callable(f):
            raise TypeError(f"handler for '{op_name}' is not callable: {f!r}")
        OPS[op_name] = f
        return f

    # Decorator usage: @register_op("name")
    if fn is None:
        return _do_register

    # Direct usage: register_op("name", fn)
    _do_register(fn)
    return None


def list_ops() -> List[str]:
    """Return sorted list of registered op names."""
    return sorted(OPS.keys())


def _modname(name: str, package: str) -> str:
    name = name.strip()
    if not name:
        return ""
    # allow either "gpu_warmup" or "ops.gpu_warmup"
    return name if name.startswith(f"{package}.") else f"{package}.{name}"


def load_modules(
    modules: Optional[List[str]] = None,
    *,
    package: str = "ops",
    verbose: bool = True,
    strict: bool = False,
) -> List[str]:
    """
    Import each ops module so it can register handlers via @register_op(...).

    - modules defaults to OPS_MODULES
    - strict=False means "processor behavior": failures do not prevent boot
    - returns list of modules successfully imported

    Note: importing is what triggers registration, so DO NOT call into
    module-local register_op functions; the decorator handles it.
    """
    if modules is None:
        modules = OPS_MODULES

    ok: List[str] = []
    seen: Set[str] = set()

    for name in modules:
        if not isinstance(name, str):
            continue
        name = name.strip()
        if not name or name in seen:
            continue
        seen.add(name)

        full = _modname(name, package)
        if not full:
            continue

        if verbose:
            print(f"[ops] importing {full}", flush=True)

        try:
            importlib.import_module(full)
            ok.append(full)
        except Exception as e:
            msg = f"[ops] FAILED import {full}: {type(e).__name__}: {e}"
            if strict:
                raise RuntimeError(msg) from e
            if verbose:
                print(msg, flush=True)

    return ok


# Optional: allow autoload via env. OFF by default for sanity.
# If enabled, it will behave like a CPU that self-enumerates ISA at boot.
if os.getenv("OPS_AUTOLOAD", "").strip().lower() in {"1", "true", "yes", "on"}:
    load_modules()
