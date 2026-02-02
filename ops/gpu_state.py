# ops/gpu_state.py
from __future__ import annotations

from typing import Any, Dict

MODEL_CACHE: Dict[str, Any] = {}
TENSOR_CACHE: Dict[str, Any] = {}
