# ops/embed.py
from __future__ import annotations

import time
import torch
from typing import Any, Dict

# Graceful import in case the node is missing the library
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from . import register_op

# Singleton: Only load the model once per agent lifetime
_MODEL = None

def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers library not installed on this agent")
            
        # 1. Auto-detect hardware (CUDA for GPU, CPU for Mac Mini/Pi)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[op:embed] 🧠 Loading BAAI/bge-base-en-v1.5 on {device.upper()}...", flush=True)
        
        # 2. Load the model (this takes time, so we only do it once)
        _MODEL = SentenceTransformer('BAAI/bge-base-en-v1.5', device=device)
        
    return _MODEL


@register_op("embed")
def map_embed(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate vector embeddings for a text chunk."""
    text = payload.get("text")
    
    if not text or not isinstance(text, str):
        raise ValueError("payload.text must be a string")

    # Safety: Prevent massive memory spikes from accidental huge payloads
    if len(text) > 100_000:
        raise ValueError("payload.text too large (max 100k chars)")

    # Load model (if not already loaded)
    model = _get_model()

    start = time.time()
    
    # Run Inference
    # .tolist() is critical: JSON cannot send raw Numpy/Torch arrays
    vector = model.encode(text).tolist()
    
    elapsed_ms = (time.time() - start) * 1000.0

    return {
        "dim": len(vector),      # Proof of work (should be 768)
        "vector": vector,        # The actual data
        "device": str(model.device),
        "compute_time_ms": elapsed_ms,
    }
