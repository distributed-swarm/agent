# ops/map_tokenize.py
from typing import Dict, Any, List
from . import register_op 

@register_op("map_tokenize")
def map_tokenize_op(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Slices text into fixed-size chunks (default 1KB).
    Enable L2 Pipeline: Lite Agent (CPU) prepares data for Heavy Agent (GPU).
    """

    text = payload.get("text") or payload.get("data", "")
    
    # 1. Configurable Chunk Size (Default 1024 chars ~= 1KB)
    chunk_size = payload.get("chunk_size", 1024)

    if not isinstance(text, str):
        return {
            "ok": False,
            "error": "payload.text must be a string"
        }
    
    if not text:
        return {"ok": True, "tokens": [], "count": 0}

    # 2. The Chunking Logic (Slicing instead of splitting)
    # This creates the "L2 Packets" for your pipeline
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    return {
        "ok": True,
        "tokens": chunks,     # The list of 1KB strings
        "count": len(chunks), # How many jobs this creates
        "total_chars": len(text)
    }
