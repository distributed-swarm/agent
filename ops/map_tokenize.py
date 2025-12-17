# ops/map_tokenize.py
from typing import Dict, Any, List, Union
from . import register_op


def _chunk_text(text: str, chunk_size: int) -> List[str]:
    if not text:
        return []
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


@register_op("map_tokenize")
def map_tokenize_op(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Slices text into fixed-size chunks (default 1KB).
    Supports either:
      - {"text": "..."} or {"data": "..."}
      - {"items": ["...", "..."], "chunk_size": 1024}
    """

    if payload is None:
        payload = {}

    chunk_size = payload.get("chunk_size", 1024)
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        return {"ok": False, "error": "payload.chunk_size must be a positive integer"}

    # NEW: list input support
    if "items" in payload and payload["items"] is not None:
        items = payload["items"]
        if not isinstance(items, list):
            return {"ok": False, "error": "payload.items must be a list of strings"}

        all_chunks: List[str] = []
        total_chars = 0

        for x in items:
            s = "" if x is None else str(x)
            total_chars += len(s)
            all_chunks.extend(_chunk_text(s, chunk_size))

        return {
            "ok": True,
            "tokens": all_chunks,      # flattened list of chunks
            "count": len(all_chunks),  # number of chunks produced
            "total_chars": total_chars,
            "items_count": len(items),
        }

    # Existing single-text behavior (text/data)
    text = payload.get("text") or payload.get("data", "")
    if not isinstance(text, str):
        return {"ok": False, "error": "payload.text must be a string"}

    chunks = _chunk_text(text, chunk_size)
    return {
        "ok": True,
        "tokens": chunks,
        "count": len(chunks),
        "total_chars": len(text),
    }
