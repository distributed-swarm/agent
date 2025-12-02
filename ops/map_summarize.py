from typing import Dict, Any


def map_summarize(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Very simple CPU-only summarizer used for the swarm demo.

    Expects a single text field in the payload:
      - payload["text"]  (preferred)
      - or payload["document"]
      - or payload["body"]

    Returns:
      {
        "ok": True/False,
        "summary": str (when ok=True),
        "error": str (when ok=False),
      }
    """
    text = (
        payload.get("text")
        or payload.get("document")
        or payload.get("body")
    )

    if not text or not isinstance(text, str):
        return {
            "ok": False,
            "error": "No text string provided in payload under 'text'/'document'/'body'.",
        }

    # Dumb but safe "summary": truncate
    max_len = 200
    if len(text) > max_len:
        summary = text[:max_len].rstrip() + "..."
    else:
        summary = text

    return {
        "ok": True,
        "summary": summary,
    }
