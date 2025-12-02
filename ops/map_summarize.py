from typing import Dict, Any
from . import register_op

# New-style registration: OP_NAME + handle()
OP_NAME = "map_summarize"


@register_op(OP_NAME)
def handle(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Very simple CPU-only summarizer used for the swarm demo.

    Expects a single text field in the task:
      - task["text"]      (preferred)
      - or task["document"]
      - or task["body"]

    Returns:
      {
        "ok": True/False,
        "summary": str (when ok=True),
        "error": str (when ok=False),
      }
    """
    text = (
        task.get("text")
        or task.get("document")
        or task.get("body")
    )

    if not text or not isinstance(text, str):
        return {
            "ok": False,
            "error": "No text string provided in 'text'/'document'/'body'.",
        }

    # Dumb but safe "summary": truncate to ~200 chars
    max_len = 200
    if len(text) > max_len:
        summary = text[:max_len].rstrip() + "..."
    else:
        summary = text

    return {
        "ok": True,
        "summary": summary,
    }
