# ops/map_summarize.py

from . import register_op # <--- STEP 1: IMPORT THE REGISTRATION FUNCTION

@register_op("map_summarize") # <--- STEP 2: USE THE DECORATOR TO REGISTER THE FUNCTION
def handle(payload):
    """
    Map-style summarization handler.

    Expected payload shape:
      {
        "items": [
          { "id": "demo-1", "text": "..." },
          { "id": "demo-2", "text": "..." },
          ...
        ],
        "params": { ... }   # optional
      }
    """
    items = payload.get("items") or []
    if not isinstance(items, list):
        raise ValueError("Expected 'items' to be a list for map_summarize")

    # NOTE: You need actual model loading and processing here!
    # The current code is just a simple string truncation placeholder.

    results = []

    for item in items:
        text = None
        item_id = None

        if isinstance(item, dict):
            item_id = item.get("id")
            # Note: The original code used .get("text") or .get("document") or .get("body")
            # We'll stick to the original logic for now.
            text = item.get("text") or item.get("document") or item.get("body")
        else:
            # Fallback: treat item itself as text
            text = str(item)

        if not isinstance(text, str) or not text.strip():
            raise ValueError("No text string provided in 'text'/'document'/'body'.")

        summary = text.strip()
        if len(summary) > 200:
            summary = summary[:197].rstrip() + "..."

        results.append({
            "id": item_id,
            "summary": summary,
        })

    return {"items": results}
