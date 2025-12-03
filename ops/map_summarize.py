from typing import Dict, Any
import logging
from . import register_op

OP_NAME = "map_summarize"

log = logging.getLogger(__name__)

_summarizer = None


def _get_summarizer():
    """Lazy-load the summarization model."""
    global _summarizer
    if _summarizer is not None:
        return _summarizer

    try:
        from transformers import pipeline
        import torch

        device = 0 if torch.cuda.is_available() else -1

        _summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=device,
        )
        log.info(
            "[map_summarize] Loaded BART model on device: %s",
            "GPU" if device == 0 else "CPU",
        )
    except Exception:
        log.exception("[map_summarize] ERROR loading BART model")
        raise

    return _summarizer


def _chunk_text(text: str, max_chunk_size: int = 1000) -> list:
    """
    Split text into chunks that fit within model's token limit.
    Simple sentence-based splitting.
    """
    if len(text) <= max_chunk_size:
        return [text]

    chunks = []
    sentences = text.split(". ")

    current_chunk = ""
    for sentence in sentences:
        test_chunk = current_chunk + sentence + ". "

        if len(test_chunk) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
        else:
            current_chunk = test_chunk

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def _truncate_fallback(text: str, max_len: int = 200) -> str:
    """Cheap but safe fallback."""
    if len(text) > max_len:
        return text[:max_len].rstrip() + "..."
    return text


@register_op(OP_NAME)
def handle(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Production-ready summarizer using BART model.

    Expects:
      - task["text"] (preferred)
      - or task["document"]
      - or task["body"]
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

    text = text.strip()
    if not text:
        return {"ok": False, "error": "Empty text provided."}

    # Short text: just echo back
    if len(text) < 100:
        return {
            "ok": True,
            "summary": text,
        }

    try:
        summarizer = _get_summarizer()
    except Exception as e:
        # Fallback instead of hard failing
        log.warning("[map_summarize] Falling back to truncate: %s", e)
        return {
            "ok": True,
            "summary": _truncate_fallback(text),
        }

    try:
        chunks = _chunk_text(text, max_chunk_size=4000)

        if len(chunks) == 1:
            result = summarizer(
                text,
                max_length=150,
                min_length=30,
                do_sample=False,
                truncation=True,
            )
            summary = result[0]["summary_text"]
        else:
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                try:
                    result = summarizer(
                        chunk,
                        max_length=100,
                        min_length=20,
                        do_sample=False,
                        truncation=True,
                    )
                    chunk_summaries.append(result[0]["summary_text"])
                except Exception as e:
                    log.warning("[map_summarize] Chunk %d failed: %s", i, e)
                    continue

            if not chunk_summaries:
                return {
                    "ok": False,
                    "error": "All chunks failed to summarize",
                }

            combined = " ".join(chunk_summaries)

            if len(combined) > 500:
                result = summarizer(
                    combined,
                    max_length=150,
                    min_length=30,
                    do_sample=False,
                    truncation=True,
                )
                summary = result[0]["summary_text"]
            else:
                summary = combined

        return {
            "ok": True,
            "summary": summary,
        }

    except Exception as e:
        log.exception("[map_summarize] Summarization failed")
        return {
            "ok": False,
            "error": f"Summarization failed: {str(e)}",
        }
