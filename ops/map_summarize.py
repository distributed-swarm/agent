from typing import Dict, Any
from . import register_op

# New-style registration: OP_NAME + handle()
OP_NAME = "map_summarize"

# Global model instance (loaded once when module imports)
_summarizer = None


def _get_summarizer():
    """Lazy-load the summarization model"""
    global _summarizer
    if _summarizer is None:
        try:
            from transformers import pipeline
            import torch
            
            # Use BART for summarization - good quality and reasonably fast
            # Check if GPU is available
            device = 0 if torch.cuda.is_available() else -1
            
            _summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=device
            )
            print(f"[map_summarize] Loaded BART model on device: {'GPU' if device == 0 else 'CPU'}")
        except Exception as e:
            print(f"[map_summarize] ERROR loading model: {e}")
            raise
    
    return _summarizer


def _chunk_text(text: str, max_chunk_size: int = 1000) -> list:
    """
    Split text into chunks that fit within model's token limit.
    Uses simple sentence-based splitting to avoid cutting mid-sentence.
    """
    # Rough estimate: 1 token ≈ 4 characters
    # BART can handle ~1024 tokens, so we use ~4000 chars per chunk with overlap
    
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    sentences = text.split('. ')
    
    current_chunk = ""
    for sentence in sentences:
        # Add sentence to current chunk
        test_chunk = current_chunk + sentence + ". "
        
        if len(test_chunk) > max_chunk_size and current_chunk:
            # Current chunk is full, save it and start new one
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
        else:
            current_chunk = test_chunk
    
    # Add remaining text
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


@register_op(OP_NAME)
def handle(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Production-ready summarizer using BART model.

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

    # Get the model (lazy-loaded)
    try:
        summarizer = _get_summarizer()
    except Exception as e:
        return {
            "ok": False,
            "error": f"Failed to load model: {str(e)}",
        }

    try:
        # Handle very short texts
        if len(text) < 100:
            return {
                "ok": True,
                "summary": text,
            }
        
        # For long documents, we'll chunk and summarize each chunk,
        # then combine the summaries
        chunks = _chunk_text(text, max_chunk_size=4000)
        
        if len(chunks) == 1:
            # Single chunk - direct summarization
            result = summarizer(
                text,
                max_length=150,
                min_length=30,
                do_sample=False,
                truncation=True
            )
            summary = result[0]['summary_text']
        else:
            # Multiple chunks - summarize each then combine
            chunk_summaries = []
            
            for i, chunk in enumerate(chunks):
                try:
                    result = summarizer(
                        chunk,
                        max_length=100,
                        min_length=20,
                        do_sample=False,
                        truncation=True
                    )
                    chunk_summaries.append(result[0]['summary_text'])
                except Exception as e:
                    # If a chunk fails, log it but continue
                    print(f"[map_summarize] WARNING: Chunk {i} failed: {e}")
                    continue
            
            if not chunk_summaries:
                return {
                    "ok": False,
                    "error": "All chunks failed to summarize",
                }
            
            # Combine chunk summaries
            combined = " ".join(chunk_summaries)
            
            # If combined is still long, summarize it again
            if len(combined) > 500:
                result = summarizer(
                    combined,
                    max_length=150,
                    min_length=30,
                    do_sample=False,
                    truncation=True
                )
                summary = result[0]['summary_text']
            else:
                summary = combined

        return {
            "ok": True,
            "summary": summary,
        }

    except Exception as e:
        return {
            "ok": False,
            "error": f"Summarization failed: {str(e)}",
        }
