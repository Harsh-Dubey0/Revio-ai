import re
from typing import List


def clean_text(text: str) -> str:
    """Normalize review text cleanly while preserving key punctuation/meaning for rules."""
    if not isinstance(text, str) or not text.strip():
        return ""
    cleaned = text.lower().strip()
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def preprocess_for_ml(text: str) -> str:
    """Preprocess text for TF-IDF / ML models."""
    base = clean_text(text)
    if not base:
        return ""
    alnum = re.sub(r"[^\w\s]", " ", base)
    return re.sub(r"\s+", " ", alnum).strip()


def split_sentences(text: str) -> List[str]:
    """Split text into lightweight clauses/sentences for aspect & emotion inspection."""
    if not isinstance(text, str) or not text.strip():
        return []
    parts = re.split(r"[.!?\n;]+", text)
    return [p.strip() for p in parts if p and p.strip()]