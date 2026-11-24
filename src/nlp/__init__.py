from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

__all__ = [
    "clean_text",
    "preprocess_for_ml",
    "split_sentences",
    "analyze_emotion",
    "extract_aspects",
    "extract_aspects_list",
    "cluster_review_topics",
    "build_business_opportunities",
]


def __getattr__(name: str) -> Any:
    if name in {"clean_text", "preprocess_for_ml", "split_sentences"}:
        from src.nlp import preprocessor
        return getattr(preprocessor, name)
    if name == "analyze_emotion":
        from src.nlp.emotion_analyzer import analyze_emotion
        return analyze_emotion
    if name in {"extract_aspects", "extract_aspects_list"}:
        from src.nlp import aspect_extractor
        return getattr(aspect_extractor, name)
    if name == "cluster_review_topics":
        from src.nlp.topic_clusterer import cluster_review_topics
        return cluster_review_topics
    if name == "build_business_opportunities":
        from src.nlp.opportunity_engine import build_business_opportunities
        return build_business_opportunities
    raise AttributeError(f"module 'src.nlp' has no attribute {name!r}")


if __name__ == "__main__":
    print("src.nlp package initialized successfully.")