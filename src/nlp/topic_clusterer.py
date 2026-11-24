from pathlib import Path
import sys
from typing import Any, Dict, List
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.nlp.preprocessor import preprocess_for_ml


def cluster_review_topics(
    df: pd.DataFrame,
    text_col: str = "review_text",
    n_clusters: int = 4,
    top_terms: int = 3,
) -> List[Dict[str, Any]]:
    """Group review texts into interpretable topic clusters with top keywords and counts."""
    if df is None or df.empty or text_col not in df.columns:
        return []

    texts = [preprocess_for_ml(t) for t in df[text_col].fillna("").tolist()]
    non_empty = [t for t in texts if t]
    if not non_empty:
        return []

    k = min(max(1, n_clusters), len(set(non_empty)), len(non_empty))
    if k == 1:
        vec = TfidfVectorizer(stop_words="english", max_features=200)
        mat = vec.fit_transform(non_empty)
        terms = vec.get_feature_names_out()
        scores = mat.sum(axis=0).A1
        top_idx = scores.argsort()[::-1][:top_terms]
        label = ", ".join([str(terms[i]) for i in top_idx]) or "General Feedback"
        return [{"cluster_id": 0, "topic": label.title(), "count": len(non_empty), "keywords": list(label.split(", "))}]

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=400)
    tfidf_matrix = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(tfidf_matrix)

    clusters: List[Dict[str, Any]] = []
    for cid in range(k):
        mask = labels == cid
        count = int(mask.sum())
        if count == 0:
            continue
        center = km.cluster_centers_[cid]
        top_indices = center.argsort()[::-1][:top_terms]
        keywords = [str(terms[idx]) for idx in top_indices if idx < len(terms)]
        topic_name = " / ".join([kw.title() for kw in keywords]) if keywords else f"Theme {cid + 1}"
        clusters.append({
            "cluster_id": cid,
            "topic": topic_name,
            "count": count,
            "keywords": keywords,
        })

    clusters.sort(key=lambda x: x["count"], reverse=True)
    return clusters


if __name__ == "__main__":
    sample_df = pd.DataFrame({
        "review_text": [
            "Fast delivery and great product quality",
            "Late delivery and broken packaging",
            "Customer support helped quickly",
            "Too expensive for the quality",
        ]
    })
    print(cluster_review_topics(sample_df, n_clusters=2))