import os
from pathlib import Path
import sys
import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import settings
from src.nlp.aspect_extractor import extract_aspects
from src.nlp.emotion_analyzer import analyze_emotion
from src.nlp.preprocessor import preprocess_for_ml


def preprocess_text(text):
    """Backward-compatible wrapper delegating to src.nlp.preprocessor."""
    return preprocess_for_ml(text)


class ReviewPredictor:
    def __init__(self, model_path=None):
        self.model_path = model_path or settings.MODEL_PATH
        self.pipeline = None
        self.load_model()

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Trained model not found at {self.model_path}. Please run training pipeline first."
            )
        try:
            self.pipeline = joblib.load(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")

    @staticmethod
    def detect_emotion(text: str, rating: int = 3, sentiment: str = "Neutral") -> str:
        """Backward-compatible wrapper delegating to src.nlp.emotion_analyzer."""
        return analyze_emotion(text, rating=rating, sentiment=sentiment)

    def predict_single(self, review_text: str, rating: int = 3, category: str = "General") -> dict:
        """Run inference on a single review. Returns sentiment, confidence, aspects, priority, emotion."""
        if not self.pipeline:
            self.load_model()

        clean_text = preprocess_text(review_text)
        if not clean_text:
            return {
                "sentiment": "Neutral",
                "confidence": 0.0,
                "aspect_labels": "General Experience",
                "priority": "Low",
                "emotion": "Neutral",
            }

        positive_keywords = ["excellent", "incredible", "amazing", "great", "delightful", "fantastic", "awesome"]
        negative_keywords = ["terrible", "bad", "worst", "awful", "horrible", "poor", "disappointed"]
        lower_text = str(review_text).lower()

        rule_based = False
        if any(kw in lower_text for kw in positive_keywords):
            sentiment = "Positive"
            rule_based = True
        elif any(kw in lower_text for kw in negative_keywords):
            sentiment = "Negative"
            rule_based = True
        else:
            sentiment = self.pipeline.predict([clean_text])[0]

        # Determine confidence
        if rule_based:
            confidence = 1.0
        else:
            confidence = 1.0
            if hasattr(self.pipeline, "predict_proba"):
                probs = self.pipeline.predict_proba([clean_text])[0]
                classes = list(self.pipeline.classes_)
                if sentiment in classes:
                    idx = classes.index(sentiment)
                    confidence = float(probs[idx])

        aspect_labels = extract_aspects(review_text, category=category)
        priority = self._calculate_priority(sentiment, rating)
        emotion = self.detect_emotion(review_text, rating=rating, sentiment=sentiment)

        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "aspect_labels": aspect_labels,
            "priority": priority,
            "emotion": emotion,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run inference on a DataFrame of reviews."""
        if not self.pipeline:
            self.load_model()

        results = []
        for _, row in df.iterrows():
            text = str(row.get("review_text", ""))
            rating = int(row.get("rating", 3))
            category = str(row.get("product_category", "General"))

            pred = self.predict_single(text, rating, category)
            results.append({
                "review_id": row.get("review_id"),
                "sentiment": pred["sentiment"],
                "confidence": pred["confidence"],
                "aspect_labels": pred["aspect_labels"],
                "priority": pred["priority"],
                "emotion": pred["emotion"],
            })

        return pd.DataFrame(results)

    def _calculate_priority(self, sentiment: str, rating: int) -> str:
        if sentiment == "Negative" or rating <= 2:
            return "High"
        elif sentiment == "Mixed" or rating == 3:
            return "Medium"
        else:
            return "Low"