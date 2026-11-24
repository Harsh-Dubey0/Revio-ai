import pytest
import os
import pandas as pd
from src.inference.predictor import ReviewPredictor

def test_predictor_initialization():
    predictor = ReviewPredictor()
    assert predictor.pipeline is not None

def test_predictor_single():
    predictor = ReviewPredictor()
    result = predictor.predict_single("The service was excellent and food arrived hot!", rating=5, category="Food delivery")

    assert result["sentiment"] == "Positive"
    assert result["confidence"] > 0.5
    assert "Service Quality" in result["aspect_labels"] or "Delivery & Shipping" in result["aspect_labels"]
    assert result["priority"] == "Low"
    assert result["emotion"] in ("Happy", "Satisfied")

def test_predictor_negative():
    predictor = ReviewPredictor()
    result = predictor.predict_single("Horrible quality, it broke immediately. Waste of money.", rating=1, category="Electronics")

    assert result["sentiment"] == "Negative"
    assert result["priority"] == "High"
    assert result["emotion"] in ("Angry", "Frustrated")

def test_emotion_detection_categories():
    predictor = ReviewPredictor()
    # Angry
    assert predictor.detect_emotion("Horrible scam, the item never arrived and seller was rude!", rating=1, sentiment="Negative") == "Angry"
    # Frustrated
    assert predictor.detect_emotion("Very disappointed with the slow shipping and damaged box.", rating=2, sentiment="Negative") == "Frustrated"
    # Happy
    assert predictor.detect_emotion("Absolutely incredible, exceeded all expectations! Love it!", rating=5, sentiment="Positive") == "Happy"
    # Satisfied
    assert predictor.detect_emotion("Good product, works well and decent price.", rating=4, sentiment="Positive") == "Satisfied"
    # Neutral
    assert predictor.detect_emotion("Average experience, nothing special.", rating=3, sentiment="Neutral") == "Neutral"

def test_predictor_batch():
    predictor = ReviewPredictor()
    test_df = pd.DataFrame([
        {"review_id": "T1", "review_text": "The product is absolutely incredible and high quality!", "rating": 5, "product_category": "Fashion"},
        {"review_id": "T2", "review_text": "Terrible fit.", "rating": 1, "product_category": "Fashion"}
    ])
    results = predictor.predict_batch(test_df)

    assert len(results) == 2
    assert results.iloc[0]["sentiment"] == "Positive"
    assert results.iloc[0]["emotion"] == "Happy"
    assert results.iloc[1]["sentiment"] == "Negative"
    assert results.iloc[1]["priority"] == "High"
    assert results.iloc[1]["emotion"] == "Angry"
