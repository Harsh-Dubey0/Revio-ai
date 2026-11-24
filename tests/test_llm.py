import pytest
from unittest.mock import patch, MagicMock
from src.llm.recommendations import generate_ai_recommendations, LocalAnalyticsEngine

@pytest.fixture
def sample_summary():
    return {
        "total_reviews": 250,
        "avg_rating": 4.15,
        "sentiment_distribution": {"Positive": 180, "Negative": 40, "Neutral": 30},
        "priority_distribution": {"High": 40, "Medium": 30, "Low": 180},
        "emotion_distribution": {"Happy": 120, "Satisfied": 60, "Frustrated": 25, "Angry": 15, "Neutral": 30},
        "category_stats": [
            {"product_category": "Electronics", "avg_rating": 4.6, "count": 150},
            {"product_category": "Fashion", "avg_rating": 3.4, "count": 100}
        ],
        "complaint_aspects": {"Delivery & Shipping": 30, "Product Quality": 10},
        "positive_aspects": {"Product Quality": 110, "Customer Service": 70}
    }

def test_local_analytics_engine_direct(sample_summary):
    """Verify Local Analytics Engine produces data-driven executive insights directly."""
    recs = LocalAnalyticsEngine.generate_insights(sample_summary)
    assert recs["engine"] == "Local Analytics Engine"
    assert recs["is_fallback"] is True
    assert "250" in recs["executive_summary"]
    assert "Electronics" in recs["executive_summary"]
    assert len(recs["major_complaints"]) > 0
    assert len(recs["positive_trends"]) > 0
    assert len(recs["action_plan"]) == 3
    assert recs["action_plan"][0]["priority"] == "High"

def test_recommendations_force_local(sample_summary):
    """Verify force_local flag routes directly to Local Analytics Engine."""
    recs = generate_ai_recommendations(sample_summary, force_local=True)
    assert recs["engine"] == "Local Analytics Engine"
    assert recs["is_fallback"] is True
    assert recs["generation_status"] == "success"

def test_recommendations_gemini_429_quota_fallback(sample_summary):
    """Verify automatic graceful failover to Local Analytics Engine on HTTP 429 Quota Exhaustion."""
    with patch("config.settings.settings.GEMINI_API_KEY", "test_key_xyz"):
        with patch("google.generativeai.GenerativeModel") as mock_model_cls:
            mock_model = MagicMock()
            mock_model.generate_content.side_effect = Exception("429 ResourceExhausted: Quota exceeded for model gemini-3.8-flash")
            mock_model_cls.return_value = mock_model

            recs = generate_ai_recommendations(sample_summary)
            assert recs["engine"] == "Local Analytics Engine"
            assert recs["is_fallback"] is True
            assert "429" in recs["fallback_reason"] or "Quota" in recs["fallback_reason"]
            assert len(recs["action_plan"]) > 0

def test_recommendations_gemini_404_model_fallback(sample_summary):
    """Verify automatic failover to Local Analytics Engine on HTTP 404 Model Not Found."""
    with patch("config.settings.settings.GEMINI_API_KEY", "test_key_xyz"):
        with patch("google.generativeai.GenerativeModel") as mock_model_cls:
            mock_model = MagicMock()
            mock_model.generate_content.side_effect = Exception("404 NotFound: models/gemini-1.5-flash not found")
            mock_model_cls.return_value = mock_model

            recs = generate_ai_recommendations(sample_summary)
            assert recs["engine"] == "Local Analytics Engine"
            assert recs["is_fallback"] is True
            assert "404" in recs["fallback_reason"]

def test_recommendations_gemini_success_mock(sample_summary):
    """Verify parsing and metadata recording when Gemini returns successful structured JSON."""
    mock_payload = {
        "executive_summary": "Gemini-generated strategic report for executive leadership.",
        "major_complaints": ["Shipping bottleneck in Region A", "Packaging issue"],
        "positive_trends": ["High product reliability", "Great support"],
        "action_plan": [
            {"priority": "High", "category": "Logistics", "action": "Switch 3PL provider."},
            {"priority": "Medium", "category": "Packaging", "action": "Reinforce cartons."}
        ]
    }
    with patch("config.settings.settings.GEMINI_API_KEY", "valid_api_key_123"):
        with patch("google.generativeai.GenerativeModel") as mock_model_cls:
            mock_model = MagicMock()
            mock_resp = MagicMock()
            mock_resp.text = f"```json\n{import_json_str(mock_payload)}\n```"
            mock_resp.usage_metadata = MagicMock(prompt_token_count=120, candidates_token_count=85, total_token_count=205)
            mock_model.generate_content.return_value = mock_resp
            mock_model_cls.return_value = mock_model

            recs = generate_ai_recommendations(sample_summary)
            assert recs["engine"] == "Gemini AI"
            assert recs["is_fallback"] is False
            assert recs["executive_summary"] == mock_payload["executive_summary"]
            assert recs["token_usage"]["total_tokens"] == 205
            assert "Shipping bottleneck" in recs["complaints_summary"]

def import_json_str(data):
    import json
    return json.dumps(data)
