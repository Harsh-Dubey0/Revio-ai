import pytest
from src.database.models import get_analytics_summary, get_all_batches, insert_ai_recommendation
from src.database.db import get_db_connection

def test_get_all_batches_schema():
    batches = get_all_batches()
    assert isinstance(batches, list)
    if batches:
        b = batches[0]
        assert "batch_id" in b
        assert "filename" in b
        assert "row_count" in b
        assert "status" in b

def test_analytics_summary_structure():
    summary = get_analytics_summary()
    assert isinstance(summary, dict)
    assert "total_reviews" in summary
    assert "avg_rating" in summary
    assert "sentiment_distribution" in summary
    assert "priority_distribution" in summary
    assert "emotion_distribution" in summary
    assert "aspect_counts" in summary
    assert "complaint_aspects" in summary
    assert "positive_aspects" in summary

def test_analytics_summary_batch_isolation():
    batches = get_all_batches()
    if len(batches) >= 2:
        b1_id = str(batches[0]["batch_id"])
        b2_id = str(batches[1]["batch_id"])
        
        sum1 = get_analytics_summary(batch_id=b1_id)
        sum2 = get_analytics_summary(batch_id=b2_id)
        
        assert sum1["batch_id"] == b1_id
        assert sum2["batch_id"] == b2_id
        assert sum1["total_reviews"] > 0
        assert sum2["total_reviews"] > 0
        assert sum1["batch_id"] != sum2["batch_id"]

def test_ai_recommendation_mapping():
    """Verify complaints_summary and positives_summary are mapped without NULL."""
    batches = get_all_batches()
    if batches:
        target_batch_id = str(batches[0]["batch_id"])
        test_payload = {
            "is_fallback": True,
            "engine": "Local Analytics Engine",
            "provider": "Local Python Engine",
            "model": "local-analytics-v1",
            "executive_summary": "Test Executive Summary for Data Integrity Audit.",
            "major_complaints": ["Complaint A about delivery", "Complaint B about packaging"],
            "positive_trends": ["Positive trend in quality", "Friendly customer support"],
            "action_plan": [
                {"priority": "High", "category": "Logistics", "action": "Improve delivery speed."}
            ]
        }
        insert_ai_recommendation(target_batch_id, test_payload)

        # Query back to verify no NULL
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT complaints_summary, positives_summary, action_plan FROM ai_recommendations WHERE batch_id = %s ORDER BY generated_at DESC LIMIT 1;",
            (target_batch_id,)
        )
        row = cur.fetchone()
        conn.close()

        assert row is not None
        assert row["complaints_summary"] is not None
        assert "Complaint A" in row["complaints_summary"]
        assert row["positives_summary"] is not None
        assert "Positive trend" in row["positives_summary"]
