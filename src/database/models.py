import logging
import json
from typing import Optional, List, Dict, Any
from src.database.db import get_db_connection

logger = logging.getLogger("revio_models")

def create_upload_batch(filename: str, row_count: int, model_version: str = "v1.0.0") -> str:
    """
    Create a new upload batch record in PostgreSQL and return batch_id.
    """
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO upload_batches (filename, row_count, status, model_version)
            VALUES (%s, %s, 'processing', %s)
            RETURNING batch_id;
            """,
            (filename, row_count, model_version)
        )
        row = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()
        return str(row["batch_id"])
    except Exception as e:
        conn.rollback()
        conn.close()
        logger.error(f"Error creating upload batch: {e}")
        raise e

def update_batch_status(batch_id: str, status: str):
    """
    Update batch status (e.g., 'completed', 'failed').
    """
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE upload_batches SET status = %s WHERE batch_id = %s;
            """,
            (status, batch_id)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        conn.rollback()
        conn.close()
        logger.error(f"Error updating batch status: {e}")
        raise e

def insert_review_and_prediction(batch_id: str, review_data: dict, prediction_data: dict):
    """
    Insert a review and its prediction transactionally.
    """
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        # Insert Review
        cur.execute(
            """
            INSERT INTO reviews (review_id, batch_id, customer_id, review_text, rating, review_date, product_category, product_name, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (review_id) DO NOTHING;
            """,
            (
                review_data.get("review_id"),
                batch_id,
                review_data.get("customer_id"),
                review_data.get("review_text"),
                int(review_data.get("rating", 3)),
                review_data.get("review_date"),
                review_data.get("product_category"),
                review_data.get("product_name"),
                review_data.get("source")
            )
        )

        # Insert Prediction (including emotion)
        cur.execute(
            """
            INSERT INTO predictions (review_id, sentiment, confidence, aspect_labels, priority, emotion)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (review_id) DO UPDATE
            SET sentiment = EXCLUDED.sentiment,
                confidence = EXCLUDED.confidence,
                aspect_labels = EXCLUDED.aspect_labels,
                priority = EXCLUDED.priority,
                emotion = EXCLUDED.emotion;
            """,
            (
                review_data.get("review_id"),
                prediction_data.get("sentiment", "Neutral"),
                float(prediction_data.get("confidence", 1.0)),  # ensure numeric
                prediction_data.get("aspect_labels", "General"),
                prediction_data.get("priority", "Low"),
                prediction_data.get("emotion", "Neutral")
            )
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        conn.rollback()
        conn.close()
        logger.error(f"Error inserting review and prediction: {e}")
        raise e

def insert_ai_recommendation(batch_id: str, recs: dict):
    """
    Save AI business recommendations (Gemini or Local Analytics Engine).
    Fixes the key mapping mismatch so complaints_summary and positives_summary are never NULL.
    """
    conn = get_db_connection()
    try:
        cur = conn.cursor()

        # Format complaints summary (handle both string and list inputs)
        complaints_val = recs.get("complaints_summary")
        if not complaints_val:
            major_complaints = recs.get("major_complaints", [])
            if isinstance(major_complaints, list):
                complaints_val = "\n".join([f"• {c}" for c in major_complaints]) if major_complaints else "No major systemic complaints identified."
            else:
                complaints_val = str(major_complaints)

        # Format positives summary (handle both string and list inputs)
        positives_val = recs.get("positives_summary")
        if not positives_val:
            positive_trends = recs.get("positive_trends", [])
            if isinstance(positive_trends, list):
                positives_val = "\n".join([f"• {p}" for p in positive_trends]) if positive_trends else "Positive customer engagement maintained across key categories."
            else:
                positives_val = str(positive_trends)

        # Prepare Action Plan JSON with engine metadata
        action_plan_data = recs.get("action_plan", [])
        plan_payload = {
            "action_items": action_plan_data if isinstance(action_plan_data, list) else action_plan_data.get("action_items", []),
            "engine": recs.get("engine", "Local Analytics Engine" if recs.get("is_fallback") else "Gemini AI"),
            "provider": recs.get("provider", "Local Python Engine" if recs.get("is_fallback") else "Google Gemini"),
            "model": recs.get("model", "local-analytics-v1" if recs.get("is_fallback") else "gemini-3.8-flash"),
            "generation_status": "success",
            "token_usage": recs.get("token_usage", None),
            "generated_at": recs.get("generated_at", "")
        }

        cur.execute(
            """
            INSERT INTO ai_recommendations (batch_id, executive_summary, complaints_summary, positives_summary, action_plan)
            VALUES (%s, %s, %s, %s, %s);
            """,
            (
                batch_id,
                recs.get("executive_summary", ""),
                complaints_val,
                positives_val,
                json.dumps(plan_payload)
            )
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        conn.rollback()
        conn.close()
        logger.error(f"Error inserting AI recommendation: {e}")
        raise e

def get_all_batches() -> List[Dict[str, Any]]:
    """
    Retrieve all upload batches ordered by upload timestamp descending.
    Returns an empty list if the database connection cannot be established.
    """
    try:
        conn = get_db_connection()
    except Exception as e:
        logger.error(f"Failed to connect to DB in get_all_batches: {e}")
        return []
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM upload_batches ORDER BY uploaded_at DESC;")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        conn.close()
        logger.error(f"Error fetching batches: {e}")
        return []


def get_analytics_summary(batch_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Query PostgreSQL for dashboard analytics metrics.
    Supports batch-wise isolation: filters by batch_id if provided.
    Returns safe defaults (with category_stats=[]) if the database is unavailable or queries fail.
    All keys guaranteed: total_reviews, avg_rating, sentiment_distribution,
    priority_distribution, emotion_distribution, rating_distribution,
    category_stats, aspect_counts, complaint_aspects, positive_aspects, batch_id.
    """
    # Safe-default structure — always contains every key the dashboard expects
    empty_summary: Dict[str, Any] = {
        "total_reviews": 0,
        "avg_rating": 0.0,
        "sentiment_distribution": {},
        "priority_distribution": {},
        "emotion_distribution": {},
        "rating_distribution": {},
        "category_stats": [],
        "aspect_counts": {},
        "complaint_aspects": {},
        "positive_aspects": {},
        "batch_id": batch_id,
    }

    try:
        conn = get_db_connection()
    except Exception as e:
        logger.error(f"Failed to connect to DB in get_analytics_summary: {e}")
        return empty_summary

    try:
        cur = conn.cursor()
        where_clause_r = "WHERE batch_id = %s" if batch_id else ""
        where_clause_p = (
            "JOIN reviews r ON p.review_id = r.review_id WHERE r.batch_id = %s"
            if batch_id else ""
        )
        params = (batch_id,) if batch_id else ()

        def safe_query(sql, default, fetch_one=False):
            try:
                cur.execute(sql, params)
                if fetch_one:
                    row = cur.fetchone()
                    return row if row is not None else default
                return cur.fetchall() or default
            except Exception as exc:
                logger.error(f"Analytics query failed: {exc}")
                return default

        # 1. Total reviews
        total_reviews_row = safe_query(
            f"SELECT COUNT(*) AS total_reviews FROM reviews {where_clause_r};",
            {"total_reviews": 0}, fetch_one=True
        )
        total_reviews = int(total_reviews_row.get("total_reviews", 0) or 0)

        # 2. Average rating
        avg_rating_row = safe_query(
            f"SELECT AVG(rating) AS avg_rating FROM reviews {where_clause_r};",
            {"avg_rating": 0.0}, fetch_one=True
        )
        avg_rating = round(float(avg_rating_row.get("avg_rating") or 0.0), 2)

        # 3. Sentiment distribution
        sentiment_rows = safe_query(
            f"SELECT p.sentiment, COUNT(*) AS count FROM predictions p {where_clause_p} GROUP BY p.sentiment;",
            []
        )
        sentiment_dist = {
            (row.get("sentiment") or "Unknown"): int(row.get("count", 0))
            for row in sentiment_rows
        }

        # 4. Priority distribution
        priority_rows = safe_query(
            f"SELECT p.priority, COUNT(*) AS count FROM predictions p {where_clause_p} GROUP BY p.priority;",
            []
        )
        priority_dist = {
            (row.get("priority") or "Unknown"): int(row.get("count", 0))
            for row in priority_rows
        }

        # 5. Emotion distribution
        emotion_rows = safe_query(
            f"SELECT p.emotion, COUNT(*) AS count FROM predictions p {where_clause_p} GROUP BY p.emotion;",
            []
        )
        emotion_dist = {
            (row.get("emotion") or "Neutral"): int(row.get("count", 0))
            for row in emotion_rows
        }

        # 6. Category performance
        cat_rows = safe_query(
            f"""
            SELECT product_category, COUNT(*) AS count, AVG(rating) AS avg_rating
            FROM reviews {where_clause_r}
            GROUP BY product_category
            ORDER BY count DESC;
            """,
            []
        )
        category_stats = [
            {
                "product_category": row.get("product_category") or "Unknown",
                "count": int(row.get("count", 0)),
                "avg_rating": round(float(row.get("avg_rating") or 0.0), 2),
            }
            for row in cat_rows
        ]

        # 7. Rating distribution
        rating_rows = safe_query(
            f"SELECT rating, COUNT(*) AS count FROM reviews {where_clause_r} GROUP BY rating ORDER BY rating;",
            []
        )
        rating_dist = {
            int(row.get("rating", 0)): int(row.get("count", 0))
            for row in rating_rows
        }

        # 8. Aspect analysis
        aspect_rows = safe_query(
            f"SELECT p.aspect_labels, p.sentiment FROM predictions p {where_clause_p};",
            []
        )
        aspect_counts: Dict[str, int] = {}
        complaint_aspects: Dict[str, int] = {}
        positive_aspects: Dict[str, int] = {}
        for row in aspect_rows:
            aspects_str = row.get("aspect_labels") or ""
            sent = row.get("sentiment") or "Neutral"
            for asp in aspects_str.split(";"):
                asp = asp.strip()
                if not asp:
                    continue
                aspect_counts[asp] = aspect_counts.get(asp, 0) + 1
                if sent in ("Negative", "Mixed"):
                    complaint_aspects[asp] = complaint_aspects.get(asp, 0) + 1
                elif sent == "Positive":
                    positive_aspects[asp] = positive_aspects.get(asp, 0) + 1

        return {
            "total_reviews": total_reviews,
            "avg_rating": avg_rating,
            "sentiment_distribution": sentiment_dist,
            "priority_distribution": priority_dist,
            "emotion_distribution": emotion_dist,
            "rating_distribution": rating_dist,
            "category_stats": category_stats,
            "aspect_counts": aspect_counts,
            "complaint_aspects": complaint_aspects,
            "positive_aspects": positive_aspects,
            "batch_id": batch_id,
        }

    except Exception as e:
        logger.error(f"Error fetching analytics summary: {e}")
        return empty_summary
    finally:
        try:
            conn.close()
        except Exception:
            pass

def delete_upload_batch(batch_id: str) -> bool:
    """
    Safely delete an upload batch and all cascading records (reviews, predictions, ai_recommendations).
    """
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM upload_batches WHERE batch_id = %s;", (batch_id,))
        deleted_count = cur.rowcount
        conn.commit()
        cur.close()
        conn.close()
        return deleted_count > 0
    except Exception as e:
        conn.rollback()
        conn.close()
        logger.error(f"Error deleting upload batch {batch_id}: {e}")
        raise e

def delete_reviews(review_ids: List[str]) -> int:
    """
    Safely delete selected reviews by ID (cascades to predictions).
    """
    if not review_ids:
        return 0
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM reviews WHERE review_id = ANY(%s);", (review_ids,))
        deleted_count = cur.rowcount
        conn.commit()
        cur.close()
        conn.close()
        return deleted_count
    except Exception as e:
        conn.rollback()
        conn.close()
        logger.error(f"Error deleting reviews: {e}")
        raise e
