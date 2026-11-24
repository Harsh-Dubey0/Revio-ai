import logging

import psycopg2
from psycopg2.extras import RealDictCursor
from supabase import create_client, Client

from config.settings import settings


logger = logging.getLogger("revio_db")


def get_db_connection():
    """Create a direct PostgreSQL connection."""
    if not settings.DATABASE_URL:
        raise ValueError("DATABASE_URL is not configured.")
    try:
        return psycopg2.connect(
            settings.DATABASE_URL,
            cursor_factory=RealDictCursor
        )
    except psycopg2.Error:
        logger.exception("Failed to connect to PostgreSQL.")
        raise


def get_supabase_client() -> Client:
    """Create a Supabase client."""
    if not settings.SUPABASE_URL or not settings.SUPABASE_SECRET_KEY:
        raise ValueError("Supabase credentials are not configured.")
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SECRET_KEY)


def test_database_connection() -> dict:
    """Test PostgreSQL connectivity."""
    result = {"success": False, "message": ""}
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            db_version = cur.fetchone()
        result["success"] = True
        result["message"] = (
            f"Successfully connected to PostgreSQL! "
            f"Version: {db_version['version']}"
        )
    except Exception as e:
        logger.exception("Database connection test failed.")
        result["message"] = f"Connection failed: {str(e)}"
    finally:
        if conn is not None:
            conn.close()
    return result
