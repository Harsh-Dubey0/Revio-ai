import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

class Settings:
    PROJECT_NAME: str = "Revio AI Intelligence Engine"
    VERSION: str = "1.0.0"

    # Project Root
    BASE_DIR: Path = Path(__file__).resolve().parent.parent

    # Database
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_SECRET_KEY: str = os.getenv("SUPABASE_SECRET_KEY", os.getenv("SUPABASE_KEY", ""))
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

    # LLM
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # Paths
    MODEL_PATH: Path = BASE_DIR / "models" / "best_review_model.joblib"
    MODEL_METADATA_PATH: Path = BASE_DIR / "models" / "model_metadata.json"
    DATASET_PATH: Path = BASE_DIR / "data" / "generated" / "reviews_dataset.csv"

settings = Settings()
