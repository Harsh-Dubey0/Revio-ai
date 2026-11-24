# Revio AI — Customer Review Intelligence & Analytics Platform

**Live App:** [https://revio-ai.streamlit.app/](https://revio-ai.streamlit.app/)


> **Live Web App:** [Open Revio AI Intelligence Engine](https://revio-ai.streamlit.app/)

**Revio AI** is an end-to-end customer feedback analytics platform built with **Streamlit**, **PostgreSQL (Supabase)**, **Scikit-Learn / NLP pipelines**, and **LLM-powered business insights**. It ingests CSV review batches, runs multi-label sentiment, emotion, aspect, and priority scoring, stores results in PostgreSQL, and surfaces executive analytics across an interactive multi-page dashboard.

---

## Key Capabilities

- **Batch CSV Ingestion & Validation (`pages/2_Upload_Reviews.py`)**: Upload customer review CSV files with automatic column mapping, schema validation, and batch scoring via `ReviewPredictor`.
- **Multi-Dimensional NLP & ML Scoring (`src/inference/`, `src/nlp/`)**:
  - **Sentiment Analysis**: Classifies reviews into `Positive`, `Negative`, or `Neutral` with calibrated prediction confidence.
  - **Emotion Detection**: Identifies customer emotion states (`Satisfied`, `Frustrated`, `Neutral`, etc.).
  - **Aspect Extraction & Topic Clustering**: Extracts key product/service aspects (`Delivery Speed`, `Build Quality`, `Pricing & Value`, etc.).
  - **Priority Triage & Opportunity Engine**: Flags high-priority negative feedback and uncovers product/service improvement opportunities.
- **Interactive Analytics Suite (`pages/`)**:
  - **Overview Dashboard**: Executive KPIs, sentiment breakdown, and recent activity.
  - **Review Explorer**: Searchable, filterable review-level inspection.
  - **Sentiment & Aspect Analytics**: Category correlations, rating vs. sentiment distributions, and aspect-level pain points.
  - **AI Business Insights**: Automated strategic recommendations powered by `src/llm/recommendations.py`.
  - **Model Performance**: Evaluation metrics and metadata inspection (`models/model_metadata.json`).
  - **Upload History & DB Sync**: Batch audit log with PostgreSQL row verification and cascade deletion.
  - **Settings**: Workspace theme, card density, and runtime preferences.

---

## Project Architecture & Directory Structure

```text
Revio AI/
├── app.py                          # Main Streamlit application entry point
├── requirements.txt                # Python dependencies
├── pytest.ini                      # Pytest configuration
├── .env.example                    # Environment variable template
├── config/
│   └── settings.py                 # Centralized app, model, and DB settings
├── data/
│   └── samples/                    # Sample valid & invalid CSV files for testing
├── models/
│   ├── best_review_model.joblib    # Trained sentiment/classification model artifact
│   └── model_metadata.json         # Model training metrics & feature metadata
├── pages/
│   ├── 1_Overview_Dashboard.py     # Executive metrics & summary charts
│   ├── 2_Upload_Reviews.py         # CSV ingestion, column mapping & batch inference
│   ├── 3_Review_Explorer.py        # Granular review & prediction explorer
│   ├── 4_Sentiment_Analytics.py    # Sentiment trends, confidence & rating correlations
│   ├── 5_Aspect_Analytics.py       # Aspect-level breakdown & topic clusters
│   ├── 6_AI_Business_Insights.py   # LLM-generated recommendations & action items
│   ├── 7_Model_Performance.py      # Model evaluation metrics & diagnostics
│   ├── 8_Upload_History.py         # Batch history, DB verification & cleanup
│   └── 9_Settings.py               # UI & workspace configuration
├── reports/
│   └── eda/                        # Exploratory data analysis plots & summary JSON
├── scripts/
│   └── generate_dataset.py         # Synthetic multi-category review dataset generator
├── src/
│   ├── database/
│   │   ├── db.py                   # PostgreSQL / Supabase connection pooling
│   │   └── models.py               # CRUD operations for batches, reviews & predictions
│   ├── inference/
│   │   └── predictor.py            # ReviewPredictor batch & single-review inference
│   ├── llm/
│   │   └── recommendations.py      # AI business insight & recommendation generator
│   ├── nlp/
│   │   ├── aspect_extractor.py     # Product/service aspect tagging
│   │   ├── emotion_analyzer.py     # Customer emotion classification
│   │   ├── opportunity_engine.py   # Opportunity scoring from negative/neutral feedback
│   │   ├── preprocessor.py         # Text cleaning & ML feature preprocessing
│   │   └── topic_clusterer.py      # Review topic grouping
│   ├── training/
│   │   ├── eda.py                  # Exploratory data analysis pipeline
│   │   └── train_model.py          # Model training & artifact export script
│   ├── utils/
│   │   ├── formatting.py           # Number, percentage & display formatting helpers
│   │   └── theme.py                # Custom Streamlit CSS & card theme styling
│   └── validation/
│       └── csv_validator.py        # Pre-ingestion CSV structure & data validation
├── supabase/
│   └── migrations/
│       └── 20261124000000_initial_schema.sql  # PostgreSQL tables & schema migration
└── tests/                          # Unit & integration test suite
```

---

## Local Setup & Installation

### 1. Clone the Repository
```bash
git clone <your-github-repo-url>
cd "Revio AI"
```

### 2. Create & Activate Virtual Environment
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.ENV` (or `.env`) and fill in your PostgreSQL / Supabase credentials and LLM API keys:
```bash
cp .env.example .ENV
```

### 5. Initialize Database Schema
Apply `supabase/migrations/20261124000000_initial_schema.sql` to your PostgreSQL / Supabase database if setting up a fresh instance.

### 6. Run the Application
```bash
streamlit run app.py
```

---

## Running Tests

```bash
pytest
```

---

## Cloud Deployment (Streamlit Community Cloud / Hugging Face Spaces)

1. Push the repository to GitHub (ensure `.ENV`, `.backup/`, and `_backups/` are excluded via `.gitignore`).
2. Create a new app on **Streamlit Community Cloud** (`share.streamlit.io`) pointing to `app.py` (or a **Streamlit Space** on Hugging Face).
3. Add all keys from `.ENV` into the platform's **Secrets / Environment Variables** settings.
4. Deploy and verify database connectivity via **Upload History** (`pages/8_Upload_History.py`).