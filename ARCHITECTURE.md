# Revio AI Intelligence Engine — Architecture Documentation

This document describes the high-level system architecture, data flow, and module interactions of the Revio AI Intelligence Engine.

## Component Overview

Revio AI uses a modular, decoupled architecture organized across UI, validation, NLP/ML inference, database persistence, and AI recommendation layers:

```text
       +-------------------------------------------------------+
       |           Streamlit Multi-Page UI (pages/)            |
       | (Overview, Upload, Explorer, Analytics, History, etc.)|
       +-----------------------+-------------------------------+
                               |
            +------------------+------------------+
            |                                     |
            v                                     v
+-----------------------+               +-----------------------+
|  CSV Validation       |               | NLP & ML Inference    |
|  (csv_validator.py)   |               | (ReviewPredictor +    |
+-----------+-----------+               |  src/nlp/ modules)    |
            |                           +-----------+-----------+
            |                                       |
            +-------------------+-------------------+
                                |
                                v
                  +---------------------------+
                  |  PostgreSQL Database      |
                  | (psycopg2 / Supabase,     |
                  |  db.py & models.py)       |
                  +-------------+-------------+
                                |
                                v
                  +---------------------------+
                  |  AI Insights Engine       |
                  | (src/llm/                 |
                  |  recommendations.py)      |
                  +---------------------------+