# Revio AI Intelligence Engine — Deployment Guide

This guide details steps for cloud and local deployment.

## Prerequisites
- Python 3.10 or higher
- Supabase PostgreSQL account

## Local Deployment
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set up environment:
   ```bash
   cp .env.example .env
   ```
3. Run migrations locally or through Supabase CLI.
4. Launch the dashboard:
   ```bash
   streamlit run app.py
   ```

## Streamlit Community Cloud
To deploy to Streamlit Community Cloud, configure the required keys in your deployment "Secrets" section:
- `GEMINI_API_KEY`
- `SUPABASE_URL`
- `SUPABASE_SECRET_KEY`
- `DATABASE_URL`
