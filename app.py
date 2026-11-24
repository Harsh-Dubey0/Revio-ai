import os
import sys

import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(
    page_title="Revio AI Intelligence Engine",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.utils.formatting import format_integer
from src.utils.theme import apply_page_theme

try:
    from src.database.models import get_all_batches, get_analytics_summary
except Exception:
    get_all_batches = None
    get_analytics_summary = None

palette = apply_page_theme(include_metric_cards=True)

compact_mode = bool(st.session_state.get("revio_compact_cards", False))
card_pad = "14px 16px" if compact_mode else "18px 20px"

# Hero Banner
st.markdown(
    """
    <div class="revio-hero">
        <div class="revio-pill">🚀 Revio AI • Hybrid Review Intelligence</div>
        <h1 style="margin: 0 0 6px 0; font-size: 2.05rem;">Revio AI Intelligence Engine</h1>
        <p class="revio-muted" style="margin: 0; font-size: 0.98rem;">
            Customer review analytics with sentiment, aspect, and emotion intelligence, CSV ingestion,
            PostgreSQL storage, and Gemini + Smart Local recommendations.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Live Workspace Metrics
total_batches_cnt = 0
total_reviews_cnt = 0
db_connected = False

if get_all_batches is not None:
    try:
        batches_list = get_all_batches() or []
        db_connected = True
        total_batches_cnt = len(batches_list)
        total_reviews_cnt = sum(
            int(b.get("row_count") or b.get("total_rows") or 0) for b in batches_list
        )
    except Exception:
        db_connected = False

pos_cnt = 0
neg_cnt = 0
if get_analytics_summary is not None and db_connected:
    try:
        summary = get_analytics_summary() or {}
        sent = summary.get("sentiment_distribution") or summary.get("sentiment_counts") or {}
        pos_cnt = int(sent.get("positive") or sent.get("Positive") or 0)
        neg_cnt = int(sent.get("negative") or sent.get("Negative") or 0)
    except Exception:
        pos_cnt = 0
        neg_cnt = 0

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Batches", format_integer(total_batches_cnt))
with c2:
    st.metric("Reviews", format_integer(total_reviews_cnt))
with c3:
    st.metric("Positive", format_integer(pos_cnt))
with c4:
    st.metric("Negative", format_integer(neg_cnt), delta_color="inverse")

st.markdown("<br>", unsafe_allow_html=True)

# Core Capabilities & Modules
left, right = st.columns(2, gap="large")

with left:
    st.markdown(
        f"""
        <div class="revio-section-head">
            <div>
                <h3 class="revio-section-title">Core Capabilities</h3>
                <p class="revio-section-desc">What powers this workspace.</p>
            </div>
        </div>
        <div class="revio-web-card" style="padding: {card_pad};">
            <div class="revio-card-title">Hybrid AI Engine</div>
            <div class="revio-card-sub" style="margin-bottom: 0;">
                Generates executive insights via Gemini with automatic Smart Local fallback.
            </div>
        </div>
        <div class="revio-web-card" style="padding: {card_pad};">
            <div class="revio-card-title">CSV Ingestion & Batch Isolation</div>
            <div class="revio-card-sub" style="margin-bottom: 0;">
                Maps columns, validates rows, stores batches in PostgreSQL, and supports safe cascade deletion.
            </div>
        </div>
        <div class="revio-web-card" style="padding: {card_pad};">
            <div class="revio-card-title">Sentiment, Aspect & Emotion Signals</div>
            <div class="revio-card-sub" style="margin-bottom: 0;">
                Classifies sentiment, confidence, aspects, priority, and emotions across customer feedback.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        f"""
        <div class="revio-section-head">
            <div>
                <h3 class="revio-section-title">Workspace Modules</h3>
                <p class="revio-section-desc">Use the sidebar to jump into any page.</p>
            </div>
        </div>
        <div class="revio-web-card" style="padding: {card_pad};">
            <div class="revio-card-title">Upload Reviews & Upload History</div>
            <div class="revio-card-sub" style="margin-bottom: 0;">
                Upload review CSVs, track batch runs, export logs, and manage batch deletion.
            </div>
        </div>
        <div class="revio-web-card" style="padding: {card_pad};">
            <div class="revio-card-title">Review Explorer & Analytics</div>
            <div class="revio-card-sub" style="margin-bottom: 0;">
                Filter review cards, inspect sentiment distributions, and analyze aspects & emotions.
            </div>
        </div>
        <div class="revio-web-card" style="padding: {card_pad};">
            <div class="revio-card-title">Growth Insights, Model Performance & Settings</div>
            <div class="revio-card-sub" style="margin-bottom: 0;">
                Generate growth roadmaps, inspect calibration & errors, and run connection diagnostics.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )