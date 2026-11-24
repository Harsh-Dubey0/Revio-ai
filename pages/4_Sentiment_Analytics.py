import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(page_title="Sentiment Analytics - Revio AI", page_icon="📈", layout="wide")

from src.database.db import get_db_connection
from src.database.models import get_all_batches
from src.utils.formatting import format_integer
from src.utils.theme import apply_page_theme, optimize_chart

t = apply_page_theme(include_metric_cards=True)

compact_mode = bool(st.session_state.get("revio_compact_cards", False))
card_pad = "14px 16px" if compact_mode else "18px 20px"

# Revio AI Hero Banner
st.markdown(
    """
    <div class="revio-hero">
        <div class="revio-pill">📈 Revio AI • Sentiment Intelligence</div>
        <h1 style="margin: 0 0 6px 0; font-size: 2rem;">Sentiment Analytics & Distribution</h1>
        <p class="revio-muted" style="margin: 0; font-size: 0.98rem;">
            Deep-dive sentiment trends, category correlations, and prediction confidence analysis.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Batch Selector
batches = get_all_batches() or []
batch_options = {"All Batches (Global)": None}
for b in batches:
    date_str = (
        b["uploaded_at"].strftime("%Y-%m-%d %H:%M")
        if b.get("uploaded_at") and hasattr(b["uploaded_at"], "strftime")
        else str(b.get("uploaded_at", ""))[:16]
    )
    batch_options[f"{b['filename']} ({date_str})"] = str(b["batch_id"])

default_scope = st.session_state.get("revio_default_batch_scope", "All Batches")
default_batch_idx = 1 if (default_scope == "Latest Batch" and len(batch_options) > 1) else 0

st.sidebar.header("Data Scope")
selected_batch_name = st.sidebar.selectbox(
    "Filter by Batch",
    list(batch_options.keys()),
    index=default_batch_idx,
)
selected_batch_id = batch_options[selected_batch_name]

try:
    conn = get_db_connection()
    cur = conn.cursor()
    if selected_batch_id:
        query = """
            SELECT
                r.product_category,
                r.rating,
                p.sentiment,
                CAST(p.confidence AS DOUBLE PRECISION) AS confidence,
                p.priority,
                p.emotion
            FROM reviews r
            LEFT JOIN predictions p ON r.review_id = p.review_id
            WHERE r.batch_id = %s;
        """
        cur.execute(query, (selected_batch_id,))
    else:
        query = """
            SELECT
                r.product_category,
                r.rating,
                p.sentiment,
                CAST(p.confidence AS DOUBLE PRECISION) AS confidence,
                p.priority,
                p.emotion
            FROM reviews r
            LEFT JOIN predictions p ON r.review_id = p.review_id;
        """
        cur.execute(query)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if rows:
        df = pd.DataFrame([dict(r) for r in rows])
    else:
        df = pd.DataFrame(columns=["product_category", "rating", "sentiment", "confidence", "priority", "emotion"])

    if "confidence" in df.columns:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    if "sentiment" in df.columns:
        df["sentiment"] = df["sentiment"].astype(str).str.strip()
    if "product_category" in df.columns:
        df["product_category"] = df["product_category"].fillna("General").astype(str).str.strip()
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    df = df[df["sentiment"].notna() & (df["sentiment"] != "None") & (df["sentiment"] != "")]
except Exception as e:
    st.error(f"Error fetching sentiment analytics data: {e}")
    df = pd.DataFrame()

if df.empty:
    st.markdown(
        f"""
        <div class="revio-web-card" style="padding: {card_pad};">
            <div class="revio-card-title">No Sentiment Data Available</div>
            <div class="revio-card-sub" style="margin-bottom: 0;">
                No sentiment analytics records available for the selected batch scope.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    total = len(df)
    pos_count = int((df["sentiment"] == "Positive").sum())
    neg_count = int((df["sentiment"] == "Negative").sum())
    avg_conf = float(df["confidence"].mean()) if total > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Reviews", format_integer(total))
    with c2:
        st.metric("Positive Sentiment", format_integer(pos_count), f"{pos_count / total * 100:.1f}%")
    with c3:
        st.metric(
            "Negative Sentiment",
            format_integer(neg_count),
            f"{neg_count / total * 100:.1f}%",
            delta_color="inverse",
        )
    with c4:
        st.metric("Mean Prediction Confidence", f"{avg_conf:.2%}")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="revio-section-head">
            <div>
                <h3 class="revio-section-title">📊 Category & Rating Correlations</h3>
                <p class="revio-section-desc">Compare sentiment volume across categories and star-rating alignment.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    color_map = {
        "Positive": "#10b981",
        "Neutral": "#64748b",
        "Negative": "#ef4444",
        "Mixed": "#f59e0b",
    }

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div class="revio-web-card" style="padding: {card_pad};">
                <div class="revio-card-title">Sentiment by Product Category</div>
                <div class="revio-card-sub">Grouped sentiment counts across categories.</div>
            """,
            unsafe_allow_html=True,
        )
        fig_cat_sent = px.histogram(
            df,
            x="product_category",
            color="sentiment",
            barmode="group",
            color_discrete_map=color_map,
        )
        fig_cat_sent.update_layout(xaxis_title="Product Category", yaxis_title="Review Count")
        optimize_chart(fig_cat_sent, height=330, showlegend=True, top_margin=16, bottom_margin=36)
        st.plotly_chart(fig_cat_sent, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            f"""
            <div class="revio-web-card" style="padding: {card_pad};">
                <div class="revio-card-title">Rating vs Sentiment Correlation</div>
                <div class="revio-card-sub">Star rating spread across predicted sentiment classes.</div>
            """,
            unsafe_allow_html=True,
        )
        rating_df = df.dropna(subset=["rating"])
        if rating_df.empty:
            st.info("No numeric rating values available for this selection.")
        else:
            fig_box = px.box(
                rating_df,
                x="sentiment",
                y="rating",
                color="sentiment",
                color_discrete_map=color_map,
            )
            fig_box.update_layout(xaxis_title="Sentiment Class", yaxis_title="Customer Rating (1-5)")
            optimize_chart(fig_box, height=330, showlegend=False, top_margin=16, bottom_margin=36)
            st.plotly_chart(fig_box, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="revio-section-head">
            <div>
                <h3 class="revio-section-title">🎯 Prediction Confidence Distribution</h3>
                <p class="revio-section-desc">How certain Revio AI predictions are across the selected reviews.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="revio-web-card" style="padding: {card_pad};">
            <div class="revio-card-title">Model Inference Confidence Score Distribution</div>
            <div class="revio-card-sub">Histogram of confidence scores from 0.0 to 1.0.</div>
        """,
        unsafe_allow_html=True,
    )
    fig_conf = px.histogram(
        df,
        x="confidence",
        nbins=25,
        color_discrete_sequence=[t["accent"]],
    )
    fig_conf.update_layout(xaxis_title="Confidence Score (0.0 to 1.0)", yaxis_title="Count of Predictions")
    optimize_chart(fig_conf, height=310, showlegend=False, top_margin=16, bottom_margin=36)
    st.plotly_chart(fig_conf, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)