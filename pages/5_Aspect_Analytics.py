import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(page_title="Aspect & Emotion Analytics - Revio AI", page_icon="🧩", layout="wide")

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
        <div class="revio-pill">🧩 Revio AI • Aspect & Emotion Intelligence</div>
        <h1 style="margin: 0 0 6px 0; font-size: 2rem;">Aspect & Emotion Analytics</h1>
        <p class="revio-muted" style="margin: 0; font-size: 0.98rem;">
            Detailed breakdown of extracted operational aspects, customer sentiment drivers, and emotion profiles.
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
    if selected_batch_id:
        query = """
            SELECT p.aspect_labels, p.sentiment, p.emotion, r.product_category, r.rating
            FROM predictions p
            JOIN reviews r ON p.review_id = r.review_id
            WHERE r.batch_id = %s;
        """
        df = pd.read_sql(query, conn, params=(selected_batch_id,))
    else:
        query = """
            SELECT p.aspect_labels, p.sentiment, p.emotion, r.product_category, r.rating
            FROM predictions p
            JOIN reviews r ON p.review_id = r.review_id;
        """
        df = pd.read_sql(query, conn)
    conn.close()
except Exception as e:
    st.error(f"Error fetching aspect & emotion data: {e}")
    df = pd.DataFrame()

if df.empty:
    st.markdown(
        f"""
        <div class="revio-web-card" style="padding: {card_pad};">
            <div class="revio-card-title">No Aspect & Emotion Data Available</div>
            <div class="revio-card-sub" style="margin-bottom: 0;">
                No aspect and emotion analytics data available for this batch scope.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    df["emotion"] = df["emotion"].fillna("Neutral").astype(str).str.strip()
    df["emotion"] = df["emotion"].replace({"": "Neutral", "None": "Neutral", "nan": "Neutral"})
    df["sentiment"] = df["sentiment"].fillna("Neutral").astype(str).str.strip()
    df["product_category"] = df["product_category"].fillna("General").astype(str).str.strip()

    # Process Aspects
    df["aspect_list"] = df["aspect_labels"].fillna("").astype(str).str.split(";")
    df_exploded = df.explode("aspect_list")
    df_exploded["aspect_list"] = df_exploded["aspect_list"].str.strip()
    df_exploded = df_exploded[
        df_exploded["aspect_list"].notna()
        & (df_exploded["aspect_list"] != "")
        & (~df_exploded["aspect_list"].str.lower().isin(["none", "nan", "general"]))
    ]

    # Top Summary Metrics
    top_emotion = str(df["emotion"].mode().iloc[0]) if not df["emotion"].mode().empty else "Neutral"
    top_aspect = (
        str(df_exploded["aspect_list"].mode().iloc[0])
        if not df_exploded.empty and not df_exploded["aspect_list"].mode().empty
        else "N/A"
    )
    neg_aspect_mentions = (
        int((df_exploded["sentiment"].str.lower() == "negative").sum()) if not df_exploded.empty else 0
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Analyzed Reviews", format_integer(len(df)))
    with m2:
        st.metric("Top Emotion", top_emotion)
    with m3:
        st.metric("Top Aspect", top_aspect)
    with m4:
        st.metric(
            "Negative Aspect Mentions",
            format_integer(neg_aspect_mentions),
            delta_color="inverse",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 1. Emotion Analytics Section
    st.markdown(
        """
        <div class="revio-section-head">
            <div>
                <h3 class="revio-section-title">🎭 Emotion Classification</h3>
                <p class="revio-section-desc">Customer emotional distribution overall and across product categories.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    emotion_counts = df["emotion"].value_counts().reset_index()
    emotion_counts.columns = ["Emotion", "Count"]

    emotion_colors = {
        "Happy": "#10b981",
        "Satisfied": "#3b82f6",
        "Neutral": "#64748b",
        "Frustrated": "#f59e0b",
        "Angry": "#ef4444",
    }
    sentiment_colors = {
        "Positive": "#10b981",
        "Neutral": "#64748b",
        "Negative": "#ef4444",
        "Mixed": "#f59e0b",
    }

    col_em1, col_em2 = st.columns(2)
    with col_em1:
        st.markdown(
            f"""
            <div class="revio-web-card" style="padding: {card_pad};">
                <div class="revio-card-title">Overall Customer Emotional Distribution</div>
                <div class="revio-card-sub">Share of detected emotions across reviews.</div>
            """,
            unsafe_allow_html=True,
        )
        fig_em_pie = px.pie(
            emotion_counts,
            names="Emotion",
            values="Count",
            hole=0.45,
            color="Emotion",
            color_discrete_map=emotion_colors,
        )
        fig_em_pie.update_traces(textposition="inside", textinfo="percent+label")
        optimize_chart(fig_em_pie, height=330, showlegend=True, top_margin=16, bottom_margin=20)
        st.plotly_chart(fig_em_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_em2:
        st.markdown(
            f"""
            <div class="revio-web-card" style="padding: {card_pad};">
                <div class="revio-card-title">Customer Emotion Across Product Categories</div>
                <div class="revio-card-sub">Stacked emotion volume per category.</div>
            """,
            unsafe_allow_html=True,
        )
        fig_em_cat = px.histogram(
            df,
            x="product_category",
            color="emotion",
            barmode="stack",
            color_discrete_map=emotion_colors,
        )
        fig_em_cat.update_layout(xaxis_title="Category", yaxis_title="Review Count")
        optimize_chart(fig_em_cat, height=330, showlegend=True, top_margin=16, bottom_margin=36)
        st.plotly_chart(fig_em_cat, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Aspect Analytics Section
    st.markdown(
        """
        <div class="revio-section-head">
            <div>
                <h3 class="revio-section-title">🔍 Customer Experience Operational Aspects</h3>
                <p class="revio-section-desc">Most frequently mentioned product/service aspects and their sentiment split.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df_exploded.empty:
        st.markdown(
            f"""
            <div class="revio-web-card" style="padding: {card_pad};">
                <div class="revio-card-title">No Aspect Tags Found</div>
                <div class="revio-card-sub" style="margin-bottom: 0;">
                    No semicolon-separated aspect labels were present in this batch.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
                <div class="revio-web-card" style="padding: {card_pad};">
                    <div class="revio-card-title">Top Mentioned Aspects</div>
                    <div class="revio-card-sub">Top 10 aspects by total mention count.</div>
                """,
                unsafe_allow_html=True,
            )
            aspect_counts = df_exploded["aspect_list"].value_counts().reset_index()
            aspect_counts.columns = ["Aspect", "Mentions"]
            fig_aspect_bar = px.bar(
                aspect_counts.head(10),
                x="Mentions",
                y="Aspect",
                orientation="h",
                color="Mentions",
                color_continuous_scale=["#334155", t["accent"]],
            )
            fig_aspect_bar.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Mentions", yaxis_title="Aspect")
            optimize_chart(fig_aspect_bar, height=340, showlegend=False, top_margin=16, bottom_margin=34, horizontal=True)
            st.plotly_chart(fig_aspect_bar, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown(
                f"""
                <div class="revio-web-card" style="padding: {card_pad};">
                    <div class="revio-card-title">Sentiment Distribution per Aspect</div>
                    <div class="revio-card-sub">Positive, neutral, and negative breakdown per aspect.</div>
                """,
                unsafe_allow_html=True,
            )
            top10_names = aspect_counts.head(10)["Aspect"].tolist()
            df_top_aspects = df_exploded[df_exploded["aspect_list"].isin(top10_names)]
            fig_sent_aspect = px.histogram(
                df_top_aspects,
                y="aspect_list",
                color="sentiment",
                barmode="stack",
                color_discrete_map=sentiment_colors,
            )
            fig_sent_aspect.update_layout(
                yaxis=dict(categoryorder="array", categoryarray=list(reversed(top10_names))),
                xaxis_title="Mentions",
                yaxis_title="Aspect",
            )
            optimize_chart(fig_sent_aspect, height=340, showlegend=True, top_margin=16, bottom_margin=34, horizontal=True)
            st.plotly_chart(fig_sent_aspect, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Aspect Health Cards (Added polish)
        st.markdown(
            """
            <div class="revio-section-head" style="margin-top: 14px;">
                <div>
                    <h3 class="revio-section-title">📌 Top Aspect Health Snapshot</h3>
                    <p class="revio-section-desc">Positive sentiment share for the top 6 most discussed aspects.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        snapshot_aspects = aspect_counts.head(6)["Aspect"].tolist()
        snap_cols = st.columns(3)
        for i, asp in enumerate(snapshot_aspects):
            sub = df_exploded[df_exploded["aspect_list"] == asp]
            tot_m = len(sub)
            pos_m = int((sub["sentiment"].str.lower() == "positive").sum())
            neg_m = int((sub["sentiment"].str.lower() == "negative").sum())
            pos_pct = (pos_m / tot_m * 100) if tot_m > 0 else 0.0

            with snap_cols[i % 3]:
                st.markdown(
                    f"""
                    <div class="revio-web-card" style="padding: {card_pad}; margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; gap: 8px;">
                            <div class="revio-card-title" style="margin: 0;">{asp}</div>
                            <span class="revio-pill" style="margin: 0;">{format_integer(tot_m)} mentions</span>
                        </div>
                        <div class="revio-card-sub" style="margin-top: 8px; margin-bottom: 6px;">
                            Positive: <b>{pos_pct:.1f}%</b> ({format_integer(pos_m)}) • Negative: <b>{format_integer(neg_m)}</b>
                        </div>
                        <div class="revio-bar-track">
                            <div class="revio-bar-fill" style="width: {min(100.0, max(0.0, pos_pct)):.1f}%;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )