import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.database.models import get_all_batches, get_analytics_summary
from src.utils.formatting import format_integer, format_percentage, format_rating
from src.utils.theme import apply_page_theme, optimize_chart

st.set_page_config(page_title="Revio AI - Overview", page_icon="🌐", layout="wide")
apply_page_theme(include_metric_cards=True)

# Sidebar: Only Data Scope (Theme is changed strictly inside 9_Settings.py)
batches = get_all_batches()
batch_options = {"All Batches (Global Analytics)": None}
for b in batches:
    date_str = (
        b["uploaded_at"].strftime("%Y-%m-%d %H:%M")
        if hasattr(b["uploaded_at"], "strftime")
        else str(b["uploaded_at"])[:16]
    )
    batch_options[f"{b['filename']} ({date_str})"] = str(b["batch_id"])

st.sidebar.header("Data Scope")
selected_label = st.sidebar.selectbox("Active Dataset", list(batch_options.keys()))
selected_batch_id = batch_options[selected_label]

summary = get_analytics_summary(batch_id=selected_batch_id)

total_reviews = summary["total_reviews"]
avg_rating = summary["avg_rating"]
sentiment_dist = summary["sentiment_distribution"] or {}
priority_dist = summary["priority_distribution"] or {}
emotion_dist = summary.get("emotion_distribution") or {}
category_stats = summary.get("category_stats") or []

pos_count = sentiment_dist.get("Positive", 0)
neg_count = sentiment_dist.get("Negative", 0)
pos_pct = (pos_count / total_reviews) if total_reviews > 0 else 0
neg_pct = (neg_count / total_reviews) if total_reviews > 0 else 0
high_priority = priority_dist.get("High", 0)
top_emotion = max(emotion_dist, key=emotion_dist.get) if emotion_dist else "N/A"

# Revio AI Hero Banner
scope_badge = f"Revio AI • Batch: {selected_label}" if selected_batch_id else "Revio AI • All Batches"
st.markdown(
    f"""
    <div class="revio-hero">
        <div class="revio-pill">✨ {scope_badge}</div>
        <h1 style="margin: 0 0 6px 0; font-size: 2rem;">Revio AI Overview</h1>
        <p class="revio-muted" style="margin: 0; font-size: 0.98rem;">
            Review sentiment, customer emotions, priority alerts, and category satisfaction across Revio AI.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# KPI Row with extra context deltas
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Reviews", format_integer(total_reviews), f"{len(category_stats)} active categories")
with col2:
    st.metric("Average Rating", format_rating(avg_rating), "Out of 5.0 stars")
with col3:
    st.metric(
        "Positive Share",
        format_percentage(pos_pct),
        f"{format_integer(pos_count)} positive vs {format_percentage(neg_pct)} negative",
    )
with col4:
    st.metric(
        "High-Priority Queue",
        format_integer(high_priority),
        f"Top emotion: {top_emotion}",
        delta_color="inverse",
    )

st.markdown("<br>", unsafe_allow_html=True)

if total_reviews == 0:
    st.markdown(
        """
        <div class="revio-web-card">
            <div class="revio-card-title">No Review Data Available Yet</div>
            <div class="revio-card-sub">
                Upload a review CSV from the <b>Upload Reviews</b> page to unlock Revio AI sentiment trends, emotion insights, and category ratings.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # Row 1: Sentiment Donut + Priority Bar
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class="revio-web-card">
                <div class="revio-card-title">Sentiment Polarity Mix</div>
                <div class="revio-card-sub">Share of Positive, Neutral, Mixed, and Negative feedback</div>
            """,
            unsafe_allow_html=True,
        )
        ordered_sent = ["Positive", "Neutral", "Mixed", "Negative"]
        sent_keys = [k for k in ordered_sent if k in sentiment_dist] + [
            k for k in sentiment_dist if k not in ordered_sent
        ]
        if sent_keys:
            fig_sent = px.pie(
                names=sent_keys,
                values=[sentiment_dist[k] for k in sent_keys],
                hole=0.55,
                color=sent_keys,
                color_discrete_map={
                    "Positive": "#10B981",
                    "Neutral": "#64748B",
                    "Mixed": "#F59E0B",
                    "Negative": "#EF4444",
                },
            )
            fig_sent.update_traces(
                textposition="inside",
                textinfo="percent",
                hovertemplate="<b>%{label}</b><br>Reviews: %{value}<br>Share: %{percent}<extra></extra>",
            )
            fig_sent = optimize_chart(fig_sent, height=290, showlegend=True, top_margin=8, bottom_margin=8)
            fig_sent.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_sent, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(
            """
            <div class="revio-web-card">
                <div class="revio-card-title">Action Priority Queue</div>
                <div class="revio-card-sub">Workload distribution across High, Medium, and Low urgency</div>
            """,
            unsafe_allow_html=True,
        )
        ordered_prio = ["High", "Medium", "Low"]
        prio_keys = [k for k in ordered_prio if k in priority_dist] + [
            k for k in priority_dist if k not in ordered_prio
        ]
        if prio_keys:
            fig_prio = px.bar(
                x=prio_keys,
                y=[priority_dist[k] for k in prio_keys],
                text=[priority_dist[k] for k in prio_keys],
                labels={"x": "Priority Tier", "y": "Reviews"},
                color=prio_keys,
                color_discrete_map={"High": "#EF4444", "Medium": "#F59E0B", "Low": "#10B981"},
            )
            fig_prio.update_traces(
                textposition="outside",
                cliponaxis=False,
                hovertemplate="<b>%{x} Priority</b><br>Reviews: %{y}<extra></extra>",
            )
            fig_prio = optimize_chart(fig_prio, height=290, showlegend=False)
            st.plotly_chart(fig_prio, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Row 2: Emotion Breakdown + Horizontal Ranked Categories
    c3, c4 = st.columns(2)

    with c3:
        st.markdown(
            """
            <div class="revio-web-card">
                <div class="revio-card-title">Customer Emotion Spectrum</div>
                <div class="revio-card-sub">Emotional tone detected across customer feedback</div>
            """,
            unsafe_allow_html=True,
        )
        if emotion_dist:
            ordered_em = ["Happy", "Satisfied", "Neutral", "Frustrated", "Angry"]
            em_keys = [k for k in ordered_em if k in emotion_dist] + [
                k for k in emotion_dist if k not in ordered_em
            ]
            fig_em = px.bar(
                x=em_keys,
                y=[emotion_dist[k] for k in em_keys],
                text=[emotion_dist[k] for k in em_keys],
                labels={"x": "Emotion", "y": "Reviews"},
                color=em_keys,
                color_discrete_map={
                    "Happy": "#10B981",
                    "Satisfied": "#3B82F6",
                    "Neutral": "#64748B",
                    "Frustrated": "#F59E0B",
                    "Angry": "#EF4444",
                },
            )
            fig_em.update_traces(
                textposition="outside",
                cliponaxis=False,
                hovertemplate="<b>%{x}</b><br>Reviews: %{y}<extra></extra>",
            )
            fig_em = optimize_chart(fig_em, height=300, showlegend=False)
            st.plotly_chart(fig_em, use_container_width=True)
        else:
            st.info("Emotion breakdown is not available for this selection yet.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown(
            """
            <div class="revio-web-card">
                <div class="revio-card-title">Category Volume & Rating Leader</div>
                <div class="revio-card-sub">Horizontal rank of top categories—color shows average star rating</div>
            """,
            unsafe_allow_html=True,
        )
        if category_stats:
            df_cat = pd.DataFrame(category_stats)
            cat_col = "product_category" if "product_category" in df_cat.columns else df_cat.columns[0]
            rating_col = "avg_rating" if "avg_rating" in df_cat.columns else None
            df_cat = df_cat.sort_values("count", ascending=True).tail(7)

            fig_cat = px.bar(
                df_cat,
                x="count",
                y=cat_col,
                orientation="h",
                text="count",
                color=rating_col,
                labels={"count": "Reviews", cat_col: "Category", "avg_rating": "Avg Rating"},
                color_continuous_scale="Teal",
                range_color=[1, 5] if rating_col else None,
            )
            fig_cat.update_traces(
                textposition="outside",
                cliponaxis=False,
                hovertemplate="<b>%{y}</b><br>Reviews: %{x}<br>Avg Rating: %{marker.color:.2f}<extra></extra>",
            )
            fig_cat = optimize_chart(fig_cat, height=300, showlegend=False, horizontal=True)
            fig_cat.update_layout(coloraxis_colorbar=dict(title="Rating", thickness=11, len=0.72))
            st.plotly_chart(fig_cat, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Row 3: Revio AI Category Deep-Dive Cards
    if category_stats:
        df_detail = pd.DataFrame(category_stats).copy()
        cat_col = "product_category" if "product_category" in df_detail.columns else df_detail.columns[0]
        rating_col = "avg_rating" if "avg_rating" in df_detail.columns else None

        df_detail[cat_col] = df_detail[cat_col].fillna("General").astype(str)
        df_detail["count"] = pd.to_numeric(df_detail["count"], errors="coerce").fillna(0).astype(int)
        if rating_col:
            df_detail[rating_col] = pd.to_numeric(df_detail[rating_col], errors="coerce").fillna(0.0)

        df_detail["share_pct"] = (
            ((df_detail["count"] / total_reviews) * 100).round(1) if total_reviews > 0 else 0.0
        )

        st.markdown(
            """
            <div class="revio-section-head">
                <div>
                    <h3 class="revio-section-title">🧭 Revio AI Category Deep-Dive</h3>
                    <p class="revio-section-desc">Review volume, traffic share, and average star ratings across each category in Revio AI.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        f_col1, f_col2 = st.columns([2, 1])
        with f_col1:
            search_q = st.text_input(
                "Filter category",
                placeholder="Search category name (e.g., Electronics, Logistics)...",
                label_visibility="collapsed",
            )
        with f_col2:
            sort_by = st.selectbox(
                "Sort categories by",
                options=["Highest Volume", "Highest Rating", "Lowest Rating"],
                label_visibility="collapsed",
            )

        filtered_df = df_detail.copy()
        if search_q.strip():
            filtered_df = filtered_df[
                filtered_df[cat_col].str.contains(search_q.strip(), case=False, na=False)
            ]

        if sort_by == "Highest Rating" and rating_col:
            filtered_df = filtered_df.sort_values([rating_col, "count"], ascending=[False, False])
        elif sort_by == "Lowest Rating" and rating_col:
            filtered_df = filtered_df.sort_values([rating_col, "count"], ascending=[True, False])
        else:
            filtered_df = filtered_df.sort_values("count", ascending=False)

        if filtered_df.empty:
            st.info("No matching category found for your filter.")
        else:
            for _, row in filtered_df.iterrows():
                c_name = row[cat_col]
                c_count = int(row["count"])
                c_share = float(row["share_pct"])
                c_rating = float(row[rating_col]) if rating_col else 0.0
                bar_width = min(max(c_share, 3.0), 100.0)

                st.markdown(
                    f"""
                    <div class="revio-web-card" style="padding: 15px 20px; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                            <div>
                                <div class="revio-card-title" style="margin: 0; font-size: 1.05rem;">{c_name}</div>
                                <div class="revio-card-sub" style="margin: 2px 0 0 0;">
                                    <b>{format_integer(c_count)}</b> reviews • <b>{c_share:.1f}%</b> of total volume
                                </div>
                            </div>
                            <div style="text-align: right;">
                                <span class="revio-pill" style="margin: 0;">★ {c_rating:.2f} / 5.0</span>
                            </div>
                        </div>
                        <div class="revio-bar-track">
                            <div class="revio-bar-fill" style="width: {bar_width:.1f}%;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )