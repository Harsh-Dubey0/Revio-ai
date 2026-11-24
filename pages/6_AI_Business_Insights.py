import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

st.set_page_config(page_title="Business Growth Insights • Revio AI", page_icon="✨", layout="wide")

from src.database.db import get_db_connection
from src.database.models import get_all_batches, get_analytics_summary, insert_ai_recommendation
from src.llm.recommendations import generate_ai_recommendations
from src.utils.theme import apply_page_theme

apply_page_theme(include_metric_cards=True)

compact_mode = bool(st.session_state.get("revio_compact_cards", False))
card_pad = "14px 16px" if compact_mode else "18px 20px"


@st.cache_data(ttl=30, show_spinner=False)
def load_uploads() -> list[dict]:
    return get_all_batches() or []


def get_latest_insights(batch_id: str) -> pd.DataFrame:
    query = """
        SELECT *
        FROM ai_recommendations
        WHERE CAST(batch_id AS TEXT) = %s
        ORDER BY generated_at DESC
        LIMIT 5;
    """
    conn = get_db_connection()
    try:
        return pd.read_sql(query, conn, params=(str(batch_id),))
    finally:
        try:
            conn.close()
        except Exception:
            pass


def parse_report_payload(raw_plan) -> tuple[dict, list[dict]]:
    if isinstance(raw_plan, str):
        try:
            parsed = json.loads(raw_plan)
        except json.JSONDecodeError:
            parsed = {}
    else:
        parsed = raw_plan or {}

    if isinstance(parsed, dict):
        items = parsed.get("action_items") or parsed.get("recommendations") or []
        return parsed, (items if isinstance(items, list) else [])
    if isinstance(parsed, list):
        return {}, parsed
    return {}, []


def build_local_fallback_insights(summary: dict) -> dict:
    """Smart local NLP/ML summary fallback if cloud insights or generator return empty."""
    sent = summary.get("sentiment_distribution") or summary.get("sentiment_counts") or {}
    pos = int(sent.get("positive") or sent.get("Positive") or 0)
    neg = int(sent.get("negative") or sent.get("Negative") or 0)
    neu = int(sent.get("neutral") or sent.get("Neutral") or 0)
    total = max(pos + neg + neu, 1)

    pos_pct = round((pos / total) * 100)
    neg_pct = round((neg / total) * 100)

    top_topics = summary.get("top_topics") or summary.get("topics") or []
    topic_names = []
    if isinstance(top_topics, list):
        for t in top_topics[:3]:
            if isinstance(t, dict):
                topic_names.append(str(t.get("topic") or t.get("name") or t.get("label") or ""))
            elif t:
                topic_names.append(str(t))
    topic_str = ", ".join([x for x in topic_names if x]) or "product quality, delivery speed, and support"

    actions = [
        {
            "priority": "High",
            "category": "Customer Retention",
            "action": f"Address recurring feedback around {topic_str} so negative sentiment ({neg_pct}%) drops in upcoming customer waves.",
            "rationale": "Improves repeat purchases and lowers churn risk.",
        },
        {
            "priority": "Medium",
            "category": "Brand Advocacy",
            "action": f"Highlight top-rated experiences driving your {pos_pct}% positive feedback in campaigns and product pages.",
            "rationale": "Turns satisfied customers into organic promoters.",
        },
        {
            "priority": "Low",
            "category": "Continuous Listening",
            "action": "Track weekly shifts in customer mood and top feedback themes after rolling out service improvements.",
            "rationale": "Keeps growth actions aligned with real customer needs.",
        },
    ]

    return {
        "engine": "Smart Local Insights",
        "model": "local-nlp-ml-v1",
        "is_fallback": True,
        "fallback_reason": "Smart Local Mode active",
        "executive_summary": (
            f"Customer feedback shows {pos_pct}% positive and {neg_pct}% negative sentiment across key themes "
            f"like {topic_str}. Focusing on high-friction touchpoints while amplifying what customers already love "
            f"will drive stronger loyalty."
        ),
        "complaints_summary": (
            f"Customers raised concerns mainly around **{topic_str}** ({neg_pct}% negative share). "
            f"Resolving bottlenecks here offers the fastest lift in customer happiness."
        ),
        "positives_summary": (
            f"**{pos_pct}%** of analyzed feedback reflects positive customer sentiment, showing strong core value "
            f"and clear brand resonance."
        ),
        "action_items": actions,
    }


def extract_customer_pulse(summary: dict) -> tuple[str, str, str]:
    if not isinstance(summary, dict) or not summary:
        return "Ready", "—", "—"

    sent = summary.get("sentiment_distribution") or summary.get("sentiment_counts") or summary
    pos = int(sent.get("positive") or sent.get("Positive") or 0)
    neg = int(sent.get("negative") or sent.get("Negative") or 0)
    neu = int(sent.get("neutral") or sent.get("Neutral") or 0)
    total = pos + neg + neu

    if total > 0:
        pos_pct = round((pos / total) * 100)
        neg_pct = round((neg / total) * 100)
        mood = "😊 Mostly Positive" if pos_pct >= 60 else ("😐 Mixed Feedback" if pos_pct >= 40 else "😟 Needs Attention")
        return mood, f"{pos_pct}%", f"{neg_pct}%"

    avg_rating = summary.get("average_rating") or summary.get("avg_rating")
    if avg_rating:
        return f"⭐ {float(avg_rating):.1f} / 5", "—", "—"

    return "Analyzed", "—", "—"


def build_executive_brief(dataset_name: str, mode_label: str, rec: pd.Series, items: list[dict]) -> str:
    lines = [
        f"# Executive Growth Brief — {dataset_name}",
        f"**Insights Mode:** {mode_label}",
        "",
        "## Overview",
        str(rec.get("executive_summary") or ""),
        "",
        "## Top Customer Concerns",
        str(rec.get("complaints_summary") or "No major concerns flagged."),
        "",
        "## What Customers Love",
        str(rec.get("positives_summary") or "No standout positive themes recorded yet."),
        "",
        "## Recommended Action Roadmap",
    ]
    for idx, item in enumerate(items, start=1):
        prio = item.get("priority", "Medium")
        cat = item.get("category", "Customer Experience")
        act = item.get("action", "")
        lines.append(f"{idx}. [{prio} Impact] {cat}: {act}")
    return "\n".join(lines)


# Revio AI Hero Banner
st.markdown(
    """
    <div class="revio-hero">
        <div class="revio-pill">✨ Revio AI • Growth Intelligence</div>
        <h1 style="margin: 0 0 6px 0; font-size: 2rem;">Business Growth Insights</h1>
        <p class="revio-muted" style="margin: 0; font-size: 0.98rem;">
            Smart recommendations powered by customer language & sentiment patterns—with automatic Smart Local Mode if cloud AI is unavailable.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

uploads = load_uploads()
if not uploads:
    st.markdown(
        f"""
        <div class="revio-web-card" style="padding: {card_pad};">
            <div class="revio-card-title">No customer reviews uploaded yet</div>
            <div class="revio-card-sub" style="margin-bottom: 0;">
                Upload a customer review collection first so we can uncover trends and growth recommendations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# Dataset Selector + Insights Mode + Generate Button
dataset_labels = {}
dataset_ids = []
for item in uploads:
    bid = str(item["batch_id"])
    dt_raw = item.get("uploaded_at")
    date_str = dt_raw.strftime("%d %b %Y") if hasattr(dt_raw, "strftime") else str(dt_raw or "")[:10]
    clean_name = str(item.get("filename", "Customer Reviews")).rsplit(".", 1)[0].replace("_", " ").title()
    row_cnt = int(item.get("row_count") or item.get("total_rows") or 0)
    dataset_labels[bid] = f"{clean_name}  •  {row_cnt:,} reviews  ({date_str})"
    dataset_ids.append(bid)

default_scope = st.session_state.get("revio_default_batch_scope", "All Batches")
default_idx = 0 if (default_scope == "Latest Batch" or not dataset_ids) else 0

st.markdown(
    f"""
    <div class="revio-web-card" style="padding: {card_pad};">
        <div class="revio-card-title">Collection & Generation Controls</div>
        <div class="revio-card-sub">Select a review batch, toggle Smart Local Mode if needed, and generate growth recommendations.</div>
    """,
    unsafe_allow_html=True,
)
top_left, top_mid, top_right = st.columns([2.8, 1.5, 1.5], vertical_alignment="bottom")
with top_left:
    selected_id = st.selectbox(
        "Choose Review Collection",
        options=dataset_ids,
        index=default_idx,
        format_func=lambda x: dataset_labels[x],
    )
with top_mid:
    use_local_mode = st.toggle(
        "Smart Local Mode",
        value=False,
        help="Turn on to run locally without external AI service, or let it switch automatically if AI service is down.",
    )
with top_right:
    generate_clicked = st.button("✨ Generate Insights", type="primary", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# Fetch NLP/ML Analytics Summary
try:
    batch_summary = get_analytics_summary(batch_id=selected_id) or {}
except Exception:
    batch_summary = {}

selected_record = next((u for u in uploads if str(u["batch_id"]) == selected_id), {})
total_reviews = int(selected_record.get("row_count") or selected_record.get("total_rows") or 0)
mood_label, happy_pct, concern_pct = extract_customer_pulse(batch_summary)

# Customer Pulse Row
p1, p2, p3, p4 = st.columns(4)
p1.metric("Reviews Analyzed", f"{total_reviews:,}")
p2.metric("Overall Customer Mood", mood_label)
p3.metric("Happy Customers", happy_pct)
p4.metric("Pain-Point Share", concern_pct)

# Generate Insights (AI Cloud with Smart Local Fallback)
if generate_clicked:
    spinner_msg = (
        "Running Smart Local NLP & ML analysis..."
        if use_local_mode
        else "Analyzing customer feedback & crafting growth insights..."
    )
    with st.spinner(spinner_msg):
        insights = None
        used_local = use_local_mode

        try:
            candidate = generate_ai_recommendations(batch_summary, force_local=use_local_mode)
            if isinstance(candidate, dict) and (
                candidate.get("executive_summary")
                or candidate.get("complaints_summary")
                or candidate.get("positives_summary")
                or candidate.get("action_plan")
                or candidate.get("action_items")
            ):
                insights = candidate
                used_local = bool(candidate.get("is_fallback") or use_local_mode)
            else:
                insights = build_local_fallback_insights(batch_summary)
                used_local = True
        except Exception:
            insights = build_local_fallback_insights(batch_summary)
            used_local = True

        try:
            insert_ai_recommendation(selected_id, insights)
        except Exception:
            pass

        if used_local:
            st.toast("Insights ready using Smart Local Mode!", icon="⚡")
        else:
            st.toast("Insights ready using AI Cloud Insights!", icon="✨")
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# Load Latest Report
try:
    df_recs = get_latest_insights(selected_id)
except Exception:
    df_recs = pd.DataFrame()

if df_recs.empty:
    st.markdown(
        f"""
        <div class="revio-web-card" style="padding: {card_pad};">
            <div class="revio-card-title">Ready to Discover Customer Growth Insights</div>
            <div class="revio-card-sub" style="margin-bottom: 0;">
                Click <b>Generate Insights</b> above to turn your customer feedback patterns into clear actions.
                Use <b>Smart Local Mode</b> anytime if you want instant local analysis without cloud AI.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

rec = df_recs.iloc[0]
plan_meta, action_items = parse_report_payload(rec.get("action_plan"))
is_local_report = bool(
    plan_meta.get("is_fallback")
    or "local" in str(plan_meta.get("engine", "")).lower()
    or "local" in str(plan_meta.get("model", "")).lower()
)
mode_badge = "⚡ Smart Local Mode" if is_local_report else "✨ AI Cloud Insights"

# Executive Brief Header + Mode Pill + Download
st.markdown(
    """
    <div class="revio-section-head">
        <div>
            <h3 class="revio-section-title">📌 Executive Growth Brief</h3>
            <p class="revio-section-desc">High-level summary of customer sentiment, friction points, and opportunities.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

brief_l, brief_m, brief_r = st.columns([3.5, 1.5, 1.4], vertical_alignment="center")
with brief_l:
    st.markdown(f"<span class='revio-pill' style='margin: 0;'>{mode_badge}</span>", unsafe_allow_html=True)
with brief_m:
    st.caption(f"Collection: **{dataset_labels[selected_id].split('  •  ')[0]}**")
with brief_r:
    brief_md = build_executive_brief(dataset_labels[selected_id], mode_badge, rec, action_items)
    st.download_button(
        "⬇ Download Brief",
        data=brief_md.encode("utf-8"),
        file_name="Revio_Executive_Growth_Brief.md",
        mime="text/markdown",
        use_container_width=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

if is_local_report:
    st.info("Generated via **Smart Local Mode** using your customer feedback & sentiment patterns—no external AI service required.")

exec_text = str(rec.get("executive_summary") or "").strip()
st.markdown(
    f"""
    <div class="revio-web-card" style="padding: {card_pad};">
        <div class="revio-card-title">Executive Overview</div>
        <div style="font-size: 0.96rem; line-height: 1.6;">
            {exec_text if exec_text else "Your executive summary is ready below."}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Customer Friction & Highlights
voc_left, voc_right = st.columns(2, gap="large")
with voc_left:
    st.markdown(
        f"""
        <div class="revio-web-card" style="padding: {card_pad}; height: 100%;">
            <div class="revio-card-title">🚨 Top Customer Friction Points</div>
            <div class="revio-card-sub">Key issues dragging down satisfaction.</div>
            <div style="font-size: 0.95rem; line-height: 1.55;">
                {str(rec.get("complaints_summary") or "No major customer complaints detected.")}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with voc_right:
    st.markdown(
        f"""
        <div class="revio-web-card" style="padding: {card_pad}; height: 100%;">
            <div class="revio-card-title">🌟 What Customers Love Most</div>
            <div class="revio-card-sub">Positive drivers worth highlighting and scaling.</div>
            <div style="font-size: 0.95rem; line-height: 1.55;">
                {str(rec.get("positives_summary") or "No standout positive themes recorded yet.")}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Prioritized Growth Roadmap
st.markdown(
    """
    <div class="revio-section-head">
        <div>
            <h3 class="revio-section-title">🚀 Recommended Growth Roadmap</h3>
            <p class="revio-section-desc">Actionable steps ranked by business impact so your team knows what to tackle first.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if action_items:
    filter_choice = st.selectbox(
        "Filter by Impact",
        options=["All Recommendations", "High Impact", "Quick Wins", "Long-Term"],
        index=0,
    )

    priority_map = {
        "High Impact": "high",
        "Quick Wins": "medium",
        "Long-Term": "low",
    }

    filtered_items = [
        item
        for item in action_items
        if filter_choice == "All Recommendations"
        or str(item.get("priority", "Medium")).lower() == priority_map.get(filter_choice, "")
    ]

    if not filtered_items:
        st.info(f"No recommendations under **{filter_choice}** for this report.")
    else:
        for item in filtered_items:
            raw_prio = str(item.get("priority") or "Medium").title()
            badge_label = {
                "High": "🔥 High Impact",
                "Medium": "⚡ Quick Win",
                "Low": "🌱 Long-Term Improvement",
            }.get(raw_prio, f"✨ {raw_prio} Priority")

            focus_area = str(item.get("category") or "Customer Experience")
            recommendation_text = str(item.get("action") or "").strip()
            expected_outcome = str(item.get("rationale") or item.get("impact") or "").strip()
            outcome_html = (
                f"<div class='revio-muted' style='font-size: 0.85rem; margin-top: 8px;'>Expected Outcome: <b>{expected_outcome}</b></div>"
                if expected_outcome
                else ""
            )

            st.markdown(
                f"""
                <div class="revio-web-card" style="padding: {card_pad}; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 8px;">
                        <span class="revio-pill" style="margin: 0;">{badge_label}</span>
                        <span class="revio-muted" style="font-size: 0.86rem;">Focus Area: <b>{focus_area}</b></span>
                    </div>
                    <div style="font-size: 0.96rem; line-height: 1.55;">{recommendation_text}</div>
                    {outcome_html}
                </div>
                """,
                unsafe_allow_html=True,
            )
else:
    st.markdown(
        f"""
        <div class="revio-web-card" style="padding: {card_pad};">
            <div class="revio-card-title">No Action Steps Yet</div>
            <div class="revio-card-sub" style="margin-bottom: 0;">
                No action steps were generated for this collection yet. Click <b>Generate Insights</b> above to refresh.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )