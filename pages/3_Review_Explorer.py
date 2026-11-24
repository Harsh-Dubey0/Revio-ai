import os
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(page_title="Review Explorer - Revio AI", page_icon="🔍", layout="wide")

from src.database.db import get_db_connection
from src.database.models import delete_reviews, get_all_batches
from src.utils.formatting import format_integer, format_number
from src.utils.theme import apply_page_theme

t = apply_page_theme(include_metric_cards=True)

compact_mode = bool(st.session_state.get("revio_compact_cards", False))
card_pad = "14px 16px" if compact_mode else "18px 20px"

# Revio AI Hero Banner
st.markdown(
    """
    <div class="revio-hero">
        <div class="revio-pill">🔍 Revio AI • Review Explorer</div>
        <h1 style="margin: 0 0 6px 0; font-size: 2rem;">Review Explorer & Search</h1>
        <p class="revio-muted" style="margin: 0; font-size: 0.98rem;">
            Browse customer feedback as interactive cards with multi-filter slicing, keyword search, and safe deletion.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# 1. Batch Selection
batches = get_all_batches() or []
batch_options = {"All Batches": None}
for b in batches:
    date_str = (
        b["uploaded_at"].strftime("%Y-%m-%d %H:%M")
        if b.get("uploaded_at") and hasattr(b["uploaded_at"], "strftime")
        else str(b.get("uploaded_at", ""))[:16]
    )
    batch_options[f"{b['filename']} ({date_str})"] = str(b["batch_id"])

default_scope = st.session_state.get("revio_default_batch_scope", "All Batches")
default_batch_idx = 0
if default_scope == "Latest Batch" and len(batch_options) > 1:
    default_batch_idx = 1

st.sidebar.header("Data Scope")
selected_batch_name = st.sidebar.selectbox(
    "Filter by Batch",
    list(batch_options.keys()),
    index=default_batch_idx,
)
selected_batch_id = batch_options[selected_batch_name]

# 2. Fetch Data from Database
try:
    conn = get_db_connection()
    if selected_batch_id:
        query = """
            SELECT r.review_id, r.customer_id, r.review_text, r.rating, r.review_date,
                   r.product_category, r.product_name, r.source,
                   p.sentiment, p.confidence, p.aspect_labels, p.priority, p.emotion
            FROM reviews r
            LEFT JOIN predictions p ON r.review_id = p.review_id
            WHERE r.batch_id = %s
            ORDER BY r.review_date DESC;
        """
        df = pd.read_sql(query, conn, params=(selected_batch_id,))
    else:
        query = """
            SELECT r.review_id, r.customer_id, r.review_text, r.rating, r.review_date,
                   r.product_category, r.product_name, r.source,
                   p.sentiment, p.confidence, p.aspect_labels, p.priority, p.emotion
            FROM reviews r
            LEFT JOIN predictions p ON r.review_id = p.review_id
            ORDER BY r.review_date DESC;
        """
        df = pd.read_sql(query, conn)
    conn.close()
except Exception as e:
    st.error(f"Error connecting to database: {e}")
    df = pd.DataFrame()

if df.empty:
    st.markdown(
        f"""
        <div class="revio-web-card" style="padding: {card_pad};">
            <div class="revio-card-title">No Reviews Found</div>
            <div class="revio-card-sub" style="margin-bottom: 0;">
                No reviews available for the selected batch scope.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # 3. Temporal Preprocessing
    df["review_datetime"] = pd.to_datetime(df["review_date"], errors="coerce")
    df["year"] = df["review_datetime"].dt.year
    df["month"] = df["review_datetime"].dt.month_name()
    df["rating_num"] = pd.to_numeric(df["rating"], errors="coerce").fillna(3).astype(int)

    # 4. Sidebar Filters
    st.sidebar.header("Filter Criteria")

    categories = ["All"] + sorted([str(c) for c in df["product_category"].dropna().unique() if str(c).strip()])
    selected_cat = st.sidebar.selectbox("Product Category", categories)

    sentiments = ["All"] + sorted([str(s) for s in df["sentiment"].dropna().unique() if str(s).strip()])
    selected_sent = st.sidebar.selectbox("Sentiment", sentiments)

    emotions = ["All"] + sorted([str(e) for e in df["emotion"].dropna().unique() if str(e).strip()])
    selected_emotion = st.sidebar.selectbox("Detected Emotion", emotions)

    ratings_range = st.sidebar.slider("Rating Stars", 1, 5, (1, 5))

    valid_years = sorted([int(y) for y in df["year"].dropna().unique()])
    year_options = ["All"] + [str(y) for y in valid_years]
    selected_year = st.sidebar.selectbox("Year", year_options)

    months_list = [
        "All",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    selected_month = st.sidebar.selectbox("Month", months_list)

    min_date_val = df["review_datetime"].dropna().min()
    max_date_val = df["review_datetime"].dropna().max()

    if pd.notnull(min_date_val) and pd.notnull(max_date_val):
        date_range = st.sidebar.date_input(
            "Date Range",
            value=[min_date_val.date(), max_date_val.date()],
            min_value=min_date_val.date(),
            max_value=max_date_val.date(),
        )
    else:
        date_range = []

    # Keyword Search Bar
    search_query = st.text_input(
        "🔍 Full-Text Keyword Search",
        "",
        placeholder="Search keywords across review texts (e.g. battery, delivery, slow, broken)...",
    )

    # 5. Apply Filter Logic
    filtered = df.copy()

    if selected_cat != "All":
        filtered = filtered[filtered["product_category"] == selected_cat]
    if selected_sent != "All":
        filtered = filtered[filtered["sentiment"] == selected_sent]
    if selected_emotion != "All":
        filtered = filtered[filtered["emotion"] == selected_emotion]

    filtered = filtered[
        (filtered["rating_num"] >= ratings_range[0]) & (filtered["rating_num"] <= ratings_range[1])
    ]

    if selected_year != "All":
        filtered = filtered[filtered["year"] == int(selected_year)]

    if selected_month != "All":
        filtered = filtered[filtered["month"] == selected_month]

    if len(date_range) == 2:
        start_d, end_d = date_range[0], date_range[1]
        filtered = filtered[
            (filtered["review_datetime"].dt.date >= start_d)
            & (filtered["review_datetime"].dt.date <= end_d)
        ]

    if search_query.strip():
        filtered = filtered[filtered["review_text"].str.contains(search_query.strip(), case=False, na=False)]

    # 6. Summary Metrics
    total_batch_count = len(df)
    filtered_count = len(filtered)

    c_s1, c_s2, c_s3 = st.columns(3)
    with c_s1:
        st.metric("Total in Batch", format_integer(total_batch_count))
    with c_s2:
        st.metric("Matching Filters", format_integer(filtered_count))
    with c_s3:
        pct = (filtered_count / total_batch_count * 100) if total_batch_count > 0 else 0
        st.metric("Filter Coverage", f"{pct:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="revio-section-head">
            <div>
                <h3 class="revio-section-title">Customer Review Stream ({format_integer(filtered_count)} matches)</h3>
                <p class="revio-section-desc">Each review is shown as a clean card—tick any card checkbox if you want to delete it.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if filtered.empty:
        st.info("No reviews match your current filter criteria.")
    else:
        max_cards = st.slider(
            "Cards per view",
            min_value=5,
            max_value=min(100, len(filtered)),
            value=min(15, len(filtered)),
            step=5,
        )

        selected_ids = []
        visible_df = filtered.head(max_cards)

        for idx, row in visible_df.iterrows():
            r_id = row.get("review_id")
            r_text = str(row.get("review_text") or "")
            r_cat = str(row.get("product_category") or "General")
            r_prod = str(row.get("product_name") or "").strip()
            r_sent = str(row.get("sentiment") or "Neutral")
            r_emo = str(row.get("emotion") or "Neutral")
            r_prio = str(row.get("priority") or "Normal")
            r_aspects = str(row.get("aspect_labels") or "").strip()
            r_rating = int(row.get("rating_num", 3))
            r_conf = format_number(row.get("confidence"), 2)
            r_dt = row.get("review_datetime")
            date_txt = r_dt.strftime("%Y-%m-%d %H:%M") if pd.notnull(r_dt) else "N/A"

            stars_txt = "★" * max(1, min(5, r_rating)) + "☆" * (5 - max(1, min(5, r_rating)))
            prod_line = f" • Product: <b>{r_prod}</b>" if r_prod and r_prod.lower() != "nan" else ""
            aspect_line = (
                f"<div class='revio-muted' style='font-size: 0.82rem; margin-top: 8px;'>🏷️ Aspects: <b>{r_aspects}</b></div>"
                if r_aspects and r_aspects.lower() not in ("", "none", "nan")
                else ""
            )

            col_chk, col_card = st.columns([0.06, 0.94])
            with col_chk:
                st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
                checked = st.checkbox(
                    "Select review",
                    key=f"sel_rev_{r_id}_{idx}",
                    label_visibility="collapsed",
                )
                if checked and r_id is not None:
                    selected_ids.append(r_id)

            with col_card:
                st.markdown(
                    f"""
                    <div class="revio-web-card" style="padding: {card_pad}; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 8px;">
                            <div style="display: flex; flex-wrap: wrap; gap: 6px; align-items: center;">
                                <span class="revio-pill" style="margin: 0;">{r_cat}</span>
                                <span class="revio-pill" style="margin: 0;">{r_sent}</span>
                                <span class="revio-pill" style="margin: 0;">{r_prio}</span>
                            </div>
                            <div class="revio-muted" style="font-size: 0.84rem;">
                                <span style="color: #F59E0B;">{stars_txt}</span> • Emotion: <b>{r_emo}</b> • Conf: <b>{r_conf}</b> • {date_txt}
                            </div>
                        </div>
                        <div style="font-size: 0.96rem; line-height: 1.55;">{r_text}</div>
                        <div class="revio-muted" style="font-size: 0.8rem; margin-top: 8px;">
                            ID: <code>{r_id}</code>{prod_line}
                        </div>
                        {aspect_line}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if selected_ids:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="revio-web-card" style="padding: {card_pad}; border-color: #EF4444;">
                    <div class="revio-card-title">🗑️ Selected Reviews ({len(selected_ids)})</div>
                    <div class="revio-card-sub">Confirm permanent deletion for the selected review records.</div>
                """,
                unsafe_allow_html=True,
            )
            c_del1, c_del2 = st.columns([2, 1])
            with c_del1:
                confirm_del = st.checkbox(
                    "I confirm that I want to permanently delete the selected review(s) from the database."
                )
            with c_del2:
                if st.button("Delete Selected Reviews", type="primary", disabled=not confirm_del, use_container_width=True):
                    deleted_cnt = delete_reviews(selected_ids)
                    st.success(f"Successfully deleted {deleted_cnt} review(s)!")
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        if filtered_count > max_cards:
            st.caption(
                f"Showing {max_cards} of {format_integer(filtered_count)} matching reviews. Increase 'Cards per view' or narrow filters to see more."
            )