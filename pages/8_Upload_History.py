import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

st.set_page_config(page_title="Upload History • Revio AI", page_icon="📜", layout="wide")

from src.database.models import delete_upload_batch, get_all_batches
from src.utils.theme import apply_page_theme

apply_page_theme(include_metric_cards=True)

compact_global = bool(st.session_state.get("revio_compact_cards", False))
card_pad = "14px 16px" if compact_global else "18px 20px"


@st.cache_data(ttl=30, show_spinner=False)
def load_batches_df() -> pd.DataFrame:
    raw = get_all_batches() or []
    if not raw:
        return pd.DataFrame()

    df = pd.DataFrame(raw)
    df["_raw_record"] = raw

    if "uploaded_at" in df.columns:
        df["uploaded_dt"] = pd.to_datetime(df["uploaded_at"], errors="coerce", utc=True)
        df = df.sort_values("uploaded_dt", ascending=False, na_position="last")
        df["uploaded_display"] = df["uploaded_dt"].dt.strftime("%b %d, %Y · %H:%M UTC").fillna("—")
        df["upload_date"] = df["uploaded_dt"].dt.date
    else:
        df["uploaded_dt"] = pd.NaT
        df["uploaded_display"] = "—"
        df["upload_date"] = pd.NaT

    row_col = "row_count" if "row_count" in df.columns else ("total_rows" if "total_rows" in df.columns else None)
    if row_col:
        df["row_count"] = pd.to_numeric(df[row_col], errors="coerce").fillna(0).astype(int)
    else:
        df["row_count"] = 0

    df["status"] = df.get("status", "unknown").fillna("unknown").astype(str).str.lower()

    badge_map = {
        "completed": "🟢 Completed",
        "processing": "🟡 Processing",
        "queued": "🟡 Queued",
        "pending": "🟡 Pending",
        "failed": "🔴 Failed",
        "error": "🔴 Error",
    }
    df["status_badge"] = df["status"].map(badge_map).fillna("⚪ " + df["status"].str.title())

    return df.reset_index(drop=True)


@st.dialog("Confirm Batch Deletion")
def confirm_delete_dialog(batch_records: list[dict]):
    count = len(batch_records)
    total_rows = sum(int(b.get("row_count", 0)) for b in batch_records)

    st.warning(
        f"Permanently delete **{count} batch{'es' if count > 1 else ''}** "
        f"(**{total_rows:,} rows**) and cascade-delete all linked reviews, predictions, and recommendations?"
    )

    st.markdown(f"<div class='revio-web-card' style='padding: {card_pad};'>", unsafe_allow_html=True)
    for b in batch_records[:6]:
        st.markdown(
            f"• **{b.get('filename', 'untitled')}** · `{b['batch_id']}` — "
            f"*{int(b.get('row_count', 0)):,} rows* (`{b.get('status', 'unknown')}`)"
        )
    if count > 6:
        st.caption(f"+ {count - 6} additional batches selected")
    st.markdown("</div>", unsafe_allow_html=True)

    confirm_text = st.text_input("Type `DELETE` to proceed", placeholder="DELETE")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Cancel", use_container_width=True):
            st.rerun()
    with c2:
        if st.button(
            "Confirm Delete",
            type="primary",
            disabled=(confirm_text.strip() != "DELETE"),
            use_container_width=True,
        ):
            deleted, failures = 0, []
            for b in batch_records:
                bid = str(b["batch_id"])
                try:
                    if delete_upload_batch(bid):
                        deleted += 1
                    else:
                        failures.append(f"{bid} (not found)")
                except Exception as exc:
                    failures.append(f"{bid} ({exc})")

            load_batches_df.clear()
            if deleted:
                st.toast(f"Removed {deleted} batch{'es' if deleted > 1 else ''}.", icon="🗑️")
            if failures:
                st.error("Failed to remove: " + ", ".join(failures))
            st.rerun()


# Revio AI Hero Banner
st.markdown(
    """
    <div class="revio-hero">
        <div class="revio-pill">📜 Revio AI • Audit & Ingestion Log</div>
        <h1 style="margin: 0 0 6px 0; font-size: 2rem;">Upload History</h1>
        <p class="revio-muted" style="margin: 0; font-size: 0.98rem;">
            Inspect uploaded review batches as interactive cards with status filters, CSV export, and cascade delete.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

h_l, h_m, h_r = st.columns([5.2, 1.4, 1.2], vertical_alignment="center")
with h_l:
    st.markdown("<span class='revio-pill' style='margin: 0;'>Batch Lifecycle Console</span>", unsafe_allow_html=True)
with h_m:
    live_poll = st.toggle("Live (30s)", value=False, help="Automatically refresh batch history every 30 seconds")
with h_r:
    if st.button("↻ Refresh", use_container_width=True):
        load_batches_df.clear()
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)


@st.fragment(run_every=30 if live_poll else None)
def render_history_console():
    df = load_batches_df()

    if df.empty:
        st.markdown(
            f"""
            <div class="revio-web-card" style="padding: {card_pad};">
                <div class="revio-card-title">No upload batches recorded</div>
                <div class="revio-card-sub" style="margin-bottom: 0;">
                    Upload a review file from <b>Upload Reviews</b> to start tracking batch runs and row counts.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    total_batches = len(df)
    total_rows = int(df["row_count"].sum())
    latest_rows = int(df.iloc[0]["row_count"]) if total_batches >= 1 else 0
    prev_rows = int(df.iloc[1]["row_count"]) if total_batches >= 2 else None
    latest_delta = f"{latest_rows - prev_rows:+,} vs prev" if prev_rows is not None else None
    completed_rate = round((df["status"] == "completed").mean() * 100, 1)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Batches", f"{total_batches:,}")
    m2.metric("Total Reviews", f"{total_rows:,}")
    m3.metric("Latest Batch Size", f"{latest_rows:,} rows", delta=latest_delta)
    m4.metric("Success Rate", f"{completed_rate}%")

    st.markdown("<br>", unsafe_allow_html=True)

    valid_dates = df["upload_date"].dropna()
    if not valid_dates.empty and valid_dates.nunique() > 1:
        with st.expander("📊 Ingestion Analytics", expanded=False):
            ch1, ch2 = st.columns([2, 1])
            with ch1:
                st.markdown("**Daily Ingested Rows**")
                daily = (
                    df.dropna(subset=["upload_date"])
                    .groupby("upload_date", as_index=False)["row_count"]
                    .sum()
                    .sort_values("upload_date")
                )
                st.area_chart(daily, x="upload_date", y="row_count", height=180)
            with ch2:
                st.markdown("**Batches by Status**")
                status_counts = df["status"].value_counts().reset_index()
                status_counts.columns = ["status", "batches"]
                st.bar_chart(status_counts, x="status", y="batches", height=180)

    # Filter Controls (Direct widgets without empty wrapper box)
    qp_status = st.query_params.get("status", "All")
    qp_q = st.query_params.get("q", "")

    status_choices = ["All"] + sorted(df["status"].str.title().unique().tolist())
    default_status = qp_status if qp_status in status_choices else "All"

    f1, f2, f3 = st.columns([2.6, 2.2, 1.2], vertical_alignment="center")
    with f1:
        active_status = st.pills("Status", status_choices, default=default_status, label_visibility="collapsed")
    with f2:
        search_q = st.text_input(
            "Search",
            value=qp_q,
            placeholder="🔍 Filename or batch ID...",
            label_visibility="collapsed",
        )
    with f3:
        export_cols = [c for c in ["batch_id", "filename", "row_count", "status", "uploaded_at"] if c in df.columns]
        st.download_button(
            "⬇ CSV",
            data=df[export_cols].to_csv(index=False).encode("utf-8"),
            file_name="revio_upload_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

    v1, v2 = st.columns([2.4, 1.6], vertical_alignment="center")
    with v1:
        if not valid_dates.empty:
            min_d, max_d = valid_dates.min(), valid_dates.max()
            date_range = st.date_input(
                "Date Range",
                value=(min_d, max_d),
                min_value=min_d,
                max_value=max_d,
                label_visibility="collapsed",
            )
        else:
            date_range = None
    with v2:
        compact_rows = st.toggle("Compact Cards", value=compact_global)

    # Sync URL query params cleanly
    if active_status and active_status != "All":
        st.query_params["status"] = active_status
    elif "status" in st.query_params:
        del st.query_params["status"]

    if search_q.strip():
        st.query_params["q"] = search_q.strip()
    elif "q" in st.query_params:
        del st.query_params["q"]

    # Apply Filtering Pipeline
    filtered = df.copy()
    if active_status and active_status != "All":
        filtered = filtered[filtered["status"] == active_status.lower()]

    if search_q.strip():
        q = search_q.strip().lower()
        mask = (
            filtered["filename"].astype(str).str.lower().str.contains(q, na=False)
            | filtered["batch_id"].astype(str).str.lower().str.contains(q, na=False)
        )
        filtered = filtered[mask]

    if isinstance(date_range, tuple) and len(date_range) == 2 and "upload_date" in filtered.columns:
        start_d, end_d = date_range
        filtered = filtered[
            filtered["upload_date"].isna()
            | ((filtered["upload_date"] >= start_d) & (filtered["upload_date"] <= end_d))
        ]

    filtered = filtered.reset_index(drop=True)

    st.markdown(
        f"""
        <div class="revio-section-head" style="margin-top: 14px;">
            <div>
                <h3 class="revio-section-title">Batch Cards ({len(filtered)} of {total_batches})</h3>
                <p class="revio-section-desc">Tick checkboxes to select batches for deletion, or expand payload to inspect raw metadata.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if filtered.empty:
        st.markdown(
            f"""
            <div class="revio-web-card" style="padding: {card_pad};">
                <div class="revio-card-title">No Matching Batches</div>
                <div class="revio-card-sub" style="margin-bottom: 0;">
                    Try clearing your search keyword, status pill, or date filter.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    max_row_val = max(int(df["row_count"].max()), 1)
    active_pad = "14px 16px" if compact_rows else card_pad
    selected_records = []

    for idx, row in filtered.iterrows():
        bid = str(row.get("batch_id", ""))
        fname = str(row.get("filename", "untitled"))
        rows_cnt = int(row.get("row_count", 0))
        badge = str(row.get("status_badge", "⚪ Unknown"))
        up_disp = str(row.get("uploaded_display", "—"))
        pct_bar = min(100.0, max(2.0, (rows_cnt / max_row_val) * 100.0))

        col_chk, col_card = st.columns([0.05, 0.95])
        with col_chk:
            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
            is_checked = st.checkbox(
                "Select batch",
                key=f"batch_sel_{bid}_{idx}",
                label_visibility="collapsed",
            )
            if is_checked:
                selected_records.append(row.to_dict())

        with col_card:
            st.markdown(
                f"""
                <div class="revio-web-card" style="padding: {active_pad}; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 8px;">
                        <div>
                            <div class="revio-card-title" style="margin: 0;">📄 {fname}</div>
                            <div class="revio-card-sub" style="margin: 2px 0 0 0;">
                                Batch ID: <code>{bid}</code> • Uploaded: <b>{up_disp}</b>
                            </div>
                        </div>
                        <span class="revio-pill" style="margin: 0;">{badge}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 4px;">
                        <span class="revio-muted">Ingested Volume</span>
                        <b>{rows_cnt:,} rows</b>
                    </div>
                    <div class="revio-bar-track">
                        <div class="revio-bar-fill" style="width: {pct_bar:.1f}%;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Selection & Delete Bar
    if selected_records:
        sel_row_sum = sum(int(r.get("row_count", 0)) for r in selected_records)
        st.markdown("<br>", unsafe_allow_html=True)
        a_left, a_right = st.columns([4.5, 1.5], vertical_alignment="center")
        with a_left:
            st.markdown(f"**{len(selected_records)}** batch(es) selected · **{sel_row_sum:,}** total rows")
        with a_right:
            if st.button(f"🗑️ Delete ({len(selected_records)})", type="primary", use_container_width=True):
                confirm_delete_dialog(selected_records)

    # Single Batch Raw Payload Inspector
    if len(selected_records) == 1:
        single = selected_records[0]
        raw_meta = {k: v for k, v in (single.get("_raw_record") or {}).items() if k != "_raw_record"}
        with st.expander(f"🔎 Raw Batch Payload · {single.get('batch_id', '')}", expanded=False):
            st.json(raw_meta)


render_history_console()