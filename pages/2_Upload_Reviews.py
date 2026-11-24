import os
import sys
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(page_title="Revio AI - Upload Reviews", page_icon="📤", layout="wide")

from src.inference.predictor import ReviewPredictor
from src.utils.formatting import format_integer
from src.utils.theme import apply_page_theme

db_import_error = None
try:
    from src.database.models import (
        create_upload_batch,
        get_all_batches,
        insert_review_and_prediction,
        update_batch_status,
    )
except Exception as exc:
    db_import_error = str(exc)
    create_upload_batch = None
    get_all_batches = None
    insert_review_and_prediction = None
    update_batch_status = None


def _normalize_sentiment(val: Any, rating_val: int | None = None) -> str:
    s = str(val or "").strip().lower()
    if s in {"positive", "pos", "4", "5"}:
        return "Positive"
    if s in {"negative", "neg", "0", "1", "2"}:
        return "Negative"
    if s in {"neutral", "neu", "3"}:
        return "Neutral"
    if rating_val is not None:
        if rating_val >= 4:
            return "Positive"
        if rating_val <= 2:
            return "Negative"
        return "Neutral"
    return "Neutral"


def _normalize_confidence(val: Any, default: float = 0.78) -> float:
    try:
        num = float(val)
        if num > 1.0:
            num = num / 100.0 if num <= 100.0 else 1.0
        return round(max(0.01, min(1.0, num)), 4)
    except Exception:
        return default


def _fallback_predict_batch(texts: List[str]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    pos_words = {
        "great", "good", "awesome", "love", "loved", "fast",
        "clean", "best", "comfortable", "smooth", "nice", "happy",
    }
    neg_words = {
        "bad", "worst", "late", "slow", "broken", "cracked", "leaked",
        "frustrated", "frustrating", "poor", "issue", "issues",
        "disappointed", "peeling", "snapped", "damaged",
    }
    for txt in texts:
        low = txt.lower()
        p_hits = sum(1 for w in pos_words if w in low)
        n_hits = sum(1 for w in neg_words if w in low)
        if n_hits > p_hits:
            sentiment = "Negative"
            emotion = "Frustrated"
            priority = "High"
            conf = min(0.92, 0.68 + 0.06 * n_hits)
        elif p_hits > n_hits:
            sentiment = "Positive"
            emotion = "Satisfied"
            priority = "Low"
            conf = min(0.94, 0.70 + 0.06 * p_hits)
        else:
            sentiment = "Neutral"
            emotion = "Neutral"
            priority = "Medium"
            conf = 0.65
        results.append(
            {
                "sentiment": sentiment,
                "confidence": round(conf, 2),
                "aspect_labels": "General",
                "priority": priority,
                "emotion": emotion,
            }
        )
    return results


@st.cache_resource(show_spinner=False)
def _get_predictor() -> ReviewPredictor | None:
    try:
        return ReviewPredictor()
    except Exception:
        return None


def run_revio_predictions(
    texts: List[str],
    df_clean: pd.DataFrame,
    text_col: str = "review_text",
) -> List[Dict[str, Any]]:
    predictor = _get_predictor()
    if predictor is not None:
        batch_df = df_clean.copy()
        if text_col not in batch_df.columns:
            batch_df[text_col] = texts
        if "review_text" not in batch_df.columns:
            batch_df["review_text"] = batch_df[text_col]

        if hasattr(predictor, "predict_batch") and callable(predictor.predict_batch):
            out = None
            try:
                out = predictor.predict_batch(batch_df)
            except TypeError:
                try:
                    out = predictor.predict_batch(texts)
                except Exception as exc:
                    st.warning(f"Predictor batch error ({exc}); using fallback scorer for this batch.")
            except Exception as exc:
                st.warning(f"Predictor batch error ({exc}); using fallback scorer for this batch.")

            if isinstance(out, pd.DataFrame) and len(out) == len(texts):
                return out.to_dict(orient="records")
            if isinstance(out, list) and len(out) == len(texts):
                return [item if isinstance(item, dict) else {"sentiment": str(item)} for item in out]

        if hasattr(predictor, "predict_dataframe") and callable(predictor.predict_dataframe):
            try:
                res_df = predictor.predict_dataframe(batch_df)
                if isinstance(res_df, pd.DataFrame) and len(res_df) == len(texts):
                    return res_df.to_dict(orient="records")
            except Exception as exc:
                st.warning(f"Predictor dataframe error ({exc}); using fallback scorer.")

        if hasattr(predictor, "predict") and callable(predictor.predict):
            try:
                out = predictor.predict(texts)
                if isinstance(out, list) and len(out) == len(texts):
                    return [item if isinstance(item, dict) else {"sentiment": str(item)} for item in out]
            except Exception as exc:
                st.warning(f"Predictor error ({exc}); using fallback scorer.")

    return _fallback_predict_batch(texts)


apply_page_theme(include_metric_cards=True)

compact_mode = bool(st.session_state.get("revio_compact_cards", False))
card_pad = "14px 16px" if compact_mode else "18px 20px"

st.markdown(
    """
    <div class="revio-hero">
        <div class="revio-pill">📤 Revio AI • Batch Ingestion</div>
        <h1 style="margin: 0 0 6px 0; font-size: 2rem;">Upload & Analyze Reviews</h1>
        <p class="revio-muted" style="margin: 0; font-size: 0.98rem;">
            Import customer review CSV files, run Revio AI sentiment & priority scoring, and save batches to PostgreSQL.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if db_import_error:
    st.error(f"Database models import error: {db_import_error}")

st.sidebar.header("Ingestion Options")
preview_rows = st.sidebar.slider("Preview Row Limit", min_value=5, max_value=50, value=10, step=5)

st.markdown(
    """
    <div class="revio-section-head">
        <div>
            <h3 class="revio-section-title">📁 Select Review Dataset</h3>
            <p class="revio-section-desc">Upload a CSV file containing review text (and optional rating or category columns).</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read CSV file: {exc}")
        df = None

    if df is not None and not df.empty:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Rows", format_integer(len(df)))
        with c2:
            st.metric("Columns", format_integer(len(df.columns)))
        with c3:
            null_cells = int(df.isna().sum().sum())
            st.metric("Missing Cells", format_integer(null_cells))

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="revio-web-card" style="padding: {card_pad};">
                <div class="revio-card-title">Column Mapping</div>
                <div class="revio-card-sub">Map your CSV columns to Revio AI review fields before running batch scoring.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        cols_list = list(df.columns)
        none_option = ["— None —"]

        m1, m2, m3 = st.columns(3)
        with m1:
            guessed_text = next(
                (c for c in cols_list if any(k in c.lower() for k in ("review", "text", "comment", "feedback", "body"))),
                cols_list[0],
            )
            text_col = st.selectbox("Review Text Column *", options=cols_list, index=cols_list.index(guessed_text))

        with m2:
            guessed_rating = next(
                (c for c in cols_list if any(k in c.lower() for k in ("rating", "star", "score"))),
                None,
            )
            rating_opts = none_option + cols_list
            rating_idx = rating_opts.index(guessed_rating) if guessed_rating in rating_opts else 0
            rating_col = st.selectbox("Rating Column (Optional)", options=rating_opts, index=rating_idx)

        with m3:
            guessed_cat = next(
                (c for c in cols_list if any(k in c.lower() for k in ("category", "product", "type", "dept"))),
                None,
            )
            cat_opts = none_option + cols_list
            cat_idx = cat_opts.index(guessed_cat) if guessed_cat in cat_opts else 0
            cat_col = st.selectbox("Category Column (Optional)", options=cat_opts, index=cat_idx)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="revio-web-card" style="padding: {card_pad};">
                <div class="revio-card-title">Dataset Preview (Top {min(preview_rows, len(df))} Rows)</div>
                <div class="revio-card-sub">Quick look at <b>{uploaded_file.name}</b> prior to processing.</div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(df.head(preview_rows), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        run_process = st.button("🚀 Process & Save Batch to Revio AI", type="primary", use_container_width=True)

        if run_process:
            if create_upload_batch is None or insert_review_and_prediction is None:
                st.error(
                    "Database functions are not available in this environment."
                    + (f" Details: {db_import_error}" if db_import_error else "")
                )
            else:
                clean_df = df.copy()
                clean_df[text_col] = clean_df[text_col].fillna("").astype(str).str.strip()
                clean_df = clean_df[clean_df[text_col] != ""].reset_index(drop=True)

                if clean_df.empty:
                    st.warning("No non-empty review text rows found in the selected column.")
                else:
                    batch_id = None
                    saved_count = 0
                    try:
                        with st.status("Processing batch with Revio AI...", expanded=True) as proc_box:
                            st.write("1. Creating upload batch record...")
                            batch_id = create_upload_batch(
                                filename=uploaded_file.name,
                                row_count=len(clean_df),
                            )

                            st.write("2. Running sentiment, emotion, and priority predictions...")
                            texts = clean_df[text_col].tolist()
                            preds = run_revio_predictions(texts, clean_df, text_col=text_col)

                            st.write("3. Saving reviews and predictions to PostgreSQL...")
                            progress_bar = st.progress(0)

                            for idx, row in clean_df.iterrows():
                                review_text = row[text_col]
                                raw_id = row["review_id"] if "review_id" in clean_df.columns and pd.notna(row["review_id"]) else None
                                rev_id = str(raw_id).strip() if raw_id and str(raw_id).strip() else f"{batch_id[:8]}-{idx + 1}"

                                raw_cust = row["customer_id"] if "customer_id" in clean_df.columns and pd.notna(row["customer_id"]) else None
                                cust_id = str(raw_cust).strip() if raw_cust and str(raw_cust).strip() else f"CUST-{idx + 1}"

                                raw_date = row["review_date"] if "review_date" in clean_df.columns and pd.notna(row["review_date"]) else None
                                rev_date = str(raw_date).strip() if raw_date and str(raw_date).strip() else None

                                raw_prod = row["product_name"] if "product_name" in clean_df.columns and pd.notna(row["product_name"]) else None
                                prod_name = str(raw_prod).strip() if raw_prod and str(raw_prod).strip() else "Standard Product"

                                raw_src = row["source"] if "source" in clean_df.columns and pd.notna(row["source"]) else None
                                src_val = str(raw_src).strip() if raw_src and str(raw_src).strip() else "CSV Upload"

                                raw_rating = row[rating_col] if rating_col != "— None —" else None
                                try:
                                    rating_val = int(round(float(raw_rating))) if raw_rating is not None and pd.notna(raw_rating) else 3
                                except Exception:
                                    rating_val = 3

                                cat_val = (
                                    str(row[cat_col]).strip()
                                    if cat_col != "— None —" and pd.notna(row[cat_col]) and str(row[cat_col]).strip()
                                    else "General"
                                )

                                raw_pred = preds[idx] if preds and idx < len(preds) and isinstance(preds[idx], dict) else {}
                                csv_sent = row["sentiment"] if "sentiment" in clean_df.columns and pd.notna(row["sentiment"]) else None
                                sent_val = _normalize_sentiment(
                                    raw_pred.get("sentiment") or raw_pred.get("label") or csv_sent,
                                    rating_val=rating_val,
                                )

                                csv_conf = row["confidence"] if "confidence" in clean_df.columns and pd.notna(row["confidence"]) else None
                                raw_conf = raw_pred.get("confidence") if raw_pred.get("confidence") is not None else raw_pred.get("score", csv_conf)
                                conf_val = _normalize_confidence(raw_conf, default=0.82 if sent_val != "Neutral" else 0.68)

                                asp_from_csv = (
                                    str(row["aspect_labels"]).strip()
                                    if "aspect_labels" in clean_df.columns and pd.notna(row["aspect_labels"]) and str(row["aspect_labels"]).strip()
                                    else None
                                )
                                emo_from_csv = (
                                    str(row["emotion"]).strip()
                                    if "emotion" in clean_df.columns and pd.notna(row["emotion"]) and str(row["emotion"]).strip()
                                    else None
                                )
                                prio_from_csv = (
                                    str(row["priority"]).strip()
                                    if "priority" in clean_df.columns and pd.notna(row["priority"]) and str(row["priority"]).strip()
                                    else None
                                )

                                default_emo = "Satisfied" if sent_val == "Positive" else ("Frustrated" if sent_val == "Negative" else "Neutral")
                                default_prio = "Low" if sent_val == "Positive" else ("High" if sent_val == "Negative" else "Medium")

                                pred_payload = {
                                    "sentiment": sent_val,
                                    "confidence": conf_val,
                                    "aspect_labels": asp_from_csv or raw_pred.get("aspect_labels") or "General",
                                    "priority": prio_from_csv or raw_pred.get("priority") or default_prio,
                                    "emotion": emo_from_csv or raw_pred.get("emotion") or default_emo,
                                }

                                review_payload = {
                                    "review_id": rev_id,
                                    "customer_id": cust_id,
                                    "review_text": review_text,
                                    "rating": rating_val,
                                    "review_date": rev_date,
                                    "product_category": cat_val,
                                    "product_name": prod_name,
                                    "source": src_val,
                                }

                                insert_review_and_prediction(batch_id, review_payload, pred_payload)
                                saved_count += 1
                                progress_bar.progress(min((idx + 1) / len(clean_df), 1.0))

                            if update_batch_status is not None:
                                update_batch_status(batch_id, "completed")

                            proc_box.update(
                                label=f"Batch Completed — {saved_count} Reviews Saved",
                                state="complete",
                                expanded=False,
                            )

                        st.success(f"Successfully processed and saved **{saved_count}** reviews from `{uploaded_file.name}`!")
                    except Exception as exc:
                        if batch_id and update_batch_status is not None:
                            try:
                                update_batch_status(batch_id, "failed")
                            except Exception:
                                pass
                        st.error(f"Batch upload failed: {exc}")
                        st.exception(exc)
else:
    st.markdown(
        f"""
        <div class="revio-web-card" style="padding: {card_pad};">
            <div class="revio-card-title">Waiting for a CSV File</div>
            <div class="revio-card-sub" style="margin-bottom: 0;">
                Drag and drop or browse a <b>.csv</b> review file above to preview columns and run Revio AI batch analysis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if get_all_batches is not None:
    try:
        recent_batches = get_all_batches() or []
    except Exception:
        recent_batches = []

    if recent_batches:
        st.markdown(
            """
            <div class="revio-section-head" style="margin-top: 24px;">
                <div>
                    <h3 class="revio-section-title">📦 Recent Upload Batches</h3>
                    <p class="revio-section-desc">Latest batches recorded in your Revio AI workspace.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for b in recent_batches[:5]:
            b_name = b.get("filename", "Batch")
            b_status = str(b.get("status", "completed")).title()
            b_rows = b.get("row_count", b.get("total_rows", 0))
            b_date = (
                b["uploaded_at"].strftime("%Y-%m-%d %H:%M")
                if b.get("uploaded_at") and hasattr(b["uploaded_at"], "strftime")
                else str(b.get("uploaded_at", ""))[:16]
            )
            st.markdown(
                f"""
                <div class="revio-web-card" style="padding: {card_pad}; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                        <div>
                            <div class="revio-card-title" style="margin: 0;">{b_name}</div>
                            <div class="revio-card-sub" style="margin: 2px 0 0 0;">
                                Uploaded: <b>{b_date}</b> • Rows: <b>{format_integer(int(b_rows or 0))}</b>
                            </div>
                        </div>
                        <span class="revio-pill" style="margin: 0;">{b_status}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )