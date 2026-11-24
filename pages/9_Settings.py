import os
import re
import sys
import time

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.set_page_config(page_title="Revio AI - Settings", page_icon="⚙️", layout="wide")

from src.utils.theme import THEMES, apply_page_theme, get_active_theme_name

try:
    from config.settings import settings
except Exception:
    settings = None

try:
    from src.database.db import test_database_connection
except Exception:
    test_database_connection = None


def test_gemini_api(api_key: str) -> dict:
    if not api_key:
        return {"ok": False, "detail": "GEMINI_API_KEY is not set in environment or settings."}

    try:
        from google import genai  # newer google-genai SDK

        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Reply with OK",
        )
        text = (getattr(resp, "text", "") or "").strip()
        return {"ok": True, "detail": f"Live response received: {text or 'OK'}"}
    except ImportError:
        pass
    except Exception as exc:
        return {"ok": False, "detail": f"Gemini SDK error: {exc}"}

    try:
        import google.generativeai as genai_legacy  # fallback if older SDK is installed

        genai_legacy.configure(api_key=api_key)
        model = genai_legacy.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content("Reply with OK")
        text = (getattr(resp, "text", "") or "").strip()
        return {"ok": True, "detail": f"Live response received: {text or 'OK'}"}
    except Exception as exc:
        return {"ok": False, "detail": f"Gemini API error: {exc}"}


apply_page_theme(include_metric_cards=True)

if "revio_default_model_version" not in st.session_state:
    st.session_state["revio_default_model_version"] = "v1.0.0"
if "revio_default_batch_scope" not in st.session_state:
    st.session_state["revio_default_batch_scope"] = "All Batches"
if "revio_compact_cards" not in st.session_state:
    st.session_state["revio_compact_cards"] = False

st.markdown(
    """
    <div class="revio-hero">
        <div class="revio-pill">⚙️ Revio AI • Settings</div>
        <h1 style="margin: 0 0 6px 0; font-size: 2rem;">Revio AI Settings</h1>
        <p class="revio-muted" style="margin: 0; font-size: 0.98rem;">
            Choose your theme, set default workspace options, and check database status.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# 1. Appearance Theme
st.markdown(
    """
    <div class="revio-section-head">
        <div>
            <h3 class="revio-section-title">🎨 Theme</h3>
            <p class="revio-section-desc">Switch between Dark, Midnight, and Light mode across Revio AI.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

theme_options = list(THEMES.keys())
current_theme = get_active_theme_name()

col_t1, col_t2 = st.columns([1.5, 1])
with col_t1:
    selected_theme = st.selectbox(
        "Appearance Theme",
        options=theme_options,
        index=theme_options.index(current_theme),
    )
    if selected_theme != current_theme:
        st.session_state["revio_theme"] = selected_theme
        st.rerun()

with col_t2:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    if st.button("Reset to Dark", use_container_width=True):
        if st.session_state.get("revio_theme") != "Dark":
            st.session_state["revio_theme"] = "Dark"
            st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# 2. Workspace Preferences
st.markdown(
    """
    <div class="revio-section-head">
        <div>
            <h3 class="revio-section-title">🎛️ Workspace Preferences</h3>
            <p class="revio-section-desc">Default behavior for review batches and dashboard cards.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

p1, p2, p3 = st.columns(3)
with p1:
    model_choices = ["v1.0.0", "v1.1.0-beta", "v2.0.0-experimental"]
    curr_model = st.session_state["revio_default_model_version"]
    st.session_state["revio_default_model_version"] = st.selectbox(
        "Default Model Version",
        options=model_choices,
        index=model_choices.index(curr_model) if curr_model in model_choices else 0,
    )

with p2:
    scope_choices = ["All Batches", "Latest Batch"]
    curr_scope = st.session_state["revio_default_batch_scope"]
    st.session_state["revio_default_batch_scope"] = st.selectbox(
        "Default Data View",
        options=scope_choices,
        index=scope_choices.index(curr_scope) if curr_scope in scope_choices else 0,
    )

with p3:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    st.session_state["revio_compact_cards"] = st.toggle(
        "Compact Card Spacing",
        value=bool(st.session_state["revio_compact_cards"]),
    )

st.markdown("<br>", unsafe_allow_html=True)

# 3. Database & Connection Check
st.markdown(
    """
    <div class="revio-section-head">
        <div>
            <h3 class="revio-section-title">🗄️ Database & System Diagnostics</h3>
            <p class="revio-section-desc">Run a live health check on PostgreSQL, core tables, and AI service configuration.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

gemini_key = getattr(settings, "GEMINI_API_KEY", None) or os.getenv("GEMINI_API_KEY") or ""
db_ready = bool(getattr(settings, "DATABASE_URL", None) or os.getenv("DATABASE_URL"))
ai_ready = bool(gemini_key)

s1, s2, s3 = st.columns([1, 1, 1.4])
with s1:
    st.metric("PostgreSQL", "Ready" if db_ready else "Not Configured")
with s2:
    st.metric("AI Insights", "Gemini Ready" if ai_ready else "Local Engine")
with s3:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    run_db_test = st.button("Run Full Connection Diagnostic", use_container_width=True)

if "revio_last_diag" not in st.session_state:
    st.session_state["revio_last_diag"] = None

if run_db_test:
    diag = {
        "db_url": db_ready,
        "ai_key": ai_ready,
        "connected": False,
        "pg_version": "Unknown",
        "raw_message": "",
        "table_counts": {},
        "gemini_status": None,
    }

    with st.status("Running Revio AI System Diagnostic...", expanded=True) as status_box:
        st.write("1. Checking environment variables & API keys...")
        time.sleep(0.2)
        st.write(
            f"   • `DATABASE_URL`: {'Configured' if db_ready else 'Missing'} | "
            f"`GEMINI_API_KEY`: {'Configured' if ai_ready else 'Missing (Local fallback active)'}"
        )

        st.write("2. Testing PostgreSQL TCP/driver connectivity...")
        time.sleep(0.2)
        if test_database_connection is None:
            diag["raw_message"] = "Database connection helper not available."
            st.error(diag["raw_message"])
        else:
            res = test_database_connection()
            diag["connected"] = bool(res.get("success"))
            diag["raw_message"] = str(res.get("message", ""))

            match = re.search(r"PostgreSQL\s+([0-9]+(?:\.[0-9]+)?)", diag["raw_message"])
            if match:
                diag["pg_version"] = f"PostgreSQL {match.group(1)}"

            if diag["connected"]:
                st.write(f"   • Connection established ({diag['pg_version']}).")
            else:
                st.write("   • Connection failed.")

        st.write("3. Inspecting core tables & stored records...")
        time.sleep(0.2)
        if diag["connected"]:
            try:
                from src.database.db import get_db_connection

                with get_db_connection() as conn:
                    with conn.cursor() as cur:
                        for tbl in ("upload_batches", "reviews", "predictions"):
                            try:
                                cur.execute(f"SELECT COUNT(*) FROM {tbl}")
                                row = cur.fetchone()
                                diag["table_counts"][tbl] = int(row[0]) if row else 0
                            except Exception:
                                diag["table_counts"][tbl] = "N/A"
            except Exception:
                pass

        st.write("4. Pinging Gemini API...")
        time.sleep(0.2)
        gem_res = test_gemini_api(gemini_key)
        diag["gemini_status"] = gem_res
        st.write(f"   • {'OK' if gem_res.get('ok') else 'Issue'}: {gem_res.get('detail')}")

        if diag["connected"]:
            status_box.update(label="Diagnostic Complete — Core Checks Finished", state="complete", expanded=False)
        else:
            status_box.update(label="Diagnostic Complete — Attention Needed", state="error", expanded=True)

    st.session_state["revio_last_diag"] = diag

diag = st.session_state.get("revio_last_diag")
if diag:
    if diag["connected"]:
        counts = diag.get("table_counts") or {}
        batches_txt = counts.get("upload_batches", "—")
        reviews_txt = counts.get("reviews", "—")
        preds_txt = counts.get("predictions", "—")
        gem = diag.get("gemini_status") or {}

        st.markdown(
            f"""
            <div class="revio-web-card">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                    <div class="revio-card-title" style="margin: 0;">✅ PostgreSQL Connection Healthy</div>
                    <span class="revio-pill" style="margin: 0;">{diag["pg_version"]}</span>
                </div>
                <div class="revio-card-sub" style="margin-top: 8px;">
                    {diag["raw_message"]}
                </div>
                <div style="display: flex; flex-wrap: wrap; gap: 14px; font-size: 0.88rem; margin-top: 8px;">
                    <span>📦 Batches: <b>{batches_txt}</b></span>
                    <span>📝 Reviews: <b>{reviews_txt}</b></span>
                    <span>🤖 Predictions: <b>{preds_txt}</b></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if gem:
            if gem.get("ok"):
                st.success(f"Gemini API: {gem.get('detail')}")
            else:
                st.warning(f"Gemini API: {gem.get('detail')}")
    else:
        st.error(diag["raw_message"] or "Database connection failed.")