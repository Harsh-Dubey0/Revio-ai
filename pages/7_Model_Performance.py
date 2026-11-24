import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

st.set_page_config(page_title="Model Performance • Revio AI", page_icon="🎯", layout="wide")

from config.settings import settings
from src.utils.theme import apply_page_theme, optimize_chart

t = apply_page_theme(include_metric_cards=True)

compact_mode = bool(st.session_state.get("revio_compact_cards", False))
card_pad = "14px 16px" if compact_mode else "18px 20px"


@st.cache_data(show_spinner=False)
def load_metadata(path_str: str, mtime: float) -> tuple[dict | None, str | None]:
    path = Path(path_str)
    if not path.exists():
        return None, f"Metadata file not found at `{path}`."
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None, "Metadata JSON root must be an object."
        return data, None
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON in metadata file: {exc}"
    except Exception as exc:
        return None, f"Failed to read metadata file: {exc}"


def extract_top_confusions(matrix: list[list[int]], labels: list[str], top_n: int = 3) -> list[dict]:
    pairs = []
    n = min(len(matrix), len(labels))
    for i in range(n):
        for j in range(n):
            if i != j:
                val = int(matrix[i][j])
                if val > 0:
                    pairs.append({"true": labels[i], "pred": labels[j], "count": val})
    pairs.sort(key=lambda x: x["count"], reverse=True)
    return pairs[:top_n]


def build_confusion_fig(matrix: list[list[int]], labels: list[str], normalized: bool):
    arr = pd.DataFrame(matrix, index=labels, columns=labels).astype(float)
    if normalized:
        row_sums = arr.sum(axis=1).replace(0, 1)
        display_df = arr.div(row_sums, axis=0)
        annotations = [[f"{val:.1%}" for val in row] for row in display_df.values]
        z_vals = display_df.values.tolist()
        hover_tmpl = "True: <b>%{y}</b> → Pred: <b>%{x}</b><br>Rate: <b>%{z:.1%}</b><extra></extra>"
    else:
        annotations = [[f"{int(val):,}" for val in row] for row in arr.values]
        z_vals = arr.values.tolist()
        hover_tmpl = "True: <b>%{y}</b> → Pred: <b>%{x}</b><br>Count: <b>%{z:,}</b><extra></extra>"

    fig = ff.create_annotated_heatmap(
        z=z_vals,
        x=labels,
        y=labels,
        annotation_text=annotations,
        colorscale="Blues",
        showscale=True,
    )
    fig.update_traces(hovertemplate=hover_tmpl)
    fig.update_layout(
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
    )
    fig.update_yaxes(autorange="reversed")
    return optimize_chart(fig, height=350, showlegend=False, top_margin=16, bottom_margin=34)


def load_error_samples(meta: dict) -> pd.DataFrame:
    for key in ("misclassified_samples", "error_examples", "sample_errors"):
        if isinstance(meta.get(key), list) and meta[key]:
            return pd.DataFrame(meta[key])

    candidate_paths = []
    for k in ("errors_path", "predictions_path", "eval_predictions_path"):
        if meta.get(k):
            candidate_paths.append(Path(meta[k]))
    for attr in ("MODEL_ERRORS_PATH", "EVAL_PREDICTIONS_PATH"):
        if hasattr(settings, attr) and getattr(settings, attr):
            candidate_paths.append(Path(getattr(settings, attr)))

    for p in candidate_paths:
        if p.exists():
            try:
                df_err = (
                    pd.read_csv(p)
                    if p.suffix.lower() == ".csv"
                    else pd.DataFrame(json.loads(p.read_text(encoding="utf-8")))
                )
                if {"true_label", "pred_label"}.issubset(df_err.columns):
                    df_err = df_err[df_err["true_label"] != df_err["pred_label"]]
                return df_err.reset_index(drop=True)
            except Exception:
                continue

    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_calibration_dataframe(meta_json_str: str, accuracy: float, total_support: int) -> tuple[pd.DataFrame, bool]:
    meta = json.loads(meta_json_str)

    for key in ("calibration_records", "eval_predictions", "test_predictions"):
        if isinstance(meta.get(key), list) and meta[key]:
            df_c = pd.DataFrame(meta[key])
            conf_col = next((c for c in ("confidence", "prob", "max_prob", "predicted_prob") if c in df_c.columns), None)
            if conf_col:
                out = pd.DataFrame()
                out["confidence"] = pd.to_numeric(df_c[conf_col], errors="coerce").clip(0.0, 1.0)
                if "is_correct" in df_c.columns:
                    out["correct"] = df_c["is_correct"].astype(bool)
                elif {"true_label", "pred_label"}.issubset(df_c.columns):
                    out["correct"] = df_c["true_label"].astype(str) == df_c["pred_label"].astype(str)
                else:
                    out["correct"] = out["confidence"] >= 0.5
                out = out.dropna(subset=["confidence"]).reset_index(drop=True)
                if not out.empty:
                    return out, False

    paths = []
    for k in ("calibration_path", "eval_predictions_path", "predictions_path"):
        if meta.get(k):
            paths.append(Path(meta[k]))
    for attr in ("EVAL_PREDICTIONS_PATH", "CALIBRATION_DATA_PATH"):
        if hasattr(settings, attr) and getattr(settings, attr):
            paths.append(Path(getattr(settings, attr)))

    for p in paths:
        if p.exists():
            try:
                df_f = (
                    pd.read_csv(p)
                    if p.suffix.lower() == ".csv"
                    else pd.DataFrame(json.loads(p.read_text(encoding="utf-8")))
                )
                conf_col = next((c for c in ("confidence", "prob", "max_prob", "predicted_prob") if c in df_f.columns), None)
                if conf_col:
                    out = pd.DataFrame()
                    out["confidence"] = pd.to_numeric(df_f[conf_col], errors="coerce").clip(0.0, 1.0)
                    if "is_correct" in df_f.columns:
                        out["correct"] = df_f["is_correct"].astype(bool)
                    elif {"true_label", "pred_label"}.issubset(df_f.columns):
                        out["correct"] = df_f["true_label"].astype(str) == df_f["pred_label"].astype(str)
                    else:
                        out["correct"] = out["confidence"] >= 0.5
                    out = out.dropna(subset=["confidence"]).reset_index(drop=True)
                    if not out.empty:
                        return out, False
            except Exception:
                continue

    rng = np.random.default_rng(42)
    n = int(np.clip(total_support if total_support > 0 else 1000, 200, 5000))
    acc = float(np.clip(accuracy, 0.05, 0.99))

    n_correct = int(round(n * acc))
    n_wrong = n - n_correct

    correct_conf = np.clip(rng.beta(8.0, 1.8, size=n_correct), 0.50, 0.999)
    wrong_conf = np.clip(rng.beta(2.2, 3.0, size=n_wrong), 0.25, 0.92)

    confidences = np.concatenate([correct_conf, wrong_conf])
    correctness = np.concatenate([np.ones(n_correct, dtype=bool), np.zeros(n_wrong, dtype=bool)])

    perm = rng.permutation(n)
    df_sim = pd.DataFrame({"confidence": confidences[perm], "correct": correctness[perm]})
    return df_sim, True


def compute_calibration_metrics(df_cal: pd.DataFrame, n_bins: int = 10) -> tuple[float, float, pd.DataFrame]:
    confs = df_cal["confidence"].to_numpy(dtype=float)
    y_true = df_cal["correct"].to_numpy(dtype=float)
    brier = float(np.mean((confs - y_true) ** 2))

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(confs, bins, right=True) - 1, 0, n_bins - 1)

    rows = []
    ece = 0.0
    total = len(confs)

    for b in range(n_bins):
        mask = bin_ids == b
        count = int(np.sum(mask))
        low, high = float(bins[b]), float(bins[b + 1])
        label = f"{low:.1f}–{high:.1f}"
        if count > 0:
            avg_conf = float(np.mean(confs[mask]))
            emp_acc = float(np.mean(y_true[mask]))
            gap = abs(emp_acc - avg_conf)
            ece += (count / max(total, 1)) * gap
        else:
            avg_conf = float((low + high) / 2.0)
            emp_acc = np.nan
            gap = 0.0

        rows.append(
            {
                "Bin": label,
                "Count": count,
                "Avg Confidence": avg_conf,
                "Empirical Accuracy": emp_acc,
                "Calibration Gap": gap,
            }
        )

    return float(ece), brier, pd.DataFrame(rows)


# Revio AI Hero Banner
st.markdown(
    """
    <div class="revio-hero">
        <div class="revio-pill">🎯 Revio AI • Evaluation & Governance</div>
        <h1 style="margin: 0 0 6px 0; font-size: 2rem;">Model Performance & Transparency</h1>
        <p class="revio-muted" style="margin: 0; font-size: 0.98rem;">
            Production evaluation telemetry, class-level diagnostics, confidence calibration, and governance card.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

meta_path = Path(settings.MODEL_METADATA_PATH)
if not meta_path.exists():
    st.markdown(
        f"""
        <div class="revio-web-card" style="padding: {card_pad};">
            <div class="revio-card-title">Model metadata artifact missing</div>
            <div class="revio-card-sub" style="margin-bottom: 0;">
                Expected file at <code>{meta_path}</code>. Run the training and evaluation pipeline to generate it.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

stat = meta_path.stat()
meta, load_err = load_metadata(str(meta_path), stat.st_mtime)

# Action Bar for Reload & Model Card Download
act_l, act_m, act_r = st.columns([4.8, 1.6, 1.6], vertical_alignment="center")
with act_l:
    st.markdown(f"<span class='revio-pill' style='margin: 0;'>Artifact: {meta_path.name}</span>", unsafe_allow_html=True)
with act_m:
    if st.button("↻ Reload", use_container_width=True, help="Force reload metadata from disk"):
        load_metadata.clear()
        load_calibration_dataframe.clear()
        st.rerun()

if load_err or not meta:
    st.error(load_err or "Unable to load metadata.")
    st.stop()

with act_r:
    st.download_button(
        "⬇ Model Card",
        data=json.dumps(meta, indent=2).encode("utf-8"),
        file_name=f"{meta.get('model_version', 'model')}_card.json",
        mime="application/json",
        use_container_width=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

test_perf = meta.get("test_performance", {})
arch_raw = str(meta.get("best_model_architecture", "Unknown"))
arch_short = arch_raw.split("_")[-1] if "_" in arch_raw else arch_raw
updated_str = pd.to_datetime(stat.st_mtime, unit="s", utc=True).strftime("%b %d, %Y · %H:%M UTC")
accuracy_val = float(test_perf.get("accuracy", 0.0))
macro_f1_val = float(test_perf.get("macro_f1", 0.0))

# Executive KPI Row
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Active Version", str(meta.get("model_version", "N/A")))
k2.metric("Architecture", arch_short)
k3.metric("Test Accuracy", f"{accuracy_val:.2%}")
k4.metric("Macro F1", f"{macro_f1_val:.2%}")
k5.metric("Artifact Updated", updated_str)

st.markdown("<br>", unsafe_allow_html=True)

raw_report = test_perf.get("classification_report", {})
class_rows = {
    label: metrics
    for label, metrics in raw_report.items()
    if isinstance(metrics, dict) and label.lower() not in {"accuracy", "macro avg", "weighted avg"}
}
df_classes = pd.DataFrame()
total_support = 0
if class_rows:
    df_classes = pd.DataFrame(class_rows).transpose()
    cols_order = [c for c in ["precision", "recall", "f1-score", "support"] if c in df_classes.columns]
    df_classes = df_classes[cols_order].copy()
    for m_col in ["precision", "recall", "f1-score"]:
        if m_col in df_classes.columns:
            df_classes[m_col] = pd.to_numeric(df_classes[m_col], errors="coerce")
    if "support" in df_classes.columns:
        df_classes["support"] = pd.to_numeric(df_classes["support"], errors="coerce").fillna(0).astype(int)
        total_support = int(df_classes["support"].sum())
    df_classes.index.name = "Class"

cm_data = test_perf.get("confusion_matrix", {})
cm_matrix = cm_data.get("matrix")
cm_labels = cm_data.get("labels")
if total_support == 0 and cm_matrix:
    total_support = int(np.sum(np.array(cm_matrix, dtype=float)))

tab_diag, tab_cal, tab_errors, tab_gov = st.tabs(
    ["📊 Diagnostics", "🎯 Confidence & Calibration", "🔎 Error Analysis", "🛡️ Governance & Artifacts"]
)

with tab_diag:
    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([1.05, 1], gap="large")

    with col_l:
        st.markdown(
            f"""
            <div class="revio-web-card" style="padding: {card_pad};">
                <div class="revio-card-title">Per-Class Metrics</div>
                <div class="revio-card-sub">Precision, recall, F1-score, and sample support per class.</div>
            """,
            unsafe_allow_html=True,
        )
        if not df_classes.empty:
            st.dataframe(
                df_classes.reset_index(),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Class": st.column_config.TextColumn("Class", width="small"),
                    "precision": st.column_config.ProgressColumn("Precision", min_value=0.0, max_value=1.0, format="%.3f"),
                    "recall": st.column_config.ProgressColumn("Recall", min_value=0.0, max_value=1.0, format="%.3f"),
                    "f1-score": st.column_config.ProgressColumn("F1-Score", min_value=0.0, max_value=1.0, format="%.3f"),
                    "support": st.column_config.NumberColumn("Support", format="%d"),
                },
            )
            if "f1-score" in df_classes.columns:
                st.markdown("**F1-Score by Class**")
                chart_df = df_classes.reset_index().sort_values("f1-score", ascending=True)
                fig_f1 = px.bar(
                    chart_df,
                    x="f1-score",
                    y="Class",
                    orientation="h",
                    range_x=[0, 1.05],
                    text=chart_df["f1-score"].map(lambda v: f"{v:.2f}"),
                )
                fig_f1.update_traces(marker_color=t["accent"], textposition="outside", cliponaxis=False)
                fig_f1.update_layout(xaxis_title="F1 Score", yaxis_title="")
                optimize_chart(fig_f1, height=220, showlegend=False, top_margin=10, bottom_margin=28, horizontal=True)
                st.plotly_chart(fig_f1, use_container_width=True)
        else:
            st.info("No per-class classification metrics available in metadata.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown(
            f"""
            <div class="revio-web-card" style="padding: {card_pad};">
                <div class="revio-card-title">Confusion Matrix</div>
                <div class="revio-card-sub">Compare ground-truth labels against model predictions.</div>
            """,
            unsafe_allow_html=True,
        )
        cm_view = st.segmented_control(
            "Confusion Mode",
            options=["Counts", "Normalized"],
            default="Counts",
            label_visibility="collapsed",
        )
        if cm_matrix and cm_labels:
            st.plotly_chart(
                build_confusion_fig(cm_matrix, cm_labels, normalized=(cm_view == "Normalized")),
                use_container_width=True,
            )
            top_pairs = extract_top_confusions(cm_matrix, cm_labels, top_n=3)
            if top_pairs:
                summary_str = " · ".join(
                    f"**{p['true']}** → **{p['pred']}** ({p['count']:,})" for p in top_pairs
                )
                st.caption(f"Top off-diagonal confusions: {summary_str}")
        else:
            st.info("Confusion matrix not available in metadata.")
        st.markdown("</div>", unsafe_allow_html=True)

with tab_cal:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="revio-web-card" style="padding: {card_pad};">
            <div class="revio-card-title">Confidence Distribution & Reliability Calibration</div>
            <div class="revio-card-sub">Tune confidence routing thresholds and inspect calibration alignment.</div>
        """,
        unsafe_allow_html=True,
    )

    df_cal, is_sim = load_calibration_dataframe(json.dumps(meta, default=str), accuracy_val, total_support)
    if is_sim:
        st.caption(
            "ℹ️ Showing **simulated probability diagnostics** calibrated to your reported test accuracy. "
            "Provide `calibration_records` in metadata or `EVAL_PREDICTIONS_PATH` in settings to plot actual model probabilities."
        )
    else:
        st.caption("Showing **live evaluation probability telemetry** from recorded predictions.")

    ctrl_l, ctrl_r = st.columns([2, 1.2], vertical_alignment="center")
    with ctrl_l:
        conf_threshold = st.slider(
            "Confidence Routing Threshold (Abstain / Low-Confidence Queue below threshold)",
            min_value=0.10,
            max_value=0.95,
            value=0.65,
            step=0.05,
        )
    with ctrl_r:
        n_bins = st.select_slider("Calibration Bins", options=[5, 8, 10, 15], value=10)
    st.markdown("</div>", unsafe_allow_html=True)

    ece_val, brier_val, bins_df = compute_calibration_metrics(df_cal, n_bins=n_bins)

    accepted_mask = df_cal["confidence"] >= conf_threshold
    coverage = float(accepted_mask.mean()) if len(df_cal) else 0.0
    filtered_acc = float(df_cal.loc[accepted_mask, "correct"].mean()) if accepted_mask.any() else 0.0

    cm1, cm2, cm3, cm4 = st.columns(4)
    cm1.metric("Expected Calibration Error", f"{ece_val:.3f}", help="Lower is better (0.0 = perfectly calibrated)")
    cm2.metric("Brier Score", f"{brier_val:.3f}", help="Mean squared error of probability forecasts (lower is better)")
    cm3.metric("Coverage @ Threshold", f"{coverage:.1%}", help="Share of predictions above confidence threshold")
    cm4.metric("Accuracy @ Threshold", f"{filtered_acc:.2%}", delta=f"{filtered_acc - accuracy_val:+.2%} vs baseline")

    st.markdown("<br>", unsafe_allow_html=True)
    cal_left, cal_right = st.columns([1.05, 1], gap="large")

    with cal_left:
        st.markdown(
            f"""
            <div class="revio-web-card" style="padding: {card_pad};">
                <div class="revio-card-title">Reliability Diagram (Calibration Curve)</div>
                <div class="revio-card-sub">Mean predicted confidence vs empirical accuracy.</div>
            """,
            unsafe_allow_html=True,
        )
        valid_bins = bins_df.dropna(subset=["Empirical Accuracy"])
        rel_fig = go.Figure()
        rel_fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Perfect Calibration",
                line=dict(dash="dash", color="#94a3b8", width=2),
            )
        )
        rel_fig.add_trace(
            go.Scatter(
                x=valid_bins["Avg Confidence"],
                y=valid_bins["Empirical Accuracy"],
                mode="lines+markers",
                name="Model Reliability",
                line=dict(color=t["accent"], width=2.5),
                marker=dict(size=8),
            )
        )
        rel_fig.update_layout(
            xaxis_title="Mean Predicted Confidence",
            yaxis_title="Empirical Accuracy",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        optimize_chart(rel_fig, height=330, showlegend=True, top_margin=24, bottom_margin=34)
        st.plotly_chart(rel_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with cal_right:
        st.markdown(
            f"""
            <div class="revio-web-card" style="padding: {card_pad};">
                <div class="revio-card-title">Confidence Histogram by Outcome</div>
                <div class="revio-card-sub">Confidence spread split by correct vs incorrect predictions.</div>
            """,
            unsafe_allow_html=True,
        )
        plot_hist_df = df_cal.copy()
        plot_hist_df["Outcome"] = np.where(plot_hist_df["correct"], "Correct", "Incorrect")
        hist_fig = px.histogram(
            plot_hist_df,
            x="confidence",
            color="Outcome",
            nbins=20,
            barmode="overlay",
            opacity=0.72,
            color_discrete_map={"Correct": "#16a34a", "Incorrect": "#dc2626"},
        )
        hist_fig.add_vline(
            x=conf_threshold,
            line_dash="dot",
            line_color=t["text"],
            annotation_text=f"Threshold {conf_threshold:.2f}",
            annotation_position="top left",
        )
        hist_fig.update_layout(
            xaxis_title="Predicted Confidence",
            yaxis_title="Sample Count",
            xaxis=dict(range=[0, 1]),
        )
        optimize_chart(hist_fig, height=330, showlegend=True, top_margin=20, bottom_margin=34)
        st.plotly_chart(hist_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Bin-by-Bin Calibration Table", expanded=False):
        st.dataframe(
            bins_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Bin": st.column_config.TextColumn("Confidence Bin"),
                "Count": st.column_config.NumberColumn("Samples", format="%d"),
                "Avg Confidence": st.column_config.NumberColumn("Avg Conf.", format="%.3f"),
                "Empirical Accuracy": st.column_config.NumberColumn("Emp. Acc.", format="%.3f"),
                "Calibration Gap": st.column_config.NumberColumn("Gap |Acc - Conf|", format="%.3f"),
            },
        )

with tab_errors:
    st.markdown("<br>", unsafe_allow_html=True)
    diag_c1, diag_c2 = st.columns(2, gap="large")
    with diag_c1:
        worst_html = "<span class='revio-muted'>Insufficient per-class data.</span>"
        if not df_classes.empty and "f1-score" in df_classes.columns:
            worst_idx = df_classes["f1-score"].idxmin()
            worst_val = float(df_classes.loc[worst_idx, "f1-score"])
            worst_html = f"<code>{worst_idx}</code> — F1: <b>{worst_val:.3f}</b>"
        st.markdown(
            f"""
            <div class="revio-web-card" style="padding: {card_pad};">
                <div class="revio-card-title">Lowest F1 Class</div>
                <div style="font-size: 0.96rem; margin-top: 6px;">{worst_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with diag_c2:
        conf_html = "<span class='revio-muted'>No off-diagonal errors recorded.</span>"
        top_pairs = extract_top_confusions(cm_matrix, cm_labels, top_n=1) if (cm_matrix and cm_labels) else []
        if top_pairs:
            tp = top_pairs[0]
            conf_html = f"True <code>{tp['true']}</code> predicted as <code>{tp['pred']}</code> (<b>{tp['count']:,}</b> times)"
        st.markdown(
            f"""
            <div class="revio-web-card" style="padding: {card_pad};">
                <div class="revio-card-title">Most Frequent Confusion Pair</div>
                <div style="font-size: 0.96rem; margin-top: 6px;">{conf_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    err_df = load_error_samples(meta)
    st.markdown(
        f"""
        <div class="revio-web-card" style="padding: {card_pad};">
            <div class="revio-card-title">Misclassified Records</div>
            <div class="revio-card-sub">Inspect sample errors to spot recurring edge cases.</div>
        """,
        unsafe_allow_html=True,
    )
    if not err_df.empty:
        if "true_label" in err_df.columns:
            filter_true = st.selectbox(
                "Filter by True Label",
                ["All"] + sorted(err_df["true_label"].astype(str).unique().tolist()),
            )
            if filter_true != "All":
                err_df = err_df[err_df["true_label"].astype(str) == filter_true]
        st.dataframe(err_df.head(100), hide_index=True, use_container_width=True)
    else:
        st.info(
            "No sample error records attached yet. To enable row-level error inspection, include a "
            "`misclassified_samples` list in your model metadata JSON or configure `MODEL_ERRORS_PATH`."
        )
    st.markdown("</div>", unsafe_allow_html=True)

with tab_gov:
    st.markdown("<br>", unsafe_allow_html=True)
    disclaimer = meta.get("limitations_disclaimer")
    if disclaimer:
        st.warning(f"**Limitations & Disclaimer:** {disclaimer}", icon="⚠️")
    else:
        st.info("No explicit limitations disclaimer was provided in the model metadata.")

    st.markdown(
        f"""
        <div class="revio-web-card" style="padding: {card_pad};">
            <div class="revio-card-title">Artifact & Environment Check</div>
            <div class="revio-card-sub">Verify presence and size of model and evaluation artifacts on disk.</div>
        """,
        unsafe_allow_html=True,
    )
    art_rows = [
        {
            "Artifact": "Model Metadata JSON",
            "Path": str(meta_path),
            "Status": "Present",
            "Size (KB)": round(stat.st_size / 1024, 2),
        }
    ]
    for path_key in ("model_path", "checkpoint_path", "tokenizer_path"):
        raw_p = meta.get(path_key) or getattr(settings, path_key.upper(), None)
        if raw_p:
            p_obj = Path(raw_p)
            exists = p_obj.exists()
            size_mb = round(p_obj.stat().st_size / (1024 * 1024), 2) if exists and p_obj.is_file() else None
            art_rows.append(
                {
                    "Artifact": path_key.replace("_", " ").title(),
                    "Path": str(p_obj),
                    "Status": "Present" if exists else "Missing",
                    "Size (KB)": f"{size_mb} MB" if size_mb is not None else "—",
                }
            )
    st.dataframe(pd.DataFrame(art_rows), hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    extra_meta = {k: v for k, v in meta.items() if k != "test_performance"}
    with st.expander("Full Model Card JSON", expanded=False):
        st.json(extra_meta, expanded=False)