from __future__ import annotations

from typing import Any, Dict
import streamlit as st

THEMES: Dict[str, Dict[str, str]] = {
    "Dark": {
        "bg": "#0E1117",
        "surface": "#161B22",
        "surface_alt": "#1F2633",
        "text": "#F0F6FC",
        "muted": "#A6B0C3",
        "border": "#30363D",
        "accent": "#6366F1",
        "accent_soft": "rgba(99, 102, 241, 0.16)",
        "plotly_template": "plotly_dark",
    },
    "Midnight": {
        "bg": "#070B14",
        "surface": "#0F172A",
        "surface_alt": "#1E293B",
        "text": "#F8FAFC",
        "muted": "#94A3B8",
        "border": "#334155",
        "accent": "#06B6D4",
        "accent_soft": "rgba(6, 182, 212, 0.16)",
        "plotly_template": "plotly_dark",
    },
    "Light": {
        "bg": "#F8FAFC",
        "surface": "#FFFFFF",
        "surface_alt": "#F1F5F9",
        "text": "#0F172A",
        "muted": "#475569",
        "border": "#CBD5E1",
        "accent": "#4F46E5",
        "accent_soft": "rgba(79, 70, 229, 0.10)",
        "plotly_template": "plotly_white",
    },
}


def get_active_theme_name() -> str:
    theme = st.session_state.get("revio_theme", "Dark")
    return theme if theme in THEMES else "Dark"


def get_theme_palette() -> Dict[str, str]:
    return THEMES[get_active_theme_name()]


def apply_page_theme(include_metric_cards: bool = False) -> Dict[str, str]:
    """Applies active theme, website cards, animations, and high-contrast controls."""
    t = get_theme_palette()
    metric_css = (
        f"""
        [data-testid="stMetric"] {{
            background-color: {t["surface"]};
            border: 1px solid {t["border"]};
            border-radius: 14px;
            padding: 16px 18px;
            transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
        }}
        [data-testid="stMetric"]:hover {{
            transform: translateY(-2px);
            border-color: {t["accent"]};
        }}
        [data-testid="stMetricLabel"] {{
            color: {t["muted"]} !important;
            font-weight: 500;
        }}
        [data-testid="stMetricValue"] {{
            color: {t["text"]} !important;
            font-weight: 700;
        }}
        """
        if include_metric_cards
        else ""
    )

    st.markdown(
        f"""
        <style>
            @keyframes revioFadeUp {{
                from {{
                    opacity: 0.2;
                    transform: translateY(8px);
                }}
                to {{
                    opacity: 1;
                    transform: translateY(0);
                }}
            }}

            .stApp,
            [data-testid="stAppViewContainer"],
            [data-testid="stHeader"] {{
                background-color: {t["bg"]} !important;
                color: {t["text"]} !important;
            }}
            [data-testid="stHeader"] {{
                background: transparent !important;
            }}
            [data-testid="stSidebar"] {{
                background-color: {t["surface"]} !important;
                border-right: 1px solid {t["border"]};
            }}
            [data-testid="stSidebarNav"] a,
            [data-testid="stSidebarNav"] span,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3 {{
                color: {t["text"]} !important;
                opacity: 1 !important;
            }}
            [data-testid="stSidebarNav"] a[aria-current="page"] {{
                background-color: {t["surface_alt"]} !important;
                border-left: 3px solid {t["accent"]} !important;
            }}

            h1, h2, h3, p, span, div {{
                color: inherit;
            }}
            h1, h2, h3 {{
                color: {t["text"]} !important;
                letter-spacing: -0.02em;
            }}
            .revio-muted {{
                color: {t["muted"]} !important;
            }}

            [data-testid="stWidgetLabel"] label,
            [data-testid="stWidgetLabel"] p,
            .stCheckbox label span,
            [data-testid="stToggle"] label span {{
                color: {t["text"]} !important;
                opacity: 1 !important;
            }}

            /* Text input, date input & multiselect contrast */
            input[type="text"],
            input[type="search"],
            div[data-baseweb="input"] > div,
            div[data-baseweb="input"] input,
            div[data-testid="stDateInput"] input {{
                background-color: {t["surface_alt"]} !important;
                color: {t["text"]} !important;
                border-color: {t["border"]} !important;
                border-radius: 10px !important;
            }}
            input::placeholder,
            textarea::placeholder {{
                color: {t["muted"]} !important;
                opacity: 1 !important;
            }}
            span[data-baseweb="tag"] {{
                background-color: {t["accent_soft"]} !important;
                color: {t["text"]} !important;
                border: 1px solid {t["accent"]} !important;
            }}
            span[data-baseweb="tag"] * {{
                color: {t["text"]} !important;
            }}

            /* Pills / segmented buttons contrast */
            [data-testid="stPills"] button,
            [data-testid="stSegmentedControl"] button {{
                background-color: {t["surface_alt"]} !important;
                color: {t["text"]} !important;
                border: 1px solid {t["border"]} !important;
                border-radius: 999px !important;
                font-weight: 600 !important;
            }}
            [data-testid="stPills"] button *,
            [data-testid="stSegmentedControl"] button * {{
                color: {t["text"]} !important;
            }}
            [data-testid="stPills"] button[aria-pressed="true"],
            [data-testid="stPills"] button[aria-selected="true"],
            [data-testid="stSegmentedControl"] button[aria-pressed="true"],
            [data-testid="stSegmentedControl"] button[aria-selected="true"] {{
                background-color: {t["accent"]} !important;
                border-color: {t["accent"]} !important;
                color: #FFFFFF !important;
            }}
            [data-testid="stPills"] button[aria-pressed="true"] *,
            [data-testid="stPills"] button[aria-selected="true"] *,
            [data-testid="stSegmentedControl"] button[aria-pressed="true"] *,
            [data-testid="stSegmentedControl"] button[aria-selected="true"] * {{
                color: #FFFFFF !important;
            }}

            /* Selectbox & dropdown contrast */
            div[data-baseweb="select"] > div {{
                background-color: {t["surface_alt"]} !important;
                color: {t["text"]} !important;
                border-color: {t["border"]} !important;
                border-radius: 10px !important;
            }}
            div[data-baseweb="select"] * {{
                color: {t["text"]} !important;
            }}
            ul[data-testid="stSelectboxVirtualDropdown"],
            div[data-baseweb="popover"] ul,
            div[data-baseweb="menu"] {{
                background-color: {t["surface"]} !important;
                color: {t["text"]} !important;
                border: 1px solid {t["border"]} !important;
            }}
            ul[data-testid="stSelectboxVirtualDropdown"] li,
            div[data-baseweb="popover"] li,
            div[data-baseweb="menu"] li {{
                background-color: {t["surface"]} !important;
                color: {t["text"]} !important;
            }}
            ul[data-testid="stSelectboxVirtualDropdown"] li:hover,
            div[data-baseweb="popover"] li:hover,
            div[data-baseweb="menu"] li:hover {{
                background-color: {t["surface_alt"]} !important;
                color: {t["accent"]} !important;
            }}

            /* Buttons & download buttons contrast */
            .stButton > button,
            .stDownloadButton > button,
            button[kind="secondary"],
            button[kind="primary"] {{
                background-color: {t["surface_alt"]} !important;
                color: {t["text"]} !important;
                border: 1px solid {t["border"]} !important;
                border-radius: 10px !important;
                font-weight: 600 !important;
                transition: all 0.2s ease !important;
            }}
            .stButton > button *,
            .stDownloadButton > button *,
            button[kind="secondary"] *,
            button[kind="primary"] * {{
                color: {t["text"]} !important;
            }}
            .stButton > button:hover,
            .stDownloadButton > button:hover,
            button[kind="secondary"]:hover,
            button[kind="primary"]:hover {{
                background-color: {t["accent"]} !important;
                border-color: {t["accent"]} !important;
                color: #FFFFFF !important;
            }}
            .stButton > button:hover *,
            .stDownloadButton > button:hover *,
            button[kind="secondary"]:hover *,
            button[kind="primary"]:hover * {{
                color: #FFFFFF !important;
            }}

            /* Status widget (st.status) contrast fix */
            [data-testid="stStatusWidget"],
            [data-testid="stStatusWidget"] > div,
            [data-testid="stStatusWidget"] summary,
            [data-testid="stStatusWidget"] [data-testid="stExpanderDetails"] {{
                background-color: {t["surface"]} !important;
                color: {t["text"]} !important;
                border-color: {t["border"]} !important;
            }}
            [data-testid="stStatusWidget"] summary:hover {{
                background-color: {t["surface_alt"]} !important;
                color: {t["text"]} !important;
            }}
            [data-testid="stStatusWidget"] *,
            [data-testid="stStatusWidget"] summary span,
            [data-testid="stStatusWidget"] summary p,
            [data-testid="stStatusWidget"] [data-testid="stMarkdownContainer"] p,
            [data-testid="stStatusWidget"] [data-testid="stMarkdownContainer"] li {{
                color: {t["text"]} !important;
                opacity: 1 !important;
            }}
            [data-testid="stStatusWidget"] code {{
                background-color: {t["surface_alt"]} !important;
                color: {t["text"]} !important;
                border: 1px solid {t["border"]} !important;
            }}

            .revio-hero {{
                background: linear-gradient(135deg, {t["surface"]} 0%, {t["surface_alt"]} 100%);
                border: 1px solid {t["border"]};
                border-radius: 18px;
                padding: 24px 28px;
                margin-bottom: 18px;
                animation: revioFadeUp 0.35s ease both;
                transition: border-color 0.25s ease;
            }}
            .revio-hero:hover {{
                border-color: {t["accent"]};
            }}
            .revio-pill {{
                display: inline-block;
                background: {t["accent_soft"]};
                color: {t["accent"]};
                border: 1px solid {t["accent"]};
                border-radius: 999px;
                padding: 4px 12px;
                font-size: 0.78rem;
                font-weight: 600;
                margin-bottom: 10px;
            }}

            .revio-web-card {{
                background-color: {t["surface"]};
                border: 1px solid {t["border"]};
                border-radius: 16px;
                padding: 18px 20px;
                margin-bottom: 14px;
                transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
                animation: revioFadeUp 0.4s ease both;
            }}
            .revio-web-card:hover {{
                transform: translateY(-2px);
                border-color: {t["accent"]};
                box-shadow: 0 10px 24px rgba(0, 0, 0, 0.14);
            }}
            .revio-card-title {{
                font-size: 1rem;
                font-weight: 650;
                color: {t["text"]};
                margin-bottom: 4px;
            }}
            .revio-card-sub {{
                font-size: 0.85rem;
                color: {t["muted"]};
                margin-bottom: 10px;
            }}

            .revio-section-head {{
                display: flex;
                justify-content: space-between;
                align-items: flex-end;
                gap: 12px;
                margin: 8px 0 14px 0;
                padding-bottom: 10px;
                border-bottom: 1px solid {t["border"]};
                animation: revioFadeUp 0.35s ease both;
            }}
            .revio-section-title {{
                font-size: 1.25rem;
                font-weight: 700;
                color: {t["text"]};
                margin: 0;
            }}
            .revio-section-desc {{
                font-size: 0.86rem;
                color: {t["muted"]};
                margin: 2px 0 0 0;
            }}
            .revio-bar-track {{
                width: 100%;
                height: 8px;
                background: {t["surface_alt"]};
                border-radius: 999px;
                overflow: hidden;
                margin-top: 6px;
            }}
            .revio-bar-fill {{
                height: 100%;
                background: linear-gradient(90deg, {t["accent"]}, #10B981);
                border-radius: 999px;
                transition: width 0.5s ease;
            }}

            {metric_css}
        </style>
        """,
        unsafe_allow_html=True,
    )
    return t


def optimize_chart(
    fig: Any,
    height: int = 310,
    showlegend: bool = False,
    top_margin: int = 16,
    bottom_margin: int = 30,
    horizontal: bool = False,
) -> Any:
    """Applies clean, chart-specific layout without visual clutter."""
    t = get_theme_palette()
    fig.update_layout(
        template=t["plotly_template"],
        height=height,
        showlegend=showlegend,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=t["text"], size=12),
        margin=dict(t=top_margin, b=bottom_margin, l=12, r=12),
        transition=dict(duration=450, easing="cubic-in-out"),
    )
    if horizontal:
        fig.update_xaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.14)", title_standoff=8)
        fig.update_yaxes(showgrid=False, title_standoff=8)
    else:
        fig.update_xaxes(showgrid=False, title_standoff=8)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.14)", title_standoff=8)
    return fig