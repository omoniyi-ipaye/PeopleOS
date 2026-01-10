"""
Styling module for PeopleOS.

Clean & minimal design system inspired by Notion and Linear.
Uses Inter font, subtle borders, and generous whitespace.
"""

from ui.design_system import COLORS_LIGHT, COLORS_DARK, TYPOGRAPHY, SPACING, RADIUS, SHADOWS, TRANSITIONS

# Map design system colors to legacy format for backwards compatibility
DARK_COLORS = {
    "background": COLORS_DARK["background"],
    "card_bg": COLORS_DARK["surface"],
    "text_primary": COLORS_DARK["text_primary"],
    "text_secondary": COLORS_DARK["text_secondary"],
    "accent": COLORS_DARK["accent"],
    "warning": COLORS_DARK["warning"],
    "danger": COLORS_DARK["danger"],
    "info": COLORS_DARK["info"],
    "chart_palette": COLORS_DARK["chart_palette"],
    "border": COLORS_DARK["border"],
    "surface_hover": COLORS_DARK["surface_hover"],
}

LIGHT_COLORS = {
    "background": COLORS_LIGHT["background"],
    "card_bg": COLORS_LIGHT["surface"],
    "text_primary": COLORS_LIGHT["text_primary"],
    "text_secondary": COLORS_LIGHT["text_secondary"],
    "accent": COLORS_LIGHT["accent"],
    "warning": COLORS_LIGHT["warning"],
    "danger": COLORS_LIGHT["danger"],
    "info": COLORS_LIGHT["info"],
    "chart_palette": COLORS_LIGHT["chart_palette"],
    "border": COLORS_LIGHT["border"],
    "surface_hover": COLORS_LIGHT["surface_hover"],
}

# Default to dark mode
COLORS = DARK_COLORS.copy()

# Risk level colors (same for both modes)
RISK_COLORS = {
    "High": "#f44336",
    "Medium": "#ff9800",
    "Low": "#4CAF50"
}


def set_theme(dark_mode: bool = True) -> dict:
    """
    Set the active theme.

    Args:
        dark_mode: True for dark mode, False for light mode.

    Returns:
        The active color palette.
    """
    global COLORS
    COLORS = DARK_COLORS.copy() if dark_mode else LIGHT_COLORS.copy()
    return COLORS


def get_current_colors(dark_mode: bool = True) -> dict:
    """
    Get colors for the specified theme.

    Args:
        dark_mode: True for dark mode, False for light mode.

    Returns:
        Color palette dictionary.
    """
    return DARK_COLORS if dark_mode else LIGHT_COLORS


def get_custom_css(dark_mode: bool = True) -> str:
    """
    Get custom CSS for the Streamlit application.

    Clean & minimal design with Inter font.

    Args:
        dark_mode: True for dark mode, False for light mode.

    Returns:
        CSS string to be injected via st.markdown.
    """
    colors = get_current_colors(dark_mode)
    border_color = colors.get('border', colors['card_bg'])

    return f"""
    <style>
        /* Import Inter font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* Global font family */
        html, body, [class*="css"] {{
            font-family: {TYPOGRAPHY['font_family']};
        }}

        /* Main background */
        .stApp {{
            background-color: {colors['background']};
        }}

        /* Clean card styling with subtle border instead of heavy shadow */
        .card {{
            background-color: {colors['card_bg']};
            border-radius: {RADIUS['md']};
            padding: {SPACING['6']};
            margin: {SPACING['4']} 0;
            border: 1px solid {border_color};
        }}

        /* KPI card - clean & minimal */
        .kpi-card {{
            background-color: {colors['card_bg']};
            border-radius: {RADIUS['md']};
            padding: {SPACING['5']};
            text-align: left;
            min-height: 100px;
            border: 1px solid {border_color};
        }}

        .kpi-value {{
            font-size: 2rem;
            font-weight: 600;
            color: {colors['text_primary']};
            margin: 0;
            line-height: 1.2;
        }}

        .kpi-label {{
            font-size: 13px;
            color: {colors['text_secondary']};
            margin-top: {SPACING['2']};
            font-weight: 500;
        }}

        /* Risk badges - pill style */
        .risk-high {{
            background-color: {RISK_COLORS['High']};
            color: white;
            padding: 4px 12px;
            border-radius: {RADIUS['full']};
            font-weight: 500;
            font-size: 12px;
        }}

        .risk-medium {{
            background-color: {RISK_COLORS['Medium']};
            color: white;
            padding: 4px 12px;
            border-radius: {RADIUS['full']};
            font-weight: 500;
            font-size: 12px;
        }}

        .risk-low {{
            background-color: {RISK_COLORS['Low']};
            color: white;
            padding: 4px 12px;
            border-radius: {RADIUS['full']};
            font-weight: 500;
            font-size: 12px;
        }}

        /* Headers - clean typography */
        h1 {{
            color: {colors['text_primary']};
            font-size: {TYPOGRAPHY['h1']['size']};
            font-weight: {TYPOGRAPHY['h1']['weight']};
            line-height: {TYPOGRAPHY['h1']['line_height']};
        }}

        h2 {{
            color: {colors['text_primary']};
            font-size: {TYPOGRAPHY['h2']['size']};
            font-weight: {TYPOGRAPHY['h2']['weight']};
        }}

        h3 {{
            color: {colors['text_primary']};
            font-size: {TYPOGRAPHY['h3']['size']};
            font-weight: {TYPOGRAPHY['h3']['weight']};
        }}

        /* Text - proper hierarchy */
        p {{
            color: {colors['text_secondary']};
            font-size: {TYPOGRAPHY['body']['size']};
            line-height: {TYPOGRAPHY['body']['line_height']};
        }}

        /* Tables - clean with subtle borders */
        .dataframe {{
            background-color: {colors['card_bg']} !important;
            border-radius: {RADIUS['md']};
        }}

        .dataframe th {{
            background-color: {colors['background']} !important;
            color: {colors['text_primary']} !important;
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .dataframe td {{
            color: {colors['text_secondary']} !important;
            font-size: 14px;
        }}

        /* Metric styling */
        [data-testid="stMetricValue"] {{
            color: {colors['text_primary']};
            font-weight: 600;
        }}

        [data-testid="stMetricLabel"] {{
            color: {colors['text_secondary']};
            font-size: 13px;
        }}

        /* Sidebar - cleaner */
        [data-testid="stSidebar"] {{
            background-color: {colors['card_bg']};
            border-right: 1px solid {border_color};
        }}

        /* Tabs - minimal style */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0;
            border-bottom: 1px solid {border_color};
        }}

        .stTabs [data-baseweb="tab"] {{
            background-color: transparent;
            border-radius: 0;
            padding: 12px 20px;
            color: {colors['text_secondary']};
            border-bottom: 2px solid transparent;
            font-weight: 500;
        }}

        .stTabs [data-baseweb="tab"]:hover {{
            color: {colors['text_primary']};
        }}

        .stTabs [data-baseweb="tab"][aria-selected="true"] {{
            background-color: transparent;
            color: {colors['accent']};
            border-bottom: 2px solid {colors['accent']};
        }}

        /* Buttons - clean minimal */
        .stButton > button {{
            background-color: {colors['accent']};
            color: white;
            border: none;
            border-radius: {RADIUS['md']};
            padding: 10px 20px;
            font-weight: 500;
            font-size: 14px;
            transition: {TRANSITIONS['base']};
        }}

        .stButton > button:hover {{
            opacity: 0.9;
        }}

        /* Secondary button style */
        .stButton > button[kind="secondary"] {{
            background-color: transparent;
            color: {colors['text_primary']};
            border: 1px solid {border_color};
        }}

        /* File uploader - cleaner */
        [data-testid="stFileUploader"] {{
            background-color: {colors['card_bg']};
            border-radius: {RADIUS['md']};
            padding: {SPACING['5']};
            border: 1px dashed {border_color};
        }}

        /* Expander - minimal */
        .streamlit-expanderHeader {{
            background-color: {colors['card_bg']};
            border-radius: {RADIUS['md']};
            border: 1px solid {border_color};
        }}

        /* Alert boxes - clean */
        .stAlert {{
            background-color: {colors['card_bg']};
            border-radius: {RADIUS['md']};
            border: 1px solid {border_color};
        }}

        /* Focus indicators for accessibility */
        button:focus, input:focus, select:focus {{
            outline: 2px solid {colors['accent']};
            outline-offset: 2px;
        }}

        /* Dividers - subtle */
        hr {{
            border: none;
            border-top: 1px solid {border_color};
            margin: {SPACING['6']} 0;
        }}

        /* Mobile responsive */
        @media (max-width: 768px) {{
            .stColumns {{
                flex-direction: column;
            }}
            .kpi-card {{
                margin-bottom: {SPACING['3']};
            }}
            .stTabs [data-baseweb="tab"] {{
                padding: 8px 12px;
                font-size: 13px;
            }}
        }}

        /* Insight Tooltip - cleaner */
        .insight-tooltip {{
            position: relative;
            display: inline-block;
            cursor: help;
        }}

        .insight-tooltip .tooltip-icon {{
            color: {colors['info']};
            font-size: 14px;
            margin-left: 4px;
            opacity: 0.6;
            transition: opacity 0.2s;
        }}

        .insight-tooltip:hover .tooltip-icon {{
            opacity: 1;
        }}

        .insight-tooltip .tooltip-text {{
            visibility: hidden;
            width: 280px;
            background: {colors['card_bg']};
            color: {colors['text_primary']};
            text-align: left;
            border-radius: {RADIUS['md']};
            padding: {SPACING['4']};
            position: absolute;
            z-index: 1000;
            bottom: 125%;
            left: 50%;
            margin-left: -140px;
            opacity: 0;
            transition: opacity 0.2s;
            font-size: 13px;
            line-height: 1.5;
            box-shadow: {SHADOWS['lg']};
            border: 1px solid {border_color};
        }}

        .insight-tooltip:hover .tooltip-text {{
            visibility: visible;
            opacity: 1;
        }}

        .insight-label {{
            font-size: 11px;
            color: {colors['accent']};
            font-weight: 600;
            margin-bottom: 4px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        /* Success/info/warning message styling */
        .stSuccess, .stInfo, .stWarning, .stError {{
            border-radius: {RADIUS['md']};
            font-size: 14px;
        }}
    </style>
    """



def get_plotly_theme(dark_mode: bool = True) -> dict:
    """
    Get Plotly theme configuration matching the app style.

    Args:
        dark_mode: True for dark mode, False for light mode.

    Returns:
        Dictionary with Plotly layout configuration.
    """
    colors = get_current_colors(dark_mode)
    grid_color = '#333' if dark_mode else '#ddd'

    return {
        'paper_bgcolor': colors['background'],
        'plot_bgcolor': colors['card_bg'],
        'font': {
            'color': colors['text_primary'],
            'family': 'Inter, sans-serif'
        },
        'title': {
            'font': {
                'color': colors['text_primary'],
                'size': 16
            }
        },
        'xaxis': {
            'gridcolor': grid_color,
            'linecolor': grid_color,
            'tickfont': {'color': colors['text_secondary']}
        },
        'yaxis': {
            'gridcolor': grid_color,
            'linecolor': grid_color,
            'tickfont': {'color': colors['text_secondary']}
        },
        'legend': {
            'bgcolor': 'rgba(0,0,0,0)',
            'font': {'color': colors['text_secondary']}
        },
        'colorway': colors['chart_palette']
    }


def apply_plotly_theme(fig, dark_mode: bool = None) -> None:
    """
    Apply the PeopleOS theme to a Plotly figure.

    Args:
        fig: Plotly figure object.
        dark_mode: Theme mode. If None, uses session state preference.
    """
    import streamlit as st
    if dark_mode is None:
        dark_mode = st.session_state.get('dark_mode', True)
    theme = get_plotly_theme(dark_mode)
    fig.update_layout(**theme)
