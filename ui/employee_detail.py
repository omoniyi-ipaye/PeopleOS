"""
Employee Detail module for PeopleOS.

Provides detailed view for individual employee risk analysis
with plain-English explanations and actionable recommendations.
"""

from typing import Any, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.styling import COLORS, RISK_COLORS


def render_employee_selector(
    high_risk_df: pd.DataFrame,
    key: str = "employee_selector"
) -> Optional[str]:
    """
    Render a dropdown to select an employee for detailed view.

    Args:
        high_risk_df: DataFrame with high-risk employees.
        key: Unique key for the selectbox widget.

    Returns:
        Selected employee ID or None.
    """
    if high_risk_df is None or high_risk_df.empty:
        return None

    if 'EmployeeID' not in high_risk_df.columns:
        return None

    employee_ids = high_risk_df['EmployeeID'].tolist()

    selected = st.selectbox(
        "Select employee for detailed analysis",
        options=["-- Select an employee --"] + employee_ids,
        key=key
    )

    if selected == "-- Select an employee --":
        return None

    return selected


def render_employee_risk_card(
    employee_data: pd.Series,
    risk_score: float,
    risk_category: str
) -> None:
    """
    Render a summary card for an employee's risk profile.

    Args:
        employee_data: Series with employee data.
        risk_score: Risk probability (0-1).
        risk_category: Risk category ('High', 'Medium', 'Low').
    """
    risk_color = RISK_COLORS.get(risk_category, COLORS['text_secondary'])

    st.markdown(f"""
    <div style="
        background-color: {COLORS['card_bg']};
        border-left: 4px solid {risk_color};
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    ">
        <h3 style="margin: 0; color: {COLORS['text_primary']};">
            Employee: {employee_data.get('EmployeeID', 'Unknown')}
        </h3>
        <p style="color: {COLORS['text_secondary']}; margin-top: 10px;">
            <strong>Department:</strong> {employee_data.get('Dept', 'N/A')} |
            <strong>Tenure:</strong> {employee_data.get('Tenure', 'N/A'):.1f} years |
            <strong>Age:</strong> {employee_data.get('Age', 'N/A')}
        </p>
        <div style="
            display: inline-block;
            background-color: {risk_color};
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            margin-top: 10px;
        ">
            {risk_category} Risk: {risk_score:.1%}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_shap_waterfall_chart(
    drivers: list[dict],
    title: str = "What's Driving This Risk?"
) -> None:
    """
    Render a waterfall chart showing factors driving risk for this employee.

    Args:
        drivers: List of dicts with 'feature', 'contribution', 'value'.
        title: Chart title.
    """
    if not drivers:
        st.info("No risk driver data available.")
        return

    # Take top 8 drivers
    top_drivers = drivers[:8]

    features = [d['feature'] for d in top_drivers]
    contributions = [d['contribution'] for d in top_drivers]
    values = [d.get('value', 'N/A') for d in top_drivers]

    # Create colors based on positive/negative contribution
    colors = [RISK_COLORS['High'] if c > 0 else COLORS['accent'] for c in contributions]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=features,
        x=contributions,
        orientation='h',
        marker_color=colors,
        text=[f"{c:+.3f}" for c in contributions],
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>Contribution: %{x:.3f}<extra></extra>"
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Influence on Risk Level",
        yaxis_title="Factor",
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['card_bg'],
        font=dict(color=COLORS['text_primary']),
        height=400,
        margin=dict(l=150, r=50, t=50, b=50),
        yaxis=dict(autorange="reversed")
    )

    st.plotly_chart(fig, use_container_width=True)


def render_driver_table(drivers: list[dict]) -> None:
    """
    Render a table of risk drivers with their contributions.

    Args:
        drivers: List of dicts with 'feature', 'contribution', 'value'.
    """
    if not drivers:
        return

    # Convert to DataFrame for display
    df = pd.DataFrame(drivers[:10])

    # Rename columns for clarity (HR-friendly)
    display_df = df[['feature', 'contribution', 'value']].copy()
    display_df.columns = ['Factor', 'Influence', 'This Employee']

    # Format influence column with direction indicator
    display_df['Influence'] = display_df['Influence'].apply(
        lambda x: f"{'Increases' if x > 0 else 'Decreases'} risk" if pd.notna(x) else "N/A"
    )

    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_recommendations(recommendations: list[str]) -> None:
    """
    Render recommendations for an employee.

    Args:
        recommendations: List of recommendation strings.
    """
    if not recommendations:
        st.info("No specific recommendations for this employee.")
        return

    st.markdown("### 📋 Recommended Actions")

    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"""
        <div style="
            background-color: {COLORS['card_bg']};
            border-left: 3px solid {COLORS['accent']};
            padding: 12px 16px;
            border-radius: 4px;
            margin-bottom: 10px;
        ">
            <strong>{i}.</strong> {rec}
        </div>
        """, unsafe_allow_html=True)


def show_employee_detail_content(
    employee_id: str,
    df: pd.DataFrame,
    ml_engine: Any
) -> None:
    """
    Show detailed employee analysis content.

    Args:
        employee_id: The employee ID to analyze.
        df: Full DataFrame with employee data.
        ml_engine: Trained MLEngine instance.
    """
    # Find employee data
    employee_mask = df['EmployeeID'] == employee_id
    if not employee_mask.any():
        st.error(f"Employee {employee_id} not found.")
        return

    employee_idx = df[employee_mask].index[0]
    employee_data = df.loc[employee_idx]

    # Get features for prediction
    feature_cols = [c for c in df.columns if c not in ['EmployeeID', 'Attrition', 'PerformanceText']]
    numeric_features = df[feature_cols].select_dtypes(include=['int64', 'float64', 'int32', 'float32'])

    # Get risk score
    try:
        risk_scores = ml_engine.predict_risk(numeric_features)
        risk_score = risk_scores[employee_idx] if employee_idx < len(risk_scores) else 0.5
        risk_category = ml_engine.get_risk_category(risk_score)
    except Exception:
        risk_score = 0.5
        risk_category = "Unknown"

    # Render employee card
    render_employee_risk_card(employee_data, risk_score, risk_category)

    # Get and display risk drivers
    st.markdown("---")
    try:
        # Find the position of this employee in the numeric features
        numeric_idx = list(numeric_features.index).index(employee_idx)
        drivers = ml_engine.get_risk_drivers(numeric_idx, numeric_features)
    except Exception:
        drivers = []

    col1, col2 = st.columns([2, 1])

    with col1:
        render_shap_waterfall_chart(drivers)

    with col2:
        st.markdown("### Top Risk Factors")
        render_driver_table(drivers[:5])

    # Recommendations
    st.markdown("---")
    try:
        recommendations = ml_engine.get_recommendations(employee_id, risk_score, drivers)
    except Exception:
        recommendations = []

    render_recommendations(recommendations)


def render_employee_detail_section(
    high_risk_df: pd.DataFrame,
    full_df: pd.DataFrame,
    ml_engine: Any
) -> None:
    """
    Render the employee detail section with selector and expandable details.

    Args:
        high_risk_df: DataFrame with high-risk employees.
        full_df: Full DataFrame with all employees.
        ml_engine: Trained MLEngine instance.
    """
    st.markdown("### 🔍 Individual Employee Analysis")

    selected_employee = render_employee_selector(high_risk_df)

    if selected_employee:
        with st.expander(f"📊 Risk Analysis for {selected_employee}", expanded=True):
            show_employee_detail_content(selected_employee, full_df, ml_engine)
