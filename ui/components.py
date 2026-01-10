"""
UI Components module for PeopleOS.

Reusable components for the dashboard including KPI cards,
tables, and charts.
"""

from typing import Any, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ui.styling import COLORS, RISK_COLORS, apply_plotly_theme, get_plotly_theme


def render_kpi_card(label: str, value: Any, subtitle: Optional[str] = None, 
                    delta: Optional[float] = None, color: str = "accent") -> None:
    """
    Render a KPI card with metric.
    
    Args:
        label: KPI label.
        value: KPI value.
        subtitle: Optional subtitle text.
        delta: Optional delta value for change indicator.
        color: Color key from COLORS dict.
    """
    # Use Streamlit's metric for accessibility
    if delta is not None:
        st.metric(label=label, value=value, delta=f"{delta:+.1%}" if isinstance(delta, float) else str(delta))
    else:
        st.metric(label=label, value=value)
    
    if subtitle:
        st.caption(subtitle)


def render_metric_with_insight(label: str, value: Any, insight: str,
                               delta: Optional[float] = None) -> None:
    """
    Render a metric with an inline tooltip showing plain language insight.
    
    Args:
        label: Metric label.
        value: Metric value.
        insight: Plain language explanation of the metric.
        delta: Optional delta value.
    """
    # Render the metric
    if delta is not None:
        st.metric(label=label, value=value, delta=f"{delta:+.1%}" if isinstance(delta, float) else str(delta))
    else:
        st.metric(label=label, value=value)
    
    # Add insight tooltip below metric
    st.markdown(f"""
    <div class="insight-tooltip" style="margin-top: -10px;">
        <span class="tooltip-icon">ℹ️ What does this mean?</span>
        <span class="tooltip-text">
            <div class="insight-label">Plain Language</div>
            {insight}
        </span>
    </div>
    """, unsafe_allow_html=True)


def render_key_takeaways(takeaways: list) -> None:
    """
    Render a list of key takeaways at the top of a section.
    
    Args:
        takeaways: List of plain language insight strings.
    """
    if not takeaways:
        return
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e293b 0%, #2d3555 100%); 
                padding: 15px 20px; border-radius: 10px; margin-bottom: 20px;
                border-left: 4px solid #22c55e;">
        <div style="font-size: 12px; color: #22c55e; font-weight: 600; 
                    text-transform: uppercase; margin-bottom: 10px;">
            Key Takeaways
        </div>
    """, unsafe_allow_html=True)
    
    for takeaway in takeaways:
        st.markdown(f"""
        <div style="color: #e2e8f0; font-size: 14px; margin-bottom: 8px; 
                    padding-left: 5px;">
            {takeaway}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)


def render_risk_badge(risk_category: str) -> str:
    """
    Return HTML for a risk badge.
    
    Args:
        risk_category: "High", "Medium", or "Low".
        
    Returns:
        HTML string for the badge.
    """
    color = RISK_COLORS.get(risk_category, COLORS['text_secondary'])
    return f'<span style="background-color: {color}; color: white; padding: 4px 12px; border-radius: 12px; font-weight: bold;">{risk_category}</span>'


def render_data_table(df: pd.DataFrame, title: Optional[str] = None, 
                      max_rows: int = 100, height: int = 400) -> None:
    """
    Render a styled data table with pagination.
    
    Args:
        df: DataFrame to display.
        title: Optional table title.
        max_rows: Maximum rows to display per page.
        height: Table height in pixels.
    """
    if title:
        st.subheader(title)
    
    if len(df) > max_rows:
        st.info(f"Showing first {max_rows} of {len(df)} rows")
        df = df.head(max_rows)
    
    st.dataframe(df, height=height, use_container_width=True)


def render_bar_chart(df: pd.DataFrame, x: str, y: str, title: str,
                     color: Optional[str] = None, orientation: str = 'v',
                     insight_text: Optional[str] = None) -> None:
    """
    Render a bar chart.
    
    Args:
        df: DataFrame with data.
        x: Column for x-axis.
        y: Column for y-axis.
        title: Chart title.
        color: Optional column for color grouping.
        orientation: 'v' for vertical, 'h' for horizontal.
        insight_text: Optional plain language insight to display below chart.
    """
    fig = px.bar(
        df, x=x, y=y, color=color,
        title=title,
        orientation=orientation,
        color_discrete_sequence=COLORS['chart_palette']
    )
    
    apply_plotly_theme(fig)
    
    # Accessibility: descriptive title
    fig.update_layout(title=title)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show insight if provided
    if insight_text:
        st.markdown(f"""
        <div style="background: rgba(34, 197, 94, 0.1); padding: 10px 15px; 
                    border-radius: 8px; margin-top: -10px; margin-bottom: 15px;
                    border-left: 3px solid #22c55e;">
            <span style="color: #22c55e; font-size: 11px; font-weight: 600;">💡 INSIGHT</span>
            <p style="color: #e2e8f0; font-size: 13px; margin: 5px 0 0 0;">{insight_text}</p>
        </div>
        """, unsafe_allow_html=True)


def render_pie_chart(df: pd.DataFrame, values: str, names: str, title: str,
                     insight_text: Optional[str] = None) -> None:
    """
    Render a pie/donut chart.
    
    Args:
        df: DataFrame with data.
        values: Column for values.
        names: Column for segment names.
        title: Chart title.
        insight_text: Optional plain language insight.
    """
    fig = px.pie(
        df, values=values, names=names,
        title=title,
        color_discrete_sequence=COLORS['chart_palette'],
        hole=0.4  # Donut style
    )
    
    apply_plotly_theme(fig)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show insight if provided
    if insight_text:
        st.markdown(f"""
        <div style="background: rgba(34, 197, 94, 0.1); padding: 10px 15px; 
                    border-radius: 8px; margin-top: -10px; margin-bottom: 15px;
                    border-left: 3px solid #22c55e;">
            <span style="color: #22c55e; font-size: 11px; font-weight: 600;">💡 INSIGHT</span>
            <p style="color: #e2e8f0; font-size: 13px; margin: 5px 0 0 0;">{insight_text}</p>
        </div>
        """, unsafe_allow_html=True)


def render_scatter_plot(df: pd.DataFrame, x: str, y: str, title: str,
                        color: Optional[str] = None, size: Optional[str] = None,
                        hover_data: Optional[list] = None) -> None:
    """
    Render a scatter plot.
    
    Args:
        df: DataFrame with data.
        x: Column for x-axis.
        y: Column for y-axis.
        title: Chart title.
        color: Optional column for color coding.
        size: Optional column for point sizing.
        hover_data: Optional list of columns for hover info.
    """
    # Use WebGL for large datasets (>1000 points)
    render_mode = 'webgl' if len(df) > 1000 else 'svg'
    
    fig = px.scatter(
        df, x=x, y=y,
        color=color, size=size,
        title=title,
        hover_data=hover_data,
        color_discrete_sequence=COLORS['chart_palette'],
        render_mode=render_mode
    )
    
    apply_plotly_theme(fig)
    
    st.plotly_chart(fig, use_container_width=True)


def render_histogram(df: pd.DataFrame, x: str, title: str,
                     nbins: int = 20, color: Optional[str] = None) -> None:
    """
    Render a histogram.

    Args:
        df: DataFrame with data.
        x: Column for histogram.
        title: Chart title.
        nbins: Number of bins.
        color: Optional column for color grouping.
    """
    fig = px.histogram(
        df, x=x,
        title=title,
        nbins=nbins,
        color=color,
        color_discrete_sequence=COLORS['chart_palette']
    )

    apply_plotly_theme(fig)

    # Use WebGL for large datasets
    if len(df) > 500:
        fig.update_traces(marker=dict(line=dict(width=0)))

    st.plotly_chart(fig, use_container_width=True)


def render_heatmap(df: pd.DataFrame, title: str) -> None:
    """
    Render a correlation heatmap.
    
    Args:
        df: DataFrame with correlation values.
        title: Chart title.
    """
    fig = go.Figure(data=go.Heatmap(
        z=df.values,
        x=df.columns.tolist(),
        y=df.index.tolist(),
        colorscale='RdYlGn',
        reversescale=True
    ))
    
    theme = get_plotly_theme()
    fig.update_layout(
        title=title,
        **{k: v for k, v in theme.items() if k != 'colorway'}
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_line_chart(df: pd.DataFrame, x: str, y: str, title: str,
                      color: Optional[str] = None) -> None:
    """
    Render a line chart.

    Args:
        df: DataFrame with data.
        x: Column for x-axis.
        y: Column for y-axis.
        title: Chart title.
        color: Optional column for color grouping.
    """
    # Use WebGL for large datasets
    render_mode = 'webgl' if len(df) > 500 else 'svg'

    fig = px.line(
        df, x=x, y=y, color=color,
        title=title,
        color_discrete_sequence=COLORS['chart_palette'],
        render_mode=render_mode
    )

    apply_plotly_theme(fig)

    st.plotly_chart(fig, use_container_width=True)


def render_feature_importance_chart(importances: pd.DataFrame, title: str = "Feature Importance") -> None:
    """
    Render a horizontal bar chart for feature importance.
    
    Args:
        importances: DataFrame with Feature and Importance columns.
        title: Chart title.
    """
    # Sort by importance
    importances = importances.sort_values('Importance', ascending=True)
    
    fig = px.bar(
        importances,
        x='Importance',
        y='Feature',
        orientation='h',
        title=title,
        color='Importance',
        color_continuous_scale='Greens'
    )
    
    apply_plotly_theme(fig)
    
    st.plotly_chart(fig, use_container_width=True)


def render_risk_distribution(risk_counts: dict, title: str = "Risk Distribution") -> None:
    """
    Render a donut chart showing risk distribution.
    
    Args:
        risk_counts: Dictionary with risk categories and counts.
        title: Chart title.
    """
    df = pd.DataFrame([
        {'Category': k, 'Count': v}
        for k, v in risk_counts.items()
    ])
    
    fig = px.pie(
        df, values='Count', names='Category',
        title=title,
        color='Category',
        color_discrete_map=RISK_COLORS,
        hole=0.4
    )
    
    apply_plotly_theme(fig)
    
    st.plotly_chart(fig, use_container_width=True)


def render_recommendation_card(recommendations: list[str], title: str = "Recommendations") -> None:
    """
    Render a card with recommendations.
    
    Args:
        recommendations: List of recommendation strings.
        title: Card title.
    """
    st.subheader(title)
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"**{i}.** {rec}")


def render_llm_insight_card(insight: dict, title: str = "AI Insights") -> None:
    """
    Render LLM-generated insights.
    
    Args:
        insight: Dictionary with status and message/recommendations.
        title: Card title.
    """
    st.subheader(title)
    
    if insight.get('status') == 'llm_unavailable':
        st.warning(insight.get('message', 'AI insights unavailable'))
        if 'recommendations' in insight:
            for rec in insight['recommendations']:
                st.markdown(f"• {rec}")
    else:
        message = insight.get('message', '')
        if message:
            st.markdown(message)
        
        if 'recommendations' in insight:
            for rec in insight['recommendations']:
                st.markdown(f"• {rec}")
        
        if 'model' in insight:
            st.caption(f"Generated by {insight['model']}")


def render_warning_banner(message: str) -> None:
    """
    Render a warning banner.
    
    Args:
        message: Warning message text.
    """
    st.warning(message)


def render_error_banner(message: str) -> None:
    """
    Render an error banner.
    
    Args:
        message: Error message text.
    """
    st.error(message)


def render_success_banner(message: str) -> None:
    """
    Render a success banner.
    
    Args:
        message: Success message text.
    """
    st.success(message)


def render_info_banner(message: str) -> None:
    """
    Render an info banner.

    Args:
        message: Info message text.
    """
    st.info(message)


def render_download_button(
    data: bytes,
    filename: str,
    label: str = "Download",
    mime_type: str = "text/csv"
) -> None:
    """
    Render a download button for data export.

    Args:
        data: File content as bytes.
        filename: Name for the downloaded file.
        label: Button label text.
        mime_type: MIME type of the file.
    """
    st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime=mime_type
    )


def render_progress_stages(
    stages: list[str],
    current_stage: int,
    completed: bool = False
) -> None:
    """
    Render a progress indicator showing stages of processing.

    Args:
        stages: List of stage names.
        current_stage: Index of current stage (0-based).
        completed: Whether all stages are completed.
    """
    from ui.styling import COLORS

    total = len(stages)
    progress = (current_stage + 1) / total if not completed else 1.0

    st.progress(progress)

    cols = st.columns(total)
    for i, (col, stage) in enumerate(zip(cols, stages)):
        with col:
            if completed or i < current_stage:
                st.markdown(f"✅ {stage}")
            elif i == current_stage:
                st.markdown(f"🔄 **{stage}**")
            else:
                st.markdown(f"⏳ {stage}")


def render_data_quality_score(
    row_count: int,
    column_count: int,
    null_percentage: float,
    features_enabled: dict
) -> None:
    """
    Render a data quality summary card.

    Args:
        row_count: Number of rows in the data.
        column_count: Number of columns.
        null_percentage: Percentage of null values.
        features_enabled: Dict showing which features are enabled.
    """
    from ui.styling import get_current_colors

    dark_mode = st.session_state.get('dark_mode', True)
    colors = get_current_colors(dark_mode)

    # Calculate quality score
    score = 100
    if null_percentage > 10:
        score -= min(30, null_percentage)
    if row_count < 100:
        score -= 20
    if not features_enabled.get('predictive', False):
        score -= 10

    score = max(0, score)
    color = colors['accent'] if score >= 80 else colors['warning'] if score >= 60 else colors['danger']

    st.markdown(f"""
    <div style="
        background-color: {colors['card_bg']};
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    ">
        <h4 style="color: {colors['text_primary']}; margin: 0;">Data Quality Score</h4>
        <div style="
            font-size: 2.5rem;
            font-weight: bold;
            color: {color};
            margin: 10px 0;
        ">{score:.0f}/100</div>
        <div style="color: {colors['text_secondary']};">
            📊 {row_count:,} rows | 📋 {column_count} columns | 🔍 {null_percentage:.1f}% nulls
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_column_mapping_preview(mappings: dict) -> None:
    """
    Render a preview of column mappings.

    Args:
        mappings: Dictionary of original -> mapped column names.
    """
    if not mappings:
        return

    from ui.styling import COLORS

    st.markdown("**Column Mappings:**")
    for original, mapped in mappings.items():
        st.markdown(f"• `{original}` → **{mapped}**")


def render_export_section(
    df: pd.DataFrame,
    title: str = "Export Data",
    prefix: str = "data"
) -> None:
    """
    Render an export section with CSV and Excel options.

    Args:
        df: DataFrame to export.
        title: Section title.
        prefix: Filename prefix for exports.
    """
    from src.export import export_to_csv, export_to_excel, generate_download_filename

    if df is None or df.empty:
        return

    st.markdown(f"**{title}**")
    col1, col2 = st.columns(2)

    with col1:
        csv_data = export_to_csv(df)
        csv_filename = generate_download_filename(prefix, 'csv')
        st.download_button(
            label="📥 Export CSV",
            data=csv_data,
            file_name=csv_filename,
            mime="text/csv",
            key=f"csv_{prefix}"
        )

    with col2:
        excel_data = export_to_excel(df)
        excel_filename = generate_download_filename(prefix, 'xlsx')
        st.download_button(
            label="📥 Export Excel",
            data=excel_data,
            file_name=excel_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"excel_{prefix}"
        )


# ============================================================================
# NLP COMPONENTS
# ============================================================================

def render_sentiment_distribution(sentiment_summary: dict, title: str = "Sentiment Distribution") -> None:
    """
    Render a pie chart showing sentiment distribution.

    Args:
        sentiment_summary: Dictionary with positive_count, neutral_count, negative_count.
        title: Chart title.
    """
    data = pd.DataFrame([
        {'Sentiment': 'Positive', 'Count': sentiment_summary.get('positive_count', 0)},
        {'Sentiment': 'Neutral', 'Count': sentiment_summary.get('neutral_count', 0)},
        {'Sentiment': 'Negative', 'Count': sentiment_summary.get('negative_count', 0)}
    ])

    colors = {'Positive': '#28a745', 'Neutral': '#ffc107', 'Negative': '#dc3545'}

    fig = px.pie(
        data, values='Count', names='Sentiment',
        title=title,
        color='Sentiment',
        color_discrete_map=colors,
        hole=0.4
    )

    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)


def render_skills_chart(skills_data: dict, title: str = "Top Skills Mentioned") -> None:
    """
    Render a bar chart of skills by frequency.

    Args:
        skills_data: Dictionary with skill_counts.
        title: Chart title.
    """
    skill_counts = skills_data.get('skill_counts', {})

    if not skill_counts:
        st.info("No skills data available")
        return

    # Sort and get top 15
    sorted_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:15]

    df = pd.DataFrame(sorted_skills, columns=['Skill', 'Count'])

    fig = px.bar(
        df, x='Count', y='Skill',
        orientation='h',
        title=title,
        color='Count',
        color_continuous_scale='Blues'
    )

    apply_plotly_theme(fig)
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)


def render_topic_cards(topics: list) -> None:
    """
    Render topic theme cards.

    Args:
        topics: List of topic dictionaries with name, description, prevalence.
    """
    if not topics:
        st.info("No topics identified")
        return

    cols = st.columns(min(len(topics), 3))

    for i, topic in enumerate(topics[:6]):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 10px;
                padding: 15px;
                margin: 5px 0;
                color: white;
            ">
                <h4 style="margin: 0; color: white;">{topic.get('name', 'Topic')}</h4>
                <p style="font-size: 0.9rem; margin: 5px 0; opacity: 0.9;">
                    {topic.get('description', '')}
                </p>
                <div style="font-size: 1.2rem; font-weight: bold;">
                    {topic.get('prevalence', 'N/A')}
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_sentiment_by_dept_chart(dept_sentiment: pd.DataFrame) -> None:
    """
    Render sentiment by department bar chart.

    Args:
        dept_sentiment: DataFrame with Dept, AvgSentiment columns.
    """
    if dept_sentiment.empty:
        st.info("No department sentiment data available")
        return

    fig = px.bar(
        dept_sentiment,
        x='Dept',
        y='AvgSentiment',
        title='Average Sentiment by Department',
        color='AvgSentiment',
        color_continuous_scale='RdYlGn',
        range_color=[0, 1]
    )

    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# COMPENSATION COMPONENTS
# ============================================================================

def render_salary_distribution_chart(df: pd.DataFrame, title: str = "Salary Distribution by Department") -> None:
    """
    Render a box plot of salary distribution by department.

    Args:
        df: DataFrame with Dept and Salary columns.
        title: Chart title.
    """
    fig = px.box(
        df, x='Dept', y='Salary',
        title=title,
        color='Dept',
        color_discrete_sequence=COLORS['chart_palette']
    )

    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
    
    # Add insight explanation
    st.markdown("""
    <div style="background: rgba(34, 197, 94, 0.1); padding: 10px 15px; border-radius: 8px; margin-top: -10px; margin-bottom: 15px; border-left: 3px solid #22c55e;">
        <span style="color: #22c55e; font-size: 11px; font-weight: 600;">💡 HOW TO READ THIS</span>
        <p style="color: #e2e8f0; font-size: 13px; margin: 5px 0 0 0;">
            Each box shows salary range for a department. The line in the middle = median salary. 
            Dots outside the whiskers are <strong>outliers</strong> (unusually high or low salaries to investigate).
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_pay_equity_scorecard(equity_df: pd.DataFrame) -> None:
    """
    Render pay equity scorecard.

    Args:
        equity_df: DataFrame with Dept, EquityScore, Status columns.
    """
    if equity_df.empty:
        return

    cols = st.columns(len(equity_df.head(4)))

    for i, (_, row) in enumerate(equity_df.head(4).iterrows()):
        with cols[i]:
            status = row['Status']
            color = '#28a745' if status == 'Good' else '#ffc107' if status == 'Fair' else '#dc3545'

            st.markdown(f"""
            <div style="
                background-color: #1a1a2e;
                border-left: 4px solid {color};
                border-radius: 5px;
                padding: 15px;
                margin: 5px 0;
            ">
                <div style="font-weight: bold; color: white;">{row['Dept']}</div>
                <div style="font-size: 1.5rem; color: {color}; font-weight: bold;">
                    {row['EquityScore']:.0%}
                </div>
                <div style="font-size: 0.8rem; color: #888;">
                    {status}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Add insight explanation
    st.markdown("""
    <div style="background: rgba(34, 197, 94, 0.1); padding: 10px 15px; border-radius: 8px; margin: 10px 0 15px 0; border-left: 3px solid #22c55e;">
        <span style="color: #22c55e; font-size: 11px; font-weight: 600;">💡 WHAT PAY EQUITY MEANS</span>
        <p style="color: #e2e8f0; font-size: 13px; margin: 5px 0 0 0;">
            Higher scores = more consistent pay within the department. Low scores may indicate pay gaps 
            that warrant investigation (e.g., gender, tenure, or role-based disparities).
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_salary_by_tenure_chart(tenure_salary: pd.DataFrame) -> None:
    """
    Render salary progression by tenure chart.

    Args:
        tenure_salary: DataFrame with TenureBucket, Mean, Median columns.
    """
    if tenure_salary.empty:
        return

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=tenure_salary['TenureBucket'],
        y=tenure_salary['Mean'],
        name='Mean Salary',
        marker_color='#667eea'
    ))

    fig.add_trace(go.Scatter(
        x=tenure_salary['TenureBucket'],
        y=tenure_salary['Median'],
        name='Median Salary',
        mode='lines+markers',
        marker_color='#f093fb',
        line=dict(width=3)
    ))

    fig.update_layout(
        title='Salary Progression by Tenure',
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )

    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# SUCCESSION COMPONENTS
# ============================================================================

def render_succession_pipeline(pipeline: dict) -> None:
    """
    Render succession pipeline visualization.

    Args:
        pipeline: Dictionary with department -> readiness level counts.
    """
    if not pipeline:
        st.info("No succession pipeline data available")
        return

    # Convert to DataFrame
    rows = []
    for dept, counts in pipeline.items():
        rows.append({
            'Dept': dept,
            'Ready Now': counts.get('Ready Now', 0),
            'Ready 1-2 Years': counts.get('Ready 1-2 Years', 0),
            'Developing': counts.get('Developing', 0),
            'Early Career': counts.get('Early Career', 0)
        })

    df = pd.DataFrame(rows)

    fig = px.bar(
        df.melt(id_vars=['Dept'], var_name='Readiness', value_name='Count'),
        x='Dept',
        y='Count',
        color='Readiness',
        title='Succession Pipeline by Department',
        color_discrete_map={
            'Ready Now': '#28a745',
            'Ready 1-2 Years': '#17a2b8',
            'Developing': '#ffc107',
            'Early Career': '#6c757d'
        },
        barmode='stack'
    )

    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)


def render_readiness_matrix(readiness_df: pd.DataFrame) -> None:
    """
    Render readiness scatter matrix (tenure vs rating).

    Args:
        readiness_df: DataFrame with Tenure, LastRating, ReadinessLevel columns.
    """
    if readiness_df.empty:
        return

    fig = px.scatter(
        readiness_df,
        x='Tenure',
        y='LastRating',
        color='ReadinessLevel',
        title='Readiness Matrix: Tenure vs Performance',
        hover_data=['EmployeeID', 'Dept'],
        color_discrete_map={
            'Ready Now': '#28a745',
            'Ready 1-2 Years': '#17a2b8',
            'Developing': '#ffc107',
            'Early Career': '#6c757d'
        }
    )

    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)


def render_9box_grid(nine_box_summary: pd.DataFrame) -> None:
    """
    Render 9-box grid visualization.

    Args:
        nine_box_summary: DataFrame with Category, Count, Percentage columns.
    """
    if nine_box_summary.empty:
        return

    # Create a visual 9-box grid representation
    st.markdown("### 9-Box Talent Grid")

    # Grid layout
    grid_order = [
        ['Potential Gems', 'High Potentials', 'Stars'],
        ['Inconsistent', 'Core Contributors', 'High Performers'],
        ['Underperformers', 'Effective', 'Solid Performers']
    ]

    grid_colors = {
        'Stars': '#28a745',
        'High Potentials': '#20c997',
        'High Performers': '#17a2b8',
        'Potential Gems': '#ffc107',
        'Core Contributors': '#6c757d',
        'Solid Performers': '#6f42c1',
        'Inconsistent': '#fd7e14',
        'Effective': '#adb5bd',
        'Underperformers': '#dc3545'
    }

    counts = dict(zip(nine_box_summary['Category'], nine_box_summary['Count']))

    for row in grid_order:
        cols = st.columns(3)
        for i, category in enumerate(row):
            count = counts.get(category, 0)
            color = grid_colors.get(category, '#6c757d')

            with cols[i]:
                st.markdown(f"""
                <div style="
                    background-color: {color};
                    border-radius: 10px;
                    padding: 15px;
                    text-align: center;
                    color: white;
                    margin: 5px;
                ">
                    <div style="font-weight: bold;">{category}</div>
                    <div style="font-size: 1.5rem;">{count}</div>
                </div>
                """, unsafe_allow_html=True)


# ============================================================================
# TEAM DYNAMICS COMPONENTS
# ============================================================================

def render_team_health_cards(health_df: pd.DataFrame) -> None:
    """
    Render team health score cards.

    Args:
        health_df: DataFrame with Dept, HealthScore, Status columns.
    """
    if health_df.empty:
        return

    status_colors = {
        'Thriving': '#28a745',
        'Healthy': '#17a2b8',
        'At Risk': '#ffc107',
        'Critical': '#dc3545'
    }

    cols = st.columns(min(len(health_df), 4))

    for i, (_, row) in enumerate(health_df.head(8).iterrows()):
        with cols[i % 4]:
            status = row['Status']
            color = status_colors.get(status, '#6c757d')

            st.markdown(f"""
            <div style="
                background-color: #1a1a2e;
                border-radius: 10px;
                padding: 15px;
                margin: 5px 0;
                border-left: 4px solid {color};
            ">
                <div style="font-weight: bold; color: white; font-size: 0.9rem;">
                    {row['Dept']}
                </div>
                <div style="font-size: 1.8rem; color: {color}; font-weight: bold;">
                    {row['HealthScore']:.0%}
                </div>
                <div style="font-size: 0.8rem; color: #888;">
                    {status} | {row['Headcount']} employees
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_diversity_radar(diversity_df: pd.DataFrame) -> None:
    """
    Render a radar chart of diversity metrics.

    Args:
        diversity_df: DataFrame with diversity metrics columns.
    """
    if diversity_df.empty:
        return

    # Average across all departments
    metrics = ['AgeDiversity', 'TenureDiversity', 'SalaryEquity']
    available_metrics = [m for m in metrics if m in diversity_df.columns]

    if not available_metrics:
        return

    avg_values = [diversity_df[m].mean() for m in available_metrics]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=avg_values + [avg_values[0]],  # Close the polygon
        theta=available_metrics + [available_metrics[0]],
        fill='toself',
        name='Organization Average',
        line_color='#667eea',
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title='Diversity Metrics Overview'
    )

    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)


def render_team_composition_chart(composition_df: pd.DataFrame) -> None:
    """Render team composition by role/level."""
    if composition_df.empty:
        return

    # Determine available columns (exclude 'Dept')
    available_cols = [col for col in composition_df.columns if col != 'Dept']
    
    if not available_cols:
        st.info("No composition data to display.")
        return

    df_melted = composition_df.melt(
        id_vars=['Dept'],
        value_vars=available_cols,
        var_name='Tenure Group',
        value_name='Count'
    )

    fig = px.bar(
        df_melted,
        x='Dept',
        y='Count',
        color='Tenure Group',
        title='Team Composition by Tenure',
        barmode='stack',
        color_discrete_sequence=['#dc3545', '#ffc107', '#17a2b8', '#28a745']
    )

    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# SEARCH COMPONENTS
# ============================================================================

def render_search_results(results: list[dict]) -> None:
    """Render semantic search results."""
    if not results:
        return
    
    for res in results:
        with st.container():
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(f"**Employee:** {res.get('EmployeeID', 'N/A')} | **Dept:** {res.get('Dept', 'N/A')}")
                text = res.get('PerformanceText', 'No text available.')
                st.markdown(f'<div style="font-style: italic; border-left: 3px solid #667eea; padding-left: 10px;">{text}</div>', unsafe_allow_html=True)
            with c2:
                score = res.get('similarity_score', 0)
                st.progress(score)
                st.caption(f"Match Similarity: {score:.1%}")
            st.divider()


# ============================================================================
# EXECUTIVE BRIEFING COMPONENTS
# ============================================================================

def render_executive_briefing_card(briefing_data: dict) -> None:
    """
    Render executive briefing with 4-section structured layout.
    
    Args:
        briefing_data: Dictionary with 'status' and 'sections' keys.
    """
    if not briefing_data:
        st.info("Executive briefing not available. Generate AI insights to view.")
        return
    
    sections = briefing_data.get('sections', {})
    status = briefing_data.get('status', 'unknown')
    
    if status == 'llm_unavailable':
        st.warning("AI powered insights require Ollama running locally.")
    
    # Section 1: Situation Analysis
    st.markdown("### 📊 Situation Analysis")
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1a1f35 0%, #2d3555 100%); 
                padding: 20px; border-radius: 10px; margin-bottom: 20px;
                border-left: 4px solid #667eea;">
        <p style="font-size: 16px; line-height: 1.6; color: #e2e8f0; margin: 0;">
            {sections.get('situation', 'Analysis pending...')}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Two-column layout for Risks and Opportunities
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔴 Key Risks")
        risks = sections.get('risks', [])
        for i, risk in enumerate(risks[:3], 1):
            st.markdown(f"""
            <div style="background: rgba(239, 68, 68, 0.1); padding: 12px; 
                        border-radius: 8px; margin-bottom: 10px;
                        border-left: 3px solid #ef4444;">
                <span style="color: #ef4444; font-weight: bold;">⚠️ Risk {i}</span>
                <p style="color: #e2e8f0; margin: 5px 0 0 0;">{risk}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🟢 Strategic Opportunities")
        opportunities = sections.get('opportunities', [])
        for i, opp in enumerate(opportunities[:3], 1):
            st.markdown(f"""
            <div style="background: rgba(34, 197, 94, 0.1); padding: 12px; 
                        border-radius: 8px; margin-bottom: 10px;
                        border-left: 3px solid #22c55e;">
                <span style="color: #22c55e; font-weight: bold;">✓ Opportunity {i}</span>
                <p style="color: #e2e8f0; margin: 5px 0 0 0;">{opp}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Section 4: Recommended Actions
    st.markdown("### 🎯 Recommended Actions")
    actions = sections.get('actions', [])
    for i, action in enumerate(actions[:3], 1):
        priority_colors = {1: '#f59e0b', 2: '#3b82f6', 3: '#8b5cf6'}
        priority_labels = {1: 'HIGH PRIORITY', 2: 'MEDIUM PRIORITY', 3: 'STANDARD'}
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e293b 0%, #334155 100%); 
                    padding: 15px; border-radius: 10px; margin-bottom: 10px;
                    border-left: 4px solid {priority_colors.get(i, '#8b5cf6')};">
            <span style="color: {priority_colors.get(i, '#8b5cf6')}; font-weight: bold; 
                         font-size: 12px; text-transform: uppercase;">
                {priority_labels.get(i, 'ACTION')}
            </span>
            <p style="color: #f1f5f9; font-size: 15px; margin: 8px 0 0 0;">{action}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Model attribution
    if briefing_data.get('model'):
        st.caption(f"Generated by {briefing_data['model']} | Advisory purposes only")


def render_loading_placeholder(section_name: str) -> None:
    """Render a placeholder while AI section is loading."""
    st.markdown(f"""
    <div style="background: #1e293b; padding: 20px; border-radius: 10px; 
                text-align: center; margin-bottom: 15px;">
        <div style="color: #667eea;">⏳ Generating {section_name}...</div>
    </div>
    """, unsafe_allow_html=True)
