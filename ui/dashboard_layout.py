"""
Dashboard Layout module for PeopleOS.

Controls the page flow, navigation, and tab structure.
No business logic should be in this module.
"""

from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.ml_engine import MLEngine

import pandas as pd
import streamlit as st

from ui.styling import get_custom_css, get_current_colors, COLORS
from ui.components import (
    render_kpi_card, render_data_table, render_bar_chart,
    render_pie_chart, render_scatter_plot, render_histogram,
    render_heatmap, render_feature_importance_chart,
    render_risk_distribution, render_recommendation_card,
    render_llm_insight_card, render_warning_banner, render_info_banner,
    render_export_section,
    # NLP components
    render_sentiment_distribution, render_skills_chart,
    render_topic_cards, render_sentiment_by_dept_chart,
    # Compensation components
    render_salary_distribution_chart, render_pay_equity_scorecard,
    render_salary_by_tenure_chart,
    # Succession components
    render_succession_pipeline, render_readiness_matrix, render_9box_grid,
    # Team dynamics components
    render_team_health_cards, render_diversity_radar, render_team_composition_chart,
    # Search components
    render_search_results,
    # Executive briefing components
    render_executive_briefing_card, render_loading_placeholder,
    # Insight components
    render_metric_with_insight, render_key_takeaways
)
from ui.employee_detail import render_employee_detail_section


def setup_page_config() -> None:
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title="PeopleOS - People Analytics",
        page_icon="👥",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def inject_custom_css(dark_mode: bool = True) -> None:
    """
    Inject custom CSS into the page.

    Args:
        dark_mode: True for dark mode, False for light mode.
    """
    st.markdown(get_custom_css(dark_mode), unsafe_allow_html=True)


def render_sidebar(
    data_loaded: bool = False,
    features_enabled: Optional[dict] = None,
    session_manager: Any = None
) -> dict:
    """
    Render the sidebar with file upload and options.

    Args:
        data_loaded: Whether data has been loaded.
        features_enabled: Dictionary of enabled features.
        session_manager: SessionManager instance for session persistence.

    Returns:
        Dictionary with sidebar selections.
    """
    with st.sidebar:
        st.title("👥 PeopleOS")
        st.caption("Local-First People Analytics")

        # Theme toggle
        dark_mode = st.toggle(
            "🌙 Dark Mode",
            value=st.session_state.get('dark_mode', True),
            key="theme_toggle"
        )
        st.session_state['dark_mode'] = dark_mode

        st.divider()

        # File upload
        st.subheader("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload HR Data",
            type=['csv', 'json', 'sqlite', 'db'],
            help="Upload CSV, JSON, or SQLite file with employee data"
        )
        
        st.divider()
        
        # Display status
        if data_loaded:
            st.success("✅ Data loaded successfully")
            
            if features_enabled:
                st.subheader("Features Status")
                
                if features_enabled.get('predictive', False):
                    st.markdown("🔮 **Predictive Analytics**: Enabled")
                else:
                    st.markdown("🔮 **Predictive Analytics**: Disabled")
                    st.caption("Upload data with Attrition column")
                
                if features_enabled.get('nlp', False):
                    st.markdown("📝 **NLP Features**: Enabled")
                else:
                    st.markdown("📝 **NLP Features**: Disabled")
                    st.caption("Upload data with PerformanceText column")
        else:
            st.info("Upload data to get started")
        
        st.divider()
        
        # LLM status
        st.subheader("🤖 AI Status")
        llm_status = st.session_state.get('llm_available', False)
        if llm_status:
            st.success("Ollama Connected")
            model_name = st.session_state.get('llm_model', 'Unknown')
            st.caption(f"Model: {model_name}")
        else:
            st.warning("Ollama Not Available")
            st.caption("AI insights will use fallback mode")
        
        st.divider()

        # Session management
        if session_manager is not None:
            with st.expander("💾 Sessions"):
                # Save session
                if data_loaded:
                    session_name = st.text_input(
                        "Session name",
                        placeholder="My analysis",
                        key="session_name_input"
                    )
                    if st.button("Save Session", disabled=not session_name):
                        st.session_state['save_session'] = session_name

                # Load session
                sessions = session_manager.list_sessions()
                if sessions:
                    st.markdown("**Saved Sessions:**")
                    session_options = ["-- Select --"] + [
                        f"{s['session_name']} ({s['created_at'][:10]})"
                        for s in sessions
                    ]
                    selected_idx = st.selectbox(
                        "Load session",
                        options=range(len(session_options)),
                        format_func=lambda x: session_options[x],
                        key="session_select"
                    )
                    if selected_idx > 0 and st.button("Load Selected"):
                        st.session_state['load_session'] = sessions[selected_idx - 1]['filepath']
                else:
                    st.caption("No saved sessions yet")

        st.divider()

        # Data Management
        with st.expander("🗂️ Data Management"):
            st.markdown("**Database Controls**")

            # Show current database stats
            try:
                from src.database import get_database
                db = get_database()
                employee_count = db.get_employee_count()
                st.caption(f"Current employees in database: {employee_count}")
            except Exception:
                employee_count = 0
                st.caption("Database not initialized")

            # Reset database with confirmation
            if 'confirm_reset' not in st.session_state:
                st.session_state.confirm_reset = False

            if not st.session_state.confirm_reset:
                if st.button("🗑️ Reset Database", type="secondary"):
                    st.session_state.confirm_reset = True
                    st.rerun()
            else:
                st.warning("⚠️ This will delete ALL employee data permanently!")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✓ Yes, Reset", type="primary"):
                        try:
                            from src.database import get_database
                            db = get_database()
                            db.clear_all_data()
                            # Reset session state
                            st.session_state.data_loaded = False
                            st.session_state.database_loaded = False
                            st.session_state.raw_data = None
                            st.session_state.processed_data = None
                            st.session_state.confirm_reset = False
                            st.success("Database cleared successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error clearing database: {e}")
                            st.session_state.confirm_reset = False
                with col2:
                    if st.button("✗ Cancel"):
                        st.session_state.confirm_reset = False
                        st.rerun()

        st.divider()

        # About
        with st.expander("ℹ️ About"):
            st.markdown("""
            **PeopleOS** is a local-first People Analytics platform.

            All data processing happens locally -
            your data never leaves your machine.

            Version: 1.1.0
            """)

        return {
            'uploaded_file': uploaded_file
        }


def render_tabs() -> str:
    """
    Render the main navigation tabs.

    Returns:
        Selected tab name.
    """
    tabs = st.tabs([
        "📊 Overview",
        "🔍 Diagnostics",
        "🔮 Future Radar",
        "🔎 Semantic Search",
        "🧠 Strategic Advisor"
    ])

    tab_names = ["Overview", "Diagnostics", "Future Radar", "Semantic Search", "Strategic Advisor"]

    # Store which tab is active
    for i, tab in enumerate(tabs):
        with tab:
            st.session_state['active_tab'] = tab_names[i]

    return st.session_state.get('active_tab', 'Overview')


def render_overview_tab(analytics_data: dict, insight_interpreter=None) -> None:
    """
    Render the Overview tab content.
    
    Args:
        analytics_data: Dictionary with analytics results.
        insight_interpreter: Optional InsightInterpreter for plain language insights.
    """
    st.header("📊 Workforce Overview")
    
    # Key takeaways (if interpreter available)
    if insight_interpreter:
        takeaways = insight_interpreter.get_key_takeaways(analytics_data)
        if takeaways:
            render_key_takeaways(takeaways)
    
    # KPI row with optional insights
    col1, col2, col3, col4 = st.columns(4)
    
    headcount = analytics_data.get('headcount', 0)
    turnover = analytics_data.get('turnover_rate')
    dept_count = analytics_data.get('department_count', 0)
    avg_tenure = analytics_data.get('tenure_mean')
    
    with col1:
        if insight_interpreter:
            insight = insight_interpreter.interpret_metric('headcount', headcount)
            render_metric_with_insight("Total Headcount", headcount, insight)
        else:
            render_kpi_card("Total Headcount", headcount)
    
    with col2:
        if turnover is not None:
            if insight_interpreter:
                insight = insight_interpreter.interpret_metric('turnover_rate', turnover)
                render_metric_with_insight("Turnover Rate", f"{turnover:.1%}", insight)
            else:
                render_kpi_card("Turnover Rate", f"{turnover:.1%}")
        else:
            render_kpi_card("Turnover Rate", "N/A")
    
    with col3:
        if insight_interpreter:
            insight = insight_interpreter.interpret_metric('department_count', dept_count)
            render_metric_with_insight("Departments", dept_count, insight)
        else:
            render_kpi_card("Departments", dept_count)
    
    with col4:
        if avg_tenure:
            if insight_interpreter:
                insight = insight_interpreter.interpret_metric('tenure_mean', avg_tenure)
                render_metric_with_insight("Avg Tenure", f"{avg_tenure:.1f} years", insight)
            else:
                render_kpi_card("Avg Tenure", f"{avg_tenure:.1f} years")
        else:
            render_kpi_card("Avg Tenure", "N/A")
    
    st.divider()
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        if 'dept_stats' in analytics_data and not analytics_data['dept_stats'].empty:
            dept_insight = None
            if insight_interpreter:
                dept_data = analytics_data['dept_stats']
                top_dept = dept_data.loc[dept_data['Headcount'].idxmax(), 'Dept'] if not dept_data.empty else 'Unknown'
                dept_insight = f"'{top_dept}' is your largest department. Consider if staffing levels align with strategic priorities."
            
            render_bar_chart(
                analytics_data['dept_stats'],
                x='Dept',
                y='Headcount',
                title="Headcount by Department",
                insight_text=dept_insight
            )
    
    with col2:
        if 'tenure_distribution' in analytics_data and not analytics_data['tenure_distribution'].empty:
            tenure_insight = None
            if insight_interpreter:
                tenure_insight = "This shows how long employees stay. A healthy mix of new hires and veterans supports knowledge transfer and fresh perspectives."
            
            render_pie_chart(
                analytics_data['tenure_distribution'],
                values='Count',
                names='Tenure_Range',
                title="Tenure Distribution",
                insight_text=tenure_insight
            )
    
    # Salary and demographics
    col1, col2 = st.columns(2)
    
    with col1:
        if 'salary_bands' in analytics_data and not analytics_data['salary_bands'].empty:
            salary_insight = None
            if insight_interpreter:
                salary_insight = "This shows salary distribution across quartiles. A bell curve suggests balanced pay; heavy skewing may indicate equity issues."
            
            render_bar_chart(
                analytics_data['salary_bands'],
                x='Band',
                y='Count',
                title="Salary Distribution by Quartile",
                insight_text=salary_insight
            )
    
    with col2:
        if 'age_distribution' in analytics_data and not analytics_data['age_distribution'].empty:
            age_insight = None
            if insight_interpreter:
                age_insight = "Age diversity supports different perspectives. Plan succession for heavily skewed demographics."
            
            render_bar_chart(
                analytics_data['age_distribution'],
                x='Age_Range',
                y='Count',
                title="Age Distribution",
                insight_text=age_insight
            )

    
    # New Temporal KPIs if available
    if 'temporal_stats' in analytics_data:
        st.divider()
        st.subheader("📈 Temporal Trends")
        t_stats = analytics_data['temporal_stats']
        c1, c2, c3 = st.columns(3)
        with c1:
            render_kpi_card("Avg Rating Velocity", f"{t_stats.get('avg_velocity', 0):.2f}")
        with c2:
            render_kpi_card("Avg Promotion Lag", f"{t_stats.get('avg_promo_lag', 0):.1f} mo")
        with c3:
            render_kpi_card("Avg Salary Growth", f"{t_stats.get('avg_salary_growth', 0):.1%}")


def render_diagnostics_tab(analytics_data: dict, comp_data: dict = None, 
                           succ_data: dict = None, team_data: dict = None, 
                           nlp_data: dict = None, df: pd.DataFrame = None) -> None:
    """
    Render the Diagnostics tab content (consolidated).
    """
    st.header("🔍 Workforce Diagnostics")

    diag_tabs = st.tabs(["Operational Stats", "Compensation", "Succession", "Team Dynamics", "NLP Analysis"])
    
    with diag_tabs[0]:
        # Operational Stats (Original Diagnostics)
        if 'dept_stats' in analytics_data and not analytics_data['dept_stats'].empty:
            st.subheader("Department Analysis")
            render_data_table(analytics_data['dept_stats'], max_rows=50)
        
        if 'high_risk_depts' in analytics_data and not analytics_data['high_risk_depts'].empty:
            st.subheader("⚠️ High-Risk Departments")
            render_data_table(analytics_data['high_risk_depts'])

    with diag_tabs[1]:
        if comp_data:
            render_compensation_tab(comp_data, df)
        else:
            st.info("No compensation data available.")

    with diag_tabs[2]:
        if succ_data:
            render_succession_tab(succ_data)
        else:
            st.info("No succession data available.")

    with diag_tabs[3]:
        if team_data:
            render_team_dynamics_tab(team_data)
        else:
            st.info("No team dynamics data available.")

    with diag_tabs[4]:
        if nlp_data:
            render_nlp_insights_tab(nlp_data, features_enabled=True)
        else:
            st.info("No NLP data available.")


def render_future_radar_tab(
    ml_data: dict,
    features_enabled: bool = True,
    full_df: Optional[pd.DataFrame] = None,
    ml_engine: Optional[Any] = None
) -> None:
    """
    Render the Future Radar (Predictive) tab content.

    Args:
        ml_data: Dictionary with ML results.
        features_enabled: Whether predictive features are enabled.
        full_df: Full DataFrame for employee detail view.
        ml_engine: Trained ML engine for predictions.
    """
    st.header("🔮 Future Radar - Attrition Prediction")
    
    if not features_enabled:
        render_warning_banner(
            "Predictive analytics requires an 'Attrition' column in your data. "
            "Please upload data with historical attrition information."
        )
        return
    
    # Prediction quality metrics with plain English explanations
    if 'metrics' in ml_data:
        st.subheader("How Reliable Are These Predictions?")
        st.caption("These numbers show how well we can predict who might leave based on your historical data")

        metrics = ml_data['metrics']
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            render_metric_with_insight(
                "Correct Predictions", f"{metrics.get('accuracy', 0):.1%}",
                f"Out of all predictions, {metrics.get('accuracy', 0)*100:.0f}% were correct. Higher is better."
            )
        with col2:
            render_metric_with_insight(
                "Alert Accuracy", f"{metrics.get('precision', 0):.1%}",
                "When we flag someone as at-risk, how often we're right. Higher means fewer false alarms."
            )
        with col3:
            render_metric_with_insight(
                "Coverage", f"{metrics.get('recall', 0):.1%}",
                "Of employees who actually left, what percentage did we catch? Higher means fewer surprises."
            )
        with col4:
            reliability_rating = "Excellent" if metrics.get('f1', 0) > 0.8 else "Good" if metrics.get('f1', 0) > 0.6 else "Fair"
            render_metric_with_insight(
                "Overall Reliability", f"{metrics.get('f1', 0):.1%}",
                f"Combined measure of prediction quality. Rating: {reliability_rating}."
            )
    
    st.divider()
    
    # What factors drive risk - plain English
    if 'feature_importances' in ml_data:
        st.subheader("What Factors Matter Most?")
        st.markdown("""
        <div style="background: rgba(59, 130, 246, 0.1); padding: 10px 15px; border-radius: 8px; margin-bottom: 15px; border-left: 3px solid #3b82f6;">
            <span style="color: #3b82f6; font-size: 12px; font-weight: 600;">WHY PEOPLE LEAVE</span>
            <p style="color: #e2e8f0; font-size: 13px; margin: 5px 0 0 0;">
                Based on your data, these factors have the biggest impact on whether employees stay or go.
                Longer bars mean stronger influence. Focus your retention efforts on improving these areas.
            </p>
        </div>
        """, unsafe_allow_html=True)
        render_feature_importance_chart(ml_data['feature_importances'], title="Top Risk Factors")
    
    # Risk distribution
    if 'risk_distribution' in ml_data:
        col1, col2 = st.columns(2)
        
        with col1:
            render_risk_distribution(ml_data['risk_distribution'])
        
        with col2:
            st.subheader("Risk Summary")
            total_emp = sum(ml_data['risk_distribution'].values())
            for category, count in ml_data['risk_distribution'].items():
                pct = (count / total_emp * 100) if total_emp > 0 else 0
                emoji = "🔴" if category == "High" else ("🟡" if category == "Medium" else "🟢")
                st.markdown(f"{emoji} **{category} Risk**: {count} employees ({pct:.0f}%)")
    
    # High risk employees with WHY column
    if 'high_risk_employees' in ml_data and not ml_data['high_risk_employees'].empty:
        st.subheader("⚠️ High-Risk Employees")
        st.markdown("""
        <div style="background: rgba(239, 68, 68, 0.1); padding: 10px 15px; border-radius: 8px; margin-bottom: 15px; border-left: 3px solid #ef4444;">
            <span style="color: #ef4444; font-size: 12px; font-weight: 600;">⚠️ ACTION NEEDED</span>
            <p style="color: #e2e8f0; font-size: 13px; margin: 5px 0 0 0;">
                These employees show elevated risk of leaving based on patterns in your data. 
                Consider proactive engagement: 1-on-1 conversations, career development discussions, or compensation reviews.
            </p>
        </div>
        """, unsafe_allow_html=True)
        render_data_table(ml_data['high_risk_employees'])
        render_export_section(ml_data['high_risk_employees'], "Export High-Risk List", "high_risk_employees")

        # Employee detail analysis
        st.divider()
        if full_df is not None and ml_engine is not None:
            render_employee_detail_section(
                ml_data['high_risk_employees'],
                full_df,
                ml_engine
            )


def render_semantic_search_tab(vector_engine: Any, df: pd.DataFrame = None) -> None:
    """Render the Semantic Search tab."""
    st.header("🔎 Semantic Search & Discovery")
    
    if vector_engine is None or not vector_engine.is_initialized():
        render_warning_banner("Vector engine is not initialized. Please ensure data with 'PerformanceText' is loaded.")
        return

    query = st.text_input("Search workforce by sentiment, skills, or reviews...", 
                         placeholder="e.g., 'exceptional leaders with python skills'")
    
    if query:
        results = vector_engine.search(query, top_k=10)
        if results:
            st.success(f"Found {len(results)} relevant records")
            render_search_results(results)
        else:
            st.info("No relevant records found.")

def render_strategic_advisor_tab(llm_data: dict, analytics_summary: dict,
                                  llm_available: bool = False) -> None:
    """
    Render the Strategic Advisor (LLM) tab content with executive briefing.
    
    Args:
        llm_data: Dictionary with LLM responses including executive_briefing.
        analytics_summary: Summary of analytics for context.
        llm_available: Whether LLM is available.
    """
    st.header("🧠 Strategic Advisor")
    st.caption("AI-Powered Executive Briefing for Leadership Presentations")
    
    if not llm_available:
        render_warning_banner(
            "AI-powered insights require Ollama running locally. "
            "Install Ollama and start the service for full functionality."
        )
    
    st.divider()
    
    # Executive Briefing Section
    if 'executive_briefing' in llm_data:
        render_executive_briefing_card(llm_data['executive_briefing'])
    else:
        # Show loading state or prompt to generate
        if llm_available:
            st.info("Click 'Generate Executive Briefing' to create AI-powered insights.")
        else:
            # Show data-driven fallback
            st.subheader("📊 Data-Driven Summary")
            if analytics_summary:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Workforce", analytics_summary.get('headcount', 0))
                with col2:
                    turnover = analytics_summary.get('turnover_rate')
                    if turnover:
                        st.metric("Turnover Rate", f"{turnover:.1%}")
                with col3:
                    high_risk = analytics_summary.get('high_risk_count', 0)
                    st.metric("High Risk", high_risk)
                with col4:
                    depts = analytics_summary.get('department_count', 0)
                    st.metric("Departments", depts)
    
    st.divider()
    
    # Key metrics context (always shown)
    st.subheader("📈 Supporting Metrics")
    if analytics_summary:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Avg Tenure", f"{analytics_summary.get('tenure_mean', 0):.1f} yrs")
        with col2:
            st.metric("Avg Rating", f"{analytics_summary.get('lastrating_mean', 0):.1f}")
        with col3:
            st.metric("Avg Salary", f"${analytics_summary.get('salary_mean', 0):,.0f}")
        with col4:
            st.metric("Avg Age", f"{analytics_summary.get('age_mean', 0):.0f}")
    
    st.divider()
    
    # Action button
    if st.button("🔄 Generate Executive Briefing", disabled=not llm_available, type="primary"):
        st.session_state['refresh_executive_briefing'] = True
        st.rerun()
    
    # Disclaimer
    st.markdown("""
    <div style="background: rgba(100, 100, 100, 0.1); padding: 10px; border-radius: 5px; 
                margin-top: 20px; font-size: 12px; color: #9ca3af;">
        <strong>Disclaimer:</strong> AI-generated insights are advisory only. 
        All recommendations should be reviewed by qualified HR professionals before implementation.
        No individual employment decisions should be made based solely on this analysis.
    </div>
    """, unsafe_allow_html=True)



def render_nlp_insights_tab(nlp_data: dict, features_enabled: bool = False) -> None:
    """
    Render the NLP Insights tab content.

    Args:
        nlp_data: Dictionary with NLP analysis results.
        features_enabled: Whether NLP features are enabled.
    """
    st.header("📝 NLP Insights - Performance Review Analysis")

    if not features_enabled:
        render_warning_banner(
            "NLP features require a 'PerformanceText' column in your data. "
            "Please upload data with performance review text."
        )
        return

    if not nlp_data:
        render_info_banner("No NLP data available. Processing may still be in progress.")
        return

    # Sentiment Summary KPIs
    if 'sentiment_summary' in nlp_data:
        summary = nlp_data['sentiment_summary']
        st.subheader("Sentiment Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            render_kpi_card("Avg Sentiment", f"{summary.get('avg_sentiment', 0):.2f}")
        with col2:
            render_kpi_card("Positive", f"{summary.get('positive_pct', 0):.0f}%")
        with col3:
            render_kpi_card("Neutral", f"{summary.get('neutral_pct', 0):.0f}%")
        with col4:
            render_kpi_card("Negative", f"{summary.get('negative_pct', 0):.0f}%")

        st.divider()

        # Sentiment charts
        col1, col2 = st.columns(2)

        with col1:
            render_sentiment_distribution(summary)

        with col2:
            if 'sentiment_by_dept' in nlp_data and not nlp_data['sentiment_by_dept'].empty:
                render_sentiment_by_dept_chart(nlp_data['sentiment_by_dept'])

    st.divider()

    # Skills Analysis
    if 'skills' in nlp_data and nlp_data['skills']:
        st.subheader("Skills Identified in Reviews")

        col1, col2 = st.columns(2)

        with col1:
            tech_skills = nlp_data['skills'].get('technical_skills', [])
            if tech_skills:
                st.markdown("**Technical Skills:**")
                for skill in tech_skills[:10]:
                    st.markdown(f"• {skill}")

        with col2:
            soft_skills = nlp_data['skills'].get('soft_skills', [])
            if soft_skills:
                st.markdown("**Soft Skills:**")
                for skill in soft_skills[:10]:
                    st.markdown(f"• {skill}")

        # Skills chart
        render_skills_chart(nlp_data['skills'])

    st.divider()

    # Topics
    if 'topics' in nlp_data and nlp_data['topics']:
        st.subheader("Key Themes in Reviews")
        render_topic_cards(nlp_data['topics'])


def render_compensation_tab(comp_data: dict, df: Optional[pd.DataFrame] = None) -> None:
    """
    Render the Compensation Analysis tab content.

    Args:
        comp_data: Dictionary with compensation analysis results.
        df: Original DataFrame for additional charts.
    """
    st.header("💰 Compensation Analysis")

    if not comp_data:
        render_info_banner("Upload data to view compensation analysis.")
        return

    # Summary KPIs
    if 'summary' in comp_data:
        summary = comp_data['summary']
        st.subheader("Compensation Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            render_kpi_card("Total Payroll", f"${summary.get('total_payroll', 0):,.0f}")
        with col2:
            render_kpi_card("Avg Salary", f"${summary.get('avg_salary', 0):,.0f}")
        with col3:
            render_kpi_card("Median Salary", f"${summary.get('median_salary', 0):,.0f}")
        with col4:
            render_kpi_card("Salary Range", f"${summary.get('salary_range', 0):,.0f}")

    st.divider()

    # Salary distribution
    if df is not None and 'Salary' in df.columns and 'Dept' in df.columns:
        render_salary_distribution_chart(df)

    # Pay equity scorecards
    if 'equity' in comp_data and not comp_data['equity'].empty:
        st.subheader("Pay Equity by Department")
        render_pay_equity_scorecard(comp_data['equity'])

        with st.expander("View Detailed Equity Data"):
            render_data_table(comp_data['equity'])

    st.divider()

    # Salary by tenure
    if 'by_tenure' in comp_data and not comp_data['by_tenure'].empty:
        st.subheader("Salary Progression by Tenure")
        render_salary_by_tenure_chart(comp_data['by_tenure'])

    # Salary outliers
    if 'outliers' in comp_data and not comp_data['outliers'].empty:
        st.subheader("⚠️ Salary Outliers")
        st.caption("Employees with salaries outside 2 standard deviations from department mean")
        render_data_table(comp_data['outliers'])

    # Department comparison
    if 'dept_comparison' in comp_data and not comp_data['dept_comparison'].empty:
        st.subheader("Department Salary Comparison")
        render_data_table(comp_data['dept_comparison'])


def render_succession_tab(succ_data: dict) -> None:
    """
    Render the Succession Planning tab content.

    Args:
        succ_data: Dictionary with succession analysis results.
    """
    st.header("🎯 Succession Planning")

    if not succ_data:
        render_info_banner("Upload data to view succession planning analysis.")
        return

    # Pipeline visualization
    if 'pipeline' in succ_data:
        st.subheader("Succession Pipeline")
        render_succession_pipeline(succ_data['pipeline'])

    st.divider()

    # Bench strength
    if 'bench_strength' in succ_data and not succ_data['bench_strength'].empty:
        st.subheader("Bench Strength by Department")

        # Summary metrics
        bench_df = succ_data['bench_strength']
        strong = len(bench_df[bench_df['Status'] == 'Strong'])
        adequate = len(bench_df[bench_df['Status'] == 'Adequate'])
        weak = len(bench_df[bench_df['Status'] == 'Weak'])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Strong Bench", strong, delta=None)
        with col2:
            st.metric("Adequate Bench", adequate, delta=None)
        with col3:
            st.metric("Weak Bench", weak, delta=None)

        render_data_table(bench_df)

    st.divider()

    # 9-box grid
    if 'nine_box_summary' in succ_data and not succ_data['nine_box_summary'].empty:
        render_9box_grid(succ_data['nine_box_summary'])

    st.divider()

    # Readiness matrix
    if 'readiness' in succ_data and not succ_data['readiness'].empty:
        st.subheader("Readiness Matrix")
        render_readiness_matrix(succ_data['readiness'])

    # High potentials
    if 'high_potentials' in succ_data and not succ_data['high_potentials'].empty:
        st.subheader("⭐ High-Potential Employees")
        render_data_table(succ_data['high_potentials'])

    # Critical gaps
    if 'gaps' in succ_data and not succ_data['gaps'].empty:
        st.subheader("⚠️ Succession Gaps")
        render_data_table(succ_data['gaps'])

    # Recommendations
    if 'recommendations' in succ_data and succ_data['recommendations']:
        st.subheader("Retention Recommendations")
        for rec in succ_data['recommendations'][:5]:
            with st.expander(f"{rec['EmployeeID']} - {rec['Dept']} ({rec['Priority']})"):
                st.markdown(f"**{rec['Recommendation']}**")
                if 'Actions' in rec:
                    for action in rec['Actions']:
                        st.markdown(f"• {action}")


def render_team_dynamics_tab(team_data: dict) -> None:
    """
    Render the Team Dynamics tab content.

    Args:
        team_data: Dictionary with team dynamics analysis results.
    """
    st.header("👥 Team Dynamics")

    if not team_data:
        render_info_banner("Upload data to view team dynamics analysis.")
        return

    # Summary KPIs
    if 'summary' in team_data:
        summary = team_data['summary']
        st.subheader("Organization Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            render_kpi_card("Total Teams", summary.get('total_teams', 0))
        with col2:
            render_kpi_card("Thriving", summary.get('thriving_teams', 0))
        with col3:
            render_kpi_card("At Risk", summary.get('at_risk_teams', 0))
        with col4:
            render_kpi_card("Avg Health", f"{summary.get('avg_health_score', 0):.0%}")

    st.divider()

    # Team health cards
    if 'health' in team_data and not team_data['health'].empty:
        st.subheader("Team Health Scores")
        render_team_health_cards(team_data['health'])

        with st.expander("View Detailed Health Data"):
            render_data_table(team_data['health'])

    st.divider()

    # Diversity metrics
    if 'diversity' in team_data and not team_data['diversity'].empty:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Diversity Metrics")
            render_diversity_radar(team_data['diversity'])

        with col2:
            st.subheader("Diversity by Department")
            render_data_table(team_data['diversity'])

    st.divider()

    # Team composition
    if 'composition' in team_data and not team_data['composition'].empty:
        st.subheader("Team Composition")
        render_team_composition_chart(team_data['composition'])

    # Performance variance
    if 'performance_variance' in team_data and not team_data['performance_variance'].empty:
        st.subheader("Performance Variance by Team")
        render_data_table(team_data['performance_variance'])

    # At-risk teams
    if 'at_risk' in team_data and not team_data['at_risk'].empty:
        st.subheader("⚠️ Teams Requiring Attention")
        render_data_table(team_data['at_risk'])


def render_empty_state() -> None:
    """Render the empty state when no data is loaded - clean & minimal design."""
    dark_mode = st.session_state.get('dark_mode', True)
    colors = get_current_colors(dark_mode)
    border_color = colors.get('border', '#e5e7eb' if not dark_mode else '#334155')

    # Hero section - welcoming and professional
    st.markdown(f"""
    <div style="text-align: center; padding: 48px 20px 32px 20px; max-width: 600px; margin: 0 auto;">
        <div style="font-size: 48px; margin-bottom: 16px;">👥</div>
        <h1 style="color: {colors['text_primary']}; font-size: 28px; font-weight: 600; margin-bottom: 12px;">
            Welcome to PeopleOS
        </h1>
        <p style="font-size: 16px; color: {colors['text_secondary']}; margin-bottom: 8px; line-height: 1.6;">
            Your command center for people analytics.<br/>
            Understand your workforce, predict turnover, and make better decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Upload prompt - clean centered card
    st.markdown(f"""
    <div style="
        max-width: 500px;
        margin: 0 auto 32px auto;
        padding: 32px;
        background-color: {colors['card_bg']};
        border: 1px solid {border_color};
        border-radius: 12px;
        text-align: center;
    ">
        <div style="font-size: 32px; margin-bottom: 16px;">📁</div>
        <h3 style="color: {colors['text_primary']}; font-size: 18px; font-weight: 600; margin-bottom: 8px;">
            Upload Your HR Data
        </h3>
        <p style="color: {colors['text_secondary']}; font-size: 14px; margin-bottom: 16px;">
            Use the sidebar to upload a CSV, JSON, or SQLite file with your employee data.
        </p>
        <div style="display: flex; gap: 16px; justify-content: center; flex-wrap: wrap;">
            <span style="color: {colors['text_secondary']}; font-size: 13px;">
                ✓ All data stays on your computer
            </span>
            <span style="color: {colors['text_secondary']}; font-size: 13px;">
                ✓ No cloud uploads
            </span>
            <span style="color: {colors['text_secondary']}; font-size: 13px;">
                ✓ 100% private
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards - clean grid
    st.markdown("### What You Can Do")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div style="
            background-color: {colors['card_bg']};
            padding: 24px;
            border-radius: 8px;
            border: 1px solid {border_color};
            height: 160px;
        ">
            <div style="font-size: 24px; margin-bottom: 12px;">📊</div>
            <h4 style="color: {colors['text_primary']}; font-size: 16px; font-weight: 600; margin-bottom: 8px;">
                Workforce Analytics
            </h4>
            <p style="color: {colors['text_secondary']}; font-size: 13px; line-height: 1.5;">
                See headcount trends, department breakdowns, salary distributions, and tenure analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="
            background-color: {colors['card_bg']};
            padding: 24px;
            border-radius: 8px;
            border: 1px solid {border_color};
            height: 160px;
        ">
            <div style="font-size: 24px; margin-bottom: 12px;">🔮</div>
            <h4 style="color: {colors['text_primary']}; font-size: 16px; font-weight: 600; margin-bottom: 8px;">
                Predict Turnover
            </h4>
            <p style="color: {colors['text_secondary']}; font-size: 13px; line-height: 1.5;">
                Identify employees at risk of leaving and understand the factors driving attrition.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="
            background-color: {colors['card_bg']};
            padding: 24px;
            border-radius: 8px;
            border: 1px solid {border_color};
            height: 160px;
        ">
            <div style="font-size: 24px; margin-bottom: 12px;">💰</div>
            <h4 style="color: {colors['text_primary']}; font-size: 16px; font-weight: 600; margin-bottom: 8px;">
                Pay Equity Analysis
            </h4>
            <p style="color: {colors['text_secondary']}; font-size: 13px; line-height: 1.5;">
                Analyze compensation fairness, identify outliers, and spot potential inequities.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Quick start - sample data
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("**New to PeopleOS?** Download our sample dataset to explore all features.")

    with col2:
        sample_csv = """EmployeeID,Dept,Tenure,Salary,LastRating,Age,Attrition
EMP001,Engineering,3.5,85000,4.2,32,0
EMP002,Sales,1.2,65000,3.8,28,1
EMP003,HR,5.0,72000,4.5,35,0
EMP004,Marketing,2.3,68000,3.5,30,0
EMP005,Engineering,0.8,78000,3.2,26,1
EMP006,Finance,4.2,82000,4.0,38,0
EMP007,Sales,0.5,58000,3.2,25,1
EMP008,Engineering,6.0,95000,4.8,42,0
"""
        st.download_button(
            label="Download Sample Data",
            data=sample_csv,
            file_name="sample_hr_data.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Data format guide - collapsible
    with st.expander("📋 Data Format Guide"):
        st.markdown("""
**Required columns** - Your data must include these:

| Column | What it is | Example |
|--------|-----------|---------|
| EmployeeID | Unique identifier for each person | "EMP001" or "12345" |
| Dept | Department or team name | "Engineering", "Sales" |
| Tenure | Years at the company | 3.5 |
| Salary | Annual compensation | 85000 |
| LastRating | Most recent performance rating | 4.2 (scale of 1-5) |
| Age | Employee's age | 32 |

**Optional columns** - Add these for more features:

| Column | What it enables |
|--------|-----------------|
| Attrition | Turnover predictions (0=stayed, 1=left) |
| PerformanceText | AI-powered review analysis |

**Supported file formats:** CSV, JSON, SQLite

💡 **Tip:** Most HRIS systems can export data in CSV format. The column names don't need to match exactly - we'll help map them automatically.
        """)


def render_loading_state() -> None:
    """Render a loading state."""
    with st.spinner("Processing your data..."):
        st.empty()


def render_error_state(error_message: str) -> None:
    """
    Render an error state.
    
    Args:
        error_message: Error message to display.
    """
    st.error(error_message)
    
    st.markdown("""
    **What you can do:**
    1. Check that your file is in the correct format (CSV, JSON, or SQLite)
    2. Ensure required columns are present
    3. Verify there are at least 50 rows of data
    4. Check the README for data format requirements
    """)


def render_branded_header(company_name: str = "PeopleOS") -> None:
    """
    Render a branded header with logo placeholder.
    
    Args:
        company_name: Company or product name to display.
    """
    st.markdown(f"""
    <div style="display: flex; align-items: center; padding: 10px 0; margin-bottom: 20px; 
                border-bottom: 1px solid rgba(148, 163, 184, 0.2);">
        <div style="width: 40px; height: 40px; background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
                    border-radius: 8px; display: flex; align-items: center; justify-content: center;
                    font-size: 20px; margin-right: 12px;">
            👥
        </div>
        <div>
            <h2 style="margin: 0; padding: 0; font-size: 24px; color: #f8fafc; font-weight: 700;">
                {company_name}
            </h2>
            <p style="margin: 0; padding: 0; font-size: 12px; color: #94a3b8;">
                Local-First People Analytics Platform
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_app_footer() -> None:
    """Render professional footer with version and disclaimer."""
    st.markdown("""
    <div style="position: fixed; bottom: 0; left: 0; right: 0; 
                background: linear-gradient(180deg, transparent 0%, #0e1117 30%);
                padding: 20px 40px 10px 40px; z-index: 999;">
        <div style="display: flex; justify-content: space-between; align-items: center;
                    max-width: 1200px; margin: 0 auto; padding-top: 10px;
                    border-top: 1px solid rgba(148, 163, 184, 0.1);">
            <div style="color: #64748b; font-size: 11px;">
                PeopleOS v1.0 | Built with Local-First Architecture
            </div>
            <div style="color: #64748b; font-size: 11px;">
                ⚠️ Advisory insights only. Not for individual employment decisions.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
