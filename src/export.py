"""
Export module for PeopleOS.

Provides functionality to export data and reports in CSV and Excel formats.
All exports are user-initiated only.
"""

import io
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from src.logger import get_logger

logger = get_logger('export')


def generate_download_filename(prefix: str, extension: str) -> str:
    """
    Generate a timestamped filename for downloads.

    Args:
        prefix: Filename prefix (e.g., 'risk_report', 'dept_stats').
        extension: File extension without dot (e.g., 'csv', 'xlsx').

    Returns:
        Formatted filename with timestamp.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"peopleos_{prefix}_{timestamp}.{extension}"


def export_to_csv(df: pd.DataFrame) -> bytes:
    """
    Export DataFrame to CSV format.

    Args:
        df: DataFrame to export.

    Returns:
        CSV content as bytes.
    """
    if df is None or df.empty:
        logger.warning("Attempted to export empty DataFrame to CSV")
        return b""

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    logger.info(f"Exported {len(df)} rows to CSV")
    return buffer.getvalue().encode('utf-8')


def export_to_excel(df: pd.DataFrame, sheet_name: str = "Data") -> bytes:
    """
    Export DataFrame to Excel format.

    Args:
        df: DataFrame to export.
        sheet_name: Name of the Excel sheet.

    Returns:
        Excel content as bytes.
    """
    if df is None or df.empty:
        logger.warning("Attempted to export empty DataFrame to Excel")
        return b""

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    logger.info(f"Exported {len(df)} rows to Excel")
    return buffer.getvalue()


def export_risk_report(
    analytics_data: dict,
    ml_data: dict,
    include_high_risk: bool = True
) -> bytes:
    """
    Export a comprehensive risk report to Excel with multiple sheets.

    Args:
        analytics_data: Dictionary with analytics results.
        ml_data: Dictionary with ML prediction results.
        include_high_risk: Whether to include high-risk employee list.

    Returns:
        Excel content as bytes.
    """
    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # Summary sheet
        summary_data = {
            'Metric': [],
            'Value': []
        }

        if analytics_data:
            summary_data['Metric'].extend([
                'Total Headcount',
                'Number of Departments',
                'Turnover Rate'
            ])
            summary_data['Value'].extend([
                analytics_data.get('headcount', 'N/A'),
                analytics_data.get('department_count', 'N/A'),
                f"{analytics_data.get('turnover_rate', 0):.1%}" if analytics_data.get('turnover_rate') else 'N/A'
            ])

        if ml_data and ml_data.get('metrics'):
            metrics = ml_data['metrics']
            summary_data['Metric'].extend([
                'Model Accuracy',
                'Model Precision',
                'Model Recall',
                'Model F1 Score'
            ])
            summary_data['Value'].extend([
                f"{metrics.get('accuracy', 0):.1%}",
                f"{metrics.get('precision', 0):.1%}",
                f"{metrics.get('recall', 0):.1%}",
                f"{metrics.get('f1', 0):.1%}"
            ])

        if ml_data and ml_data.get('risk_distribution'):
            dist = ml_data['risk_distribution']
            summary_data['Metric'].extend([
                'High Risk Employees',
                'Medium Risk Employees',
                'Low Risk Employees'
            ])
            summary_data['Value'].extend([
                dist.get('High', 0),
                dist.get('Medium', 0),
                dist.get('Low', 0)
            ])

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Department stats sheet
        if analytics_data and 'dept_stats' in analytics_data:
            dept_stats = analytics_data['dept_stats']
            if isinstance(dept_stats, pd.DataFrame) and not dept_stats.empty:
                dept_stats.to_excel(writer, sheet_name='Department Stats', index=False)

        # High-risk departments sheet
        if analytics_data and 'high_risk_depts' in analytics_data:
            high_risk_depts = analytics_data['high_risk_depts']
            if isinstance(high_risk_depts, pd.DataFrame) and not high_risk_depts.empty:
                high_risk_depts.to_excel(writer, sheet_name='High Risk Depts', index=False)

        # High-risk employees sheet
        if include_high_risk and ml_data and 'high_risk_employees' in ml_data:
            high_risk_emp = ml_data['high_risk_employees']
            if isinstance(high_risk_emp, pd.DataFrame) and not high_risk_emp.empty:
                high_risk_emp.to_excel(writer, sheet_name='High Risk Employees', index=False)

        # Feature importance sheet
        if ml_data and 'feature_importances' in ml_data:
            feature_imp = ml_data['feature_importances']
            if isinstance(feature_imp, pd.DataFrame) and not feature_imp.empty:
                feature_imp.to_excel(writer, sheet_name='Feature Importance', index=False)

    logger.info("Generated comprehensive risk report")
    return buffer.getvalue()


def export_filtered_employees(
    df: pd.DataFrame,
    risk_category: Optional[str] = None,
    department: Optional[str] = None
) -> bytes:
    """
    Export filtered employee list to CSV.

    Args:
        df: Full employee DataFrame.
        risk_category: Filter by risk category ('High', 'Medium', 'Low').
        department: Filter by department name.

    Returns:
        CSV content as bytes.
    """
    if df is None or df.empty:
        return b""

    filtered_df = df.copy()

    if risk_category and 'Risk_Category' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Risk_Category'] == risk_category]

    if department and 'Dept' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Dept'] == department]

    logger.info(f"Exported filtered employee list: {len(filtered_df)} employees")
    return export_to_csv(filtered_df)


def export_executive_briefing_pdf(
    briefing_data: dict,
    metrics: dict,
    company_name: str = "PeopleOS"
) -> bytes:
    """
    Export executive briefing to PDF format.

    Args:
        briefing_data: Dictionary with executive briefing sections.
        metrics: Dictionary with workforce metrics.
        company_name: Company name for branding.

    Returns:
        PDF content as bytes.
    """
    try:
        from fpdf import FPDF
    except ImportError:
        logger.warning("fpdf2 not installed. PDF export unavailable.")
        return b""

    if not briefing_data or 'sections' not in briefing_data:
        logger.warning("No briefing data to export")
        return b""

    sections = briefing_data.get('sections', {})
    
    class ExecutivePDF(FPDF):
        def header(self):
            self.set_font('Helvetica', 'B', 16)
            self.set_text_color(34, 197, 94)  # Green
            self.cell(0, 10, f'{company_name} - Executive Workforce Briefing', align='C', new_x='LMARGIN', new_y='NEXT')
            self.set_font('Helvetica', '', 10)
            self.set_text_color(100, 100, 100)
            self.cell(0, 5, f'Generated: {datetime.now().strftime("%B %d, %Y")}', align='C', new_x='LMARGIN', new_y='NEXT')
            self.ln(5)
            self.set_draw_color(200, 200, 200)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, 'ADVISORY ONLY - Not for individual employment decisions', align='C')

    pdf = ExecutivePDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Key Metrics Summary
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, 'KEY METRICS', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 10)
    
    metric_items = [
        ('Headcount', metrics.get('headcount', 'N/A')),
        ('Avg Tenure', f"{metrics.get('tenure_mean', 0):.1f} yrs"),
        ('Avg Rating', f"{metrics.get('lastrating_mean', 0):.1f}"),
    ]
    for label, value in metric_items:
        pdf.cell(60, 6, f'{label}: {value}')
    pdf.ln(10)

    # Situation Analysis
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(59, 130, 246)  # Blue
    pdf.cell(0, 8, 'SITUATION ANALYSIS', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(0, 0, 0)
    situation = sections.get('situation', 'Analysis pending.')
    pdf.multi_cell(0, 6, situation)
    pdf.ln(5)

    # Key Risks
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(239, 68, 68)  # Red
    pdf.cell(0, 8, 'KEY RISKS', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(0, 0, 0)
    risks = sections.get('risks', [])
    if risks:
        for i, risk in enumerate(risks[:3], 1):
            if risk:
                pdf.multi_cell(0, 6, f'{i}. {risk}')
    else:
        pdf.multi_cell(0, 6, 'No risks identified.')
    pdf.ln(5)

    # Strategic Opportunities
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(34, 197, 94)  # Green
    pdf.cell(0, 8, 'STRATEGIC OPPORTUNITIES', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(0, 0, 0)
    opps = sections.get('opportunities', [])
    if opps:
        for i, opp in enumerate(opps[:3], 1):
            if opp:
                pdf.multi_cell(0, 6, f'{i}. {opp}')
    else:
        pdf.multi_cell(0, 6, 'No opportunities identified.')
    pdf.ln(5)

    # Recommended Actions
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(245, 158, 11)  # Amber
    pdf.cell(0, 8, 'RECOMMENDED ACTIONS', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(0, 0, 0)
    priority_labels = ['HIGH PRIORITY', 'MEDIUM PRIORITY', 'STANDARD']
    actions = sections.get('actions', [])
    if actions:
        for i, action in enumerate(actions[:3], 1):
            if action:
                priority = priority_labels[i-1] if i <= 3 else 'ACTION'
                pdf.multi_cell(0, 6, f'[{priority}] {action}')
    else:
        pdf.multi_cell(0, 6, 'No actions identified.')

    # Output to bytes
    buffer = io.BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    
    logger.info("Generated executive briefing PDF")
    return buffer.getvalue()
