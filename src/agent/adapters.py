"""Governed aggregate adapters for the People Intelligence Agent.

Evidence describes what a deterministic tool observed. Model quality and
statistical limitations are metadata/caveats; they are not converted into a
pseudo-probability that the evidence is 'true'.
"""

from time import perf_counter
from src.platform.provenance import IntegrityError, validated_risk_scores
from typing import Any, Dict, List

import pandas as pd

from src.agent.evidence import EvidenceItem, EvidenceKind, ToolResult, ToolResultStatus
from src.agent.tools import ToolContext


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 3)


class WorkforceSummaryTool:
    tool_id = 'workforce.summary'
    description = 'Aggregate current workforce headcount, observed attrition share, tenure, rating and salary metrics.'

    def __init__(self, state: Any): self.state = state

    def execute(self, context: ToolContext) -> ToolResult:
        started = perf_counter(); engine = getattr(self.state, 'analytics_engine', None)
        if engine is None:
            return ToolResult(tool_id=self.tool_id, status=ToolResultStatus.BLOCKED, summary='Workforce analytics are unavailable because no dataset is loaded.', error='analytics_engine_unavailable', duration_ms=_elapsed_ms(started))
        try:
            stats = engine.get_summary_statistics(); evidence: List[EvidenceItem] = []
            labels = {
                'headcount': 'Current active employee count', 'record_count': 'Current employee record count',
                'observed_attrition_share': 'Observed attrition share', 'salary_mean': 'Average active-employee salary',
                'tenure_mean': 'Average active-employee tenure', 'age_mean': 'Average active-employee age', 'lastrating_mean': 'Average active-employee rating',
                'department_count': 'Active department count',
            }
            warnings = []
            for metric, label in labels.items():
                value = stats.get(metric)
                metadata = {'confidence_basis': 'deterministic_calculation'}
                if metric.endswith('_mean'):
                    column = metric[:-5]
                    measured = stats.get(f'{column}_observations')
                    excluded = stats.get(f'{column}_excluded_count')
                    metadata.update(population='current active employees', measured_count=measured,
                                    eligible_count=stats.get('headcount'), excluded_count=excluded)
                    if excluded:
                        warnings.append(f'{label} uses {measured} of {stats.get("headcount")} active employees; {excluded} missing or invalid measurements were excluded.')
                    elif value is None:
                        warnings.append(f'{label} is unavailable because no valid measurement is present.')
                if metric == 'observed_attrition_share':
                    metadata.update(
                        population='current employee records with a known attrition outcome',
                        measured_count=stats.get('attrition_known_count'),
                        eligible_count=stats.get('record_count'),
                        not_period_turnover_rate=True,
                    )
                if value is not None:
                    evidence.append(EvidenceItem(kind=EvidenceKind.DERIVED, claim=f'{label}: {value}', source_tool=self.tool_id, value=value, metric=metric, confidence=1.0, dataset_version=context.dataset_version, metadata=metadata))
            return ToolResult(tool_id=self.tool_id, status=ToolResultStatus.SUCCESS, summary='Current workforce summary calculated.', evidence=evidence, warnings=warnings, duration_ms=_elapsed_ms(started), metadata={'metrics': stats, 'metric_contract': 'current_state_active_first'})
        except Exception as exc:
            return ToolResult(tool_id=self.tool_id, status=ToolResultStatus.FAILED, summary='Workforce summary calculation failed.', error=str(exc), duration_ms=_elapsed_ms(started))


class DepartmentRiskTool:
    tool_id = 'workforce.department_risk'
    description = 'Identify departments with elevated observed attrition share without exposing employees.'

    def __init__(self, state: Any): self.state = state

    def execute(self, context: ToolContext) -> ToolResult:
        started = perf_counter(); engine = getattr(self.state, 'analytics_engine', None)
        if engine is None:
            return ToolResult(tool_id=self.tool_id, status=ToolResultStatus.BLOCKED, summary='Department outcome analysis is unavailable.', error='analytics_engine_unavailable', duration_ms=_elapsed_ms(started))
        try:
            all_departments = engine.get_department_aggregates()
            if all_departments is None or all_departments.empty or 'Outcome_Observations' not in all_departments.columns:
                return ToolResult(
                    tool_id=self.tool_id,
                    status=ToolResultStatus.PARTIAL,
                    summary='Department attrition comparison is unavailable because department outcome observations are missing.',
                    warnings=['No department-level known attrition outcomes are available for comparison.'],
                    duration_ms=_elapsed_ms(started),
                    metadata={'departments': [], 'metric_semantics': 'observed_attrition_share_not_period_turnover'},
                )
            observed = pd.to_numeric(all_departments['Outcome_Observations'], errors='coerce').fillna(0)
            if int(observed.sum()) == 0:
                return ToolResult(
                    tool_id=self.tool_id,
                    status=ToolResultStatus.PARTIAL,
                    summary='Department attrition comparison is unavailable because no known attrition outcomes are present.',
                    warnings=['An empty hotspot list would be ambiguous, so PeopleOS is not interpreting missing outcome data as no hotspots.'],
                    duration_ms=_elapsed_ms(started),
                    metadata={'departments': all_departments.to_dict('records'), 'metric_semantics': 'observed_attrition_share_not_period_turnover'},
                )

            threshold = context.parameters.get('turnover_threshold')
            frame = engine.get_high_risk_departments(threshold=threshold)
            if frame is None or frame.empty:
                return ToolResult(
                    tool_id=self.tool_id,
                    status=ToolResultStatus.SUCCESS,
                    summary='No department with observed attrition outcomes exceeds the configured observed-attrition threshold.',
                    evidence=[],
                    duration_ms=_elapsed_ms(started),
                    metadata={'departments': [], 'eligible_department_count': int((observed > 0).sum()), 'metric_semantics': 'observed_attrition_share_not_period_turnover'},
                )
            records = frame.head(10).replace({pd.NA: None}).to_dict('records'); evidence = []
            for row in records:
                dept = str(row.get('Dept', 'Unknown')); share = row.get('Observed_Attrition_Share', row.get('Turnover_Rate'))
                evidence.append(EvidenceItem(
                    kind=EvidenceKind.DERIVED,
                    claim=f'{dept} observed attrition share is {share}',
                    source_tool=self.tool_id,
                    value=share,
                    metric='department_observed_attrition_share',
                    confidence=1.0,
                    dataset_version=context.dataset_version,
                    metadata={
                        'department': dept,
                        'active_headcount': row.get('Headcount'),
                        'record_count': row.get('Total_Records'),
                        'measured_count': row.get('Outcome_Observations'),
                        'eligible_count': row.get('Total_Records'),
                        'confidence_basis': 'deterministic_calculation',
                        'not_period_turnover_rate': True,
                    },
                ))
            return ToolResult(tool_id=self.tool_id, status=ToolResultStatus.SUCCESS, summary=f'Identified {len(records)} department observed-attrition hotspot(s).', evidence=evidence, duration_ms=_elapsed_ms(started), metadata={'departments': records, 'metric_semantics': 'observed_attrition_share_not_period_turnover'})
        except Exception as exc:
            return ToolResult(tool_id=self.tool_id, status=ToolResultStatus.FAILED, summary='Department outcome analysis failed.', error=str(exc), duration_ms=_elapsed_ms(started))


class RetentionRiskTool:
    tool_id = 'workforce.retention_risk'
    description = 'Aggregate predictive model-score distribution with separate model-quality evidence.'

    def __init__(self, state: Any): self.state = state

    def execute(self, context: ToolContext) -> ToolResult:
        started = perf_counter()
        try:
            scores = validated_risk_scores(self.state)
        except IntegrityError as exc:
            return ToolResult(tool_id=self.tool_id, status=ToolResultStatus.PARTIAL,
                summary='Predictive retention model output is unavailable for the current dataset.',
                warnings=[str(exc)], duration_ms=_elapsed_ms(started))
        try:
            counts = scores['risk_category'].value_counts().to_dict(); total = int(len(scores))
            high, medium, low = (int(counts.get(k, 0)) for k in ('High', 'Medium', 'Low'))
            mean_risk = float(scores['risk_score'].mean()) if 'risk_score' in scores.columns else None
            metrics = getattr(self.state, 'model_metrics', None) or {}; evidence: List[EvidenceItem] = []
            prospectively_validated = metrics.get('future_departure_validated') is True
            score_kind = EvidenceKind.DERIVED if prospectively_validated else EvidenceKind.ASSUMED
            common_metadata = {
                'confidence_basis': 'deterministic_summary_of_model_output',
                'population': 'current active employees scored by the active model',
                'measured_count': total,
                'eligible_count': total,
                'future_departure_validated': prospectively_validated,
                'not_observed_attrition': True,
            }
            for label, value, metric in [
                ('Model-classified high-score population', high, 'high_risk_count'),
                ('Model-classified medium-score population', medium, 'medium_risk_count'),
                ('Model-classified low-score population', low, 'low_risk_count'),
            ]:
                evidence.append(EvidenceItem(kind=score_kind, claim=f'{label}: {value} of {total}', source_tool=self.tool_id, value=value, metric=metric, confidence=1.0, dataset_version=context.dataset_version, metadata=common_metadata))
            if mean_risk is not None:
                evidence.append(EvidenceItem(
                    kind=score_kind,
                    claim=f'Mean active-model score: {mean_risk:.3f}',
                    source_tool=self.tool_id,
                    value=mean_risk,
                    metric='mean_risk_score',
                    confidence=1.0,
                    dataset_version=context.dataset_version,
                    metadata={**common_metadata, 'model_probability_requires_calibration_review': True},
                ))
            for key, label in [('f1', 'F1'), ('roc_auc', 'ROC AUC'), ('brier_score', 'Brier score'), ('calibration_error', 'Calibration error')]:
                if metrics.get(key) is not None:
                    evidence.append(EvidenceItem(kind=EvidenceKind.DERIVED, claim=f'Model {label}: {metrics[key]}', source_tool=self.tool_id, value=metrics[key], metric=f'model_{key}', confidence=1.0, dataset_version=context.dataset_version, metadata={'confidence_basis': 'held_out_model_evaluation_metric', 'future_departure_validated': prospectively_validated}))
            warnings = list(metrics.get('warnings', []))
            status = ToolResultStatus.SUCCESS
            summary = 'Aggregate predictive model output calculated with validation state preserved.'
            if not prospectively_validated:
                status = ToolResultStatus.PARTIAL
                warnings.append('Model scores are deterministic outputs of the fitted model but have not been prospectively validated as future-departure probabilities; do not interpret score categories as observed retention facts.')
                summary = 'Aggregate model-score distribution calculated; prospective future-departure validity is not established.'
            return ToolResult(tool_id=self.tool_id, status=status, summary=summary, evidence=evidence, warnings=warnings, duration_ms=_elapsed_ms(started), metadata={'distribution': {'High': high, 'Medium': medium, 'Low': low, 'total': total}, 'model_metrics': metrics, 'future_departure_validated': prospectively_validated})
        except Exception as exc:
            return ToolResult(tool_id=self.tool_id, status=ToolResultStatus.FAILED, summary='Retention-risk analysis failed.', error=str(exc), duration_ms=_elapsed_ms(started))


class CompensationEquityTool:
    tool_id = 'workforce.compensation_equity'
    description = 'Aggregate current compensation dispersion and pay-gap screening evidence.'

    def __init__(self, state: Any): self.state = state

    def execute(self, context: ToolContext) -> ToolResult:
        started = perf_counter(); engine = getattr(self.state, 'compensation_engine', None)
        if engine is None:
            return ToolResult(tool_id=self.tool_id, status=ToolResultStatus.PARTIAL, summary='Compensation analysis is unavailable.', warnings=['Valid salary data may be missing.'], duration_ms=_elapsed_ms(started))
        try:
            evidence: List[EvidenceItem] = []; warnings = list(getattr(engine, 'warnings', [])); metadata: Dict[str, Any] = {}
            dispersion = engine.calculate_pay_equity_score()
            if dispersion is not None and not dispersion.empty:
                records = dispersion.head(10).to_dict('records'); metadata['department_salary_dispersion'] = records
                for row in records:
                    dept = row.get('Dept', 'Unknown')
                    headcount = row.get('Headcount')
                    for metric, label, key in [
                        ('department_salary_cv', 'salary coefficient of variation', 'CV'),
                        ('department_salary_gini', 'salary Gini coefficient', 'Gini'),
                    ]:
                        value = row.get(key)
                        if value is not None:
                            evidence.append(EvidenceItem(
                                kind=EvidenceKind.DERIVED,
                                claim=f'{dept} {label}: {value}',
                                source_tool=self.tool_id,
                                value=value,
                                metric=metric,
                                confidence=1.0,
                                dataset_version=context.dataset_version,
                                metadata={
                                    'department': dept,
                                    'measured_count': headcount,
                                    'eligible_count': headcount,
                                    'confidence_basis': 'deterministic_descriptive_metric',
                                    'not_adjusted_pay_equity': True,
                                    'custom_dispersion_score_not_used_as_agent_evidence': True,
                                },
                            ))
            association = engine.correlate_salary_with_attrition(); metadata['salary_attrition_association'] = association
            if association.get('available'):
                evidence.append(EvidenceItem(kind=EvidenceKind.DERIVED, claim=association.get('interpretation', 'Salary–attrition association calculated.'), source_tool=self.tool_id, value=association.get('correlation'), metric='salary_attrition_association', confidence=1.0, dataset_version=context.dataset_version, metadata={'p_value': association.get('p_value'), 'sample_size': association.get('sample_size'), 'measured_count': association.get('sample_size'), 'confidence_basis': 'statistical_estimate_reported_with_p_value', 'not_causal': True}))
            gap = engine.calculate_gender_pay_gap(); metadata['gender_pay_gap'] = gap
            if gap.get('available'):
                evidence.append(EvidenceItem(
                    kind=EvidenceKind.DERIVED,
                    claim=f"Unadjusted gender pay gap is {gap.get('raw_gap_pct'):.1f}%",
                    source_tool=self.tool_id,
                    value=gap.get('raw_gap_pct'),
                    metric='unadjusted_gender_pay_gap_pct',
                    confidence=1.0,
                    dataset_version=context.dataset_version,
                    metadata={
                        'p_value': gap.get('p_value'),
                        'male_n': gap.get('male_n'),
                        'female_n': gap.get('female_n'),
                        'job_title_stratified_gap_pct': gap.get('job_title_stratified_gap_pct'),
                        'eligible_job_title_strata': gap.get('eligible_job_title_strata'),
                        'confidence_basis': 'descriptive_statistic_with_significance_test',
                        'not_legal_equity_determination': True,
                        'not_regression_adjusted': True,
                    },
                ))
                warnings.append(gap.get('warning'))
            return ToolResult(tool_id=self.tool_id, status=ToolResultStatus.SUCCESS, summary='Compensation disparity evidence calculated with primitive metrics and limitations preserved.', evidence=evidence, warnings=[w for w in warnings if w], duration_ms=_elapsed_ms(started), metadata=metadata)
        except Exception as exc:
            return ToolResult(tool_id=self.tool_id, status=ToolResultStatus.FAILED, summary='Compensation analysis failed.', error=str(exc), duration_ms=_elapsed_ms(started))


class OrganizationStructureTool:
    tool_id = 'workforce.organization_structure'
    description = 'Aggregate span-of-control and role-stagnation hotspots.'

    def __init__(self, state: Any): self.state = state

    def execute(self, context: ToolContext) -> ToolResult:
        started = perf_counter(); engine = getattr(self.state, 'structural_engine', None)
        if engine is None:
            return ToolResult(tool_id=self.tool_id, status=ToolResultStatus.PARTIAL, summary='Organization-structure analysis is unavailable.', warnings=['ManagerID and/or role-tenure fields may be missing.'], duration_ms=_elapsed_ms(started))
        try:
            evidence: List[EvidenceItem] = []; metadata: Dict[str, Any] = {}
            span_analysis = engine.analyze_manager_burnout_risk()
            if span_analysis.get('available'):
                summary = span_analysis.get('summary', {}); metadata['span_of_control'] = {'summary': summary, 'department_summary': span_analysis.get('department_summary', [])}
                evidence.append(EvidenceItem(kind=EvidenceKind.DERIVED, claim=f"Managers above configured span warning threshold: {summary.get('at_risk_count', 0)}", source_tool=self.tool_id, value=summary.get('at_risk_count', 0), metric='manager_span_warning_count', confidence=1.0, dataset_version=context.dataset_version, metadata={'confidence_basis': 'deterministic_configured_threshold', 'not_burnout_measurement': True}))
                if summary.get('avg_span') is not None:
                    evidence.append(EvidenceItem(kind=EvidenceKind.DERIVED, claim=f"Average manager span of control: {summary.get('avg_span')}", source_tool=self.tool_id, value=summary.get('avg_span'), metric='average_span_of_control', confidence=1.0, dataset_version=context.dataset_version, metadata={'confidence_basis': 'deterministic_calculation'}))
            stagnation = engine.identify_stagnation_hotspots()
            if stagnation.get('available'):
                summary = stagnation.get('summary', {}); metadata['stagnation'] = {'summary': summary, 'hotspots': stagnation.get('hotspots', [])}
                evidence.append(EvidenceItem(kind=EvidenceKind.DERIVED, claim=f"Configured critical role-duration threshold count: {summary.get('critical_count', 0)}", source_tool=self.tool_id, value=summary.get('critical_count', 0), metric='critical_stagnation_count', confidence=1.0, dataset_version=context.dataset_version, metadata={'confidence_basis': 'deterministic_configured_threshold', 'not_promotion_readiness': True}))
            if not evidence:
                return ToolResult(tool_id=self.tool_id, status=ToolResultStatus.PARTIAL, summary='Structural engines ran but the dataset lacks enough fields for aggregate findings.', warnings=[x for x in [span_analysis.get('reason', '') if isinstance(span_analysis, dict) else '', stagnation.get('reason', '') if isinstance(stagnation, dict) else ''] if x], duration_ms=_elapsed_ms(started), metadata=metadata)
            return ToolResult(tool_id=self.tool_id, status=ToolResultStatus.SUCCESS, summary='Organization-structure evidence calculated with threshold semantics preserved.', evidence=evidence, duration_ms=_elapsed_ms(started), metadata=metadata)
        except Exception as exc:
            return ToolResult(tool_id=self.tool_id, status=ToolResultStatus.FAILED, summary='Organization-structure analysis failed.', error=str(exc), duration_ms=_elapsed_ms(started))
