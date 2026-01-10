export interface AnalyticsSummary {
    headcount: number
    turnover_rate?: number
    department_count: number
    salary_mean?: number
    salary_median?: number
    tenure_mean?: number
    tenure_median?: number
    age_mean?: number
    lastrating_mean?: number
    attrition_count?: number
    active_count?: number
    takeaways?: string[]
    insights?: Record<string, string>
}

export interface DepartmentStats {
    dept: string
    headcount: number
    avg_salary?: number
    median_salary?: number
    salary_std_dev?: number
    avg_tenure?: number
    avg_rating?: number
    avg_age?: number
    turnover_rate?: number
}

export interface DepartmentList {
    departments: DepartmentStats[]
    total_departments: number
}

export interface PredictionSummary {
    distribution: {
        high_risk: number
        medium_risk: number
        low_risk: number
        high_risk_pct: number
        medium_risk_pct: number
        low_risk_pct: number
    }
}

export interface TenureDistribution {
    tenure_range: string
    count: number
    turnover_rate?: number
}

export interface AgeDistribution {
    age_range: string
    count: number
}

export interface SalaryBand {
    band: string
    lower: number
    upper: number
    count: number
}

export interface DistributionsResponse {
    tenure: TenureDistribution[]
    age: AgeDistribution[]
    salary_bands: SalaryBand[]
}

export interface CorrelationData {
    feature: string
    correlation: number
    abs_correlation: number
}

export interface CorrelationsResponse {
    correlations: CorrelationData[]
    target_column: string
}

export interface HighRiskDepartment {
    dept: string
    turnover_rate: number
    headcount: number
    avg_salary?: number
    avg_rating?: number
    reason?: string
}

export interface HighRiskDepartmentsResponse {
    departments: HighRiskDepartment[]
    threshold: number
}

export interface UploadStatus {
    has_data: boolean
    employee_count: number
    features_enabled: Record<string, boolean>
    // Extended fields for system status endpoint
    data?: {
        loaded: boolean
        row_count: number
    }
    engines?: Record<string, boolean>
}

export interface UploadResponse {
    success: boolean
    message: string
    rows_loaded: number
    columns: string[]
    features_enabled: Record<string, boolean>
}

export interface TeamHealth {
    dept: string
    health_score: number
    avg_tenure: number | null
    avg_rating: number | null
    headcount: number
    attrition_rate: number | null
    status: string
}

export interface DiversityMetrics {
    dept: string
    headcount: number
    tenure_diversity: number | null
    age_diversity: number | null
    salary_equity: number | null
    overall_diversity: number
}

export interface TeamAnalysis {
    health: TeamHealth[]
    diversity: DiversityMetrics[]
    at_risk_teams: any[]
    summary: any
}

export interface NineBoxSummary {
    category: string
    count: number
    percentage: number
}

export interface Session {
    session_name: string
    filepath: string
    created_at: string
    row_count: number
    features_enabled: Record<string, boolean>
}

export interface SessionListResponse {
    sessions: Session[]
    count: number
}

export interface ModelMetrics {
    accuracy: number
    f1: number
    precision: number
    recall: number
    best_model: string
    reliability: string
    warnings: string[]
}

export interface EmployeeRiskDetail {
    employee_id: string
    dept: string
    tenure: number
    salary: number
    last_rating: number
    age: number
    risk_score: number
    risk_category: string
    drivers: {
        feature: string
        contribution: number
        value: number | null
        abs_contribution: number
    }[]
    recommendations: string[]
    base_value?: number
    confidence?: {
        ci_lower: number | null
        ci_upper: number | null
        confidence_level: number | null
    }
}

export interface FeatureImportance {
    features: {
        feature: string
        importance: number
    }[]
}

export interface SearchResult {
    results: {
        employee_id: string
        dept: string
        text: string
        similarity_score: number
    }[]
}

export interface SearchStatus {
    available: boolean
    reason?: string
    indexed_records: number
}

export interface AdvisorStatus {
    available: boolean
    reason?: string
    model?: string
}

export interface AdvisorSummary {
    summary: string
    key_insights: string[]
    action_items: string[]
    generated_by: string
}

export interface AdvisorAskResponse {
    answer: string
    status: string
}

// ============================================
// Survival / Retention Forecast Types
// ============================================

export interface SurvivalPoint {
    time_months: number
    time_years?: number
    survival_probability: number
    at_risk?: number
}

export interface KaplanMeierResult {
    available: boolean
    reason?: string
    overall?: {
        survival_function: SurvivalPoint[]
        median_survival_months: number | null
        median_survival_years: number | null
        mean_survival_months?: number
        confidence_intervals?: {
            lower: number[]
            upper: number[]
        }
        survival_at_6mo?: number
        survival_at_12mo?: number
        survival_at_24mo?: number
        survival_at_36mo?: number
        survival_at_60mo?: number
    }
    segments?: Record<string, {
        survival_function: SurvivalPoint[]
        median_survival_months: number | null
        sample_size: number
        events: number
    }>
    interpretation?: string[]
}

export interface CoxCoefficient {
    feature: string
    coefficient: number
    hazard_ratio: number
    p_value: number
    is_significant: boolean
    ci_lower: number
    ci_upper: number
    direction: 'increases' | 'decreases'
    interpretation: string
}

export interface CoxModelResult {
    available: boolean
    reason?: string
    coefficients?: Record<string, CoxCoefficient>
    model_metrics?: {
        concordance_index: number
        log_likelihood: number
        aic: number
        sample_size: number
        events: number
        quality_interpretation?: string
    }
    concordance?: number
    log_likelihood?: number
    covariates_used?: string[]
    recommendations?: string[]
}

export interface RiskFactor {
    factor: string
    impact: 'High' | 'Medium' | 'Low'
    direction: 'Increase Risk' | 'Decrease Risk'
    score: number
    description: string
}

export interface AtRiskEmployee {
    EmployeeID: string
    Dept?: string
    Location?: string
    JobTitle?: string
    current_tenure_years: number
    current_rating?: number
    survival_3mo?: number
    survival_6mo?: number
    survival_12mo?: number
    attrition_risk_3mo?: number
    attrition_risk_6mo?: number
    attrition_risk_12mo?: number
    risk_category: 'High' | 'Medium' | 'Low'
    risk_factors?: RiskFactor[]
    YearsSinceLastPromotion?: number
    CompaRatio?: number
}

export interface CohortInsight {
    cohort_description: string
    cohort_name?: string
    cohort_size: number
    filters_applied?: Record<string, any>
    attrition_count?: number
    attrition_rate?: number
    avg_tenure_years?: number
    median_tenure?: number
    survival_probability_12mo?: number
    attrition_probability_12mo?: number
    key_risk_factors?: string[]
    narrative?: string
    insight?: string
    risk_level?: 'High' | 'Medium' | 'Low'
    warning?: string
}

export interface SurvivalAnalysisResult {
    kaplan_meier: KaplanMeierResult
    kaplan_meier_by_dept?: KaplanMeierResult
    cox_model: CoxModelResult
    hazard_over_time?: {
        available: boolean
        reason?: string
        hazard_over_time?: Array<{
            time_years: number
            baseline_hazard: number
            cumulative_hazard: number
            survival: number
        }>
        risk_periods?: Array<{
            time_years: number
            relative_risk: number
            interpretation: string
        }>
    }
    cohort_insights: CohortInsight[]
    at_risk_employees: AtRiskEmployee[]
    summary: {
        total_employees: number
        attrition_available: boolean
        attrition_count: number | null
        overall_attrition_rate: number | null
        cox_model_fitted: boolean
        covariates_used: string[]
        high_risk_count: number
        medium_risk_count: number
        median_tenure: number | null
        avg_12mo_risk: number | null
    }
    recommendations: string[]
    warnings: string[]
}

// ============================================
// Quality of Hire Types
// ============================================

export interface SourceEffectiveness {
    HireSource: string
    hire_count: number
    pct_of_total: number
    avg_performance: number | null
    retention_rate_pct: number | null
    high_performer_rate: number
    quality_score: number
    grade: 'A' | 'B' | 'C' | 'D' | 'F'
    recommendation?: string
}

export interface PreHireCorrelation {
    signal: string
    display_name: string
    correlation: number
    p_value: number
    strength: 'Strong' | 'Moderate' | 'Weak' | 'None'
    is_significant: boolean
    interpretation: string
}

export interface QoHCorrelationsResult {
    available: boolean
    reason?: string
    outcome: string
    correlations?: PreHireCorrelation[]
    best_predictors?: PreHireCorrelation[]
    non_predictors?: PreHireCorrelation[]
    recommendations?: string[]
}

export interface NewHireRisk {
    EmployeeID: string
    HireDate: string
    HireSource: string
    Dept: string
    risk_score: number
    risk_category: 'High' | 'Medium' | 'Low'
    risk_factors: RiskFactor[]
    risk_factors_text: string
    recommendation: string
}

export interface QualityOfHireAnalysisResult {
    source_effectiveness: SourceEffectiveness[]
    correlations: QoHCorrelationsResult
    insights: {
        available: boolean
        best_source?: string
        best_source_score?: number
        top_predictor?: string
        predictive_signals?: string[]
        quality_trend?: string
    }
    cohort_analysis: Array<{
        cohort: string
        employee_count: number
        avg_performance: number
        retention_rate: number
        high_performer_pct: number
        quality_score: number
    }>
    new_hire_risks: NewHireRisk[]
    summary: {
        total_employees: number
        sources_analyzed: number
        prehire_signals_count: number
        new_hires_at_risk: number
        best_source: string | null
        top_predictor: string | null
    }
    recommendations: string[]
    warnings: string[]
}

// ============================================
// Structural Analysis Types
// ============================================

export interface StagnantEmployee {
    EmployeeID: string
    Dept: string
    JobTitle: string
    Tenure: number
    YearsInCurrentRole: number
    StagnationIndex: number
    stagnation_category: 'Critical' | 'High' | 'Moderate' | 'Normal'
    YearsSinceLastPromotion?: number
    LastRating?: number
}

export interface StagnationHotspot {
    segment: string
    segment_type: 'department' | 'job_level'
    total_employees: number
    stagnant_count: number
    stagnation_pct: number
    avg_stagnation_index: number
    severity: 'Critical' | 'High' | 'Medium' | 'Low'
    recommendation: string
}

export interface SpanOfControlResult {
    ManagerID: string
    Dept?: string
    direct_report_count: number
    category: 'Under-Span' | 'Optimal' | 'Over-Span' | 'Critical'
    recommendation: string
}

export interface SpanAnalysis {
    available: boolean
    overall_stats: {
        total_managers: number
        avg_span: number
        median_span: number
        under_span_pct: number
        optimal_pct: number
        over_span_pct: number
    }
    by_category: Array<{
        category: string
        count: number
        percentage: number
    }>
    recommendations: string[]
}

export interface PromotionEquityResult {
    available: boolean
    reason?: string
    analysis_type?: string
    protected_attribute?: string
    reference_group?: string
    results?: Array<{
        group: string
        count: number
        avg_promotion_velocity: number
        gap_vs_reference: number
        pct_promoted: number
    }>
    statistical_test?: {
        test_type: string
        statistic: number
        p_value: number
        is_significant: boolean
    }
    summary?: string
    recommendations?: string[]
}

export interface StructuralAnalysisResult {
    stagnation: {
        hotspots: StagnationHotspot[]
        at_risk_employees: StagnantEmployee[]
        summary: {
            total_analyzed: number
            stagnant_count: number
            stagnation_pct: number
            avg_stagnation_index: number
        }
    }
    span_of_control: SpanAnalysis
    promotion_equity: PromotionEquityResult
    promotion_bottlenecks: Array<{
        job_level: string
        avg_wait_years: number
        bottleneck_severity: string
    }>
    recommendations: string[]
    warnings: string[]
}

// ============================================
// Sentiment Analysis Types
// ============================================

export interface ENPSResult {
    available: boolean
    reason?: string
    overall?: {
        enps_score: number
        promoters_pct: number
        passives_pct: number
        detractors_pct: number
        total_responses: number
        interpretation: string
    }
    by_group?: Record<string, {
        enps_score: number
        response_count: number
        promoters_pct: number
        detractors_pct: number
    }>
    trends?: Array<{
        period: string
        enps_score: number
        response_count: number
        change_from_prior?: number
    }>
    drivers?: Array<{
        dimension: string
        correlation: number
        impact: 'Strong' | 'Moderate' | 'Weak'
        recommendation: string
    }>
    improvement_areas?: Array<{
        dimension: string
        avg_score: number
        benchmark: number
        gap: number
        priority: 'High' | 'Medium' | 'Low'
    }>
}

export interface OnboardingTrajectory {
    EmployeeID: string
    surveys: Array<{
        survey_type: string
        score: number
        date: string
    }>
    trajectory_direction: 'Improving' | 'Declining' | 'Stable'
    at_risk: boolean
    current_score: number
    recommendation: string
}

export interface OnboardingHealth {
    available: boolean
    reason?: string
    overall_health_score?: number
    response_rate?: number
    at_risk_count?: number
    by_dimension?: Record<string, number>
    trends?: Array<{
        survey_type: string
        avg_score: number
        response_count: number
    }>
}

export interface EarlyWarning {
    EmployeeID: string
    Dept?: string
    warning_type: 'eNPS' | 'Onboarding' | 'Combined'
    severity: 'Critical' | 'High' | 'Medium'
    indicators: string[]
    recommended_action: string
}

export interface SentimentAnalysisResult {
    enps: ENPSResult
    onboarding: {
        available: boolean
        health: OnboardingHealth
        trajectories: OnboardingTrajectory[]
    }
    early_warnings: EarlyWarning[]
    summary: {
        enps_available: boolean
        onboarding_available: boolean
        overall_sentiment: 'Positive' | 'Neutral' | 'Negative'
        critical_warnings_count: number
    }
    recommendations: string[]
    warnings: string[]
}

// ============================================
// Common Error Response
// ============================================

export interface APIError {
    error: string
    detail: string
    status_code?: number
}
