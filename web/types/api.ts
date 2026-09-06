export interface AnalyticsSummary {
    headcount: number
    record_count?: number
    observed_attrition_share?: number
    turnover_rate?: number // compatibility alias: observed attrition share, not period turnover
    turnover_rate_semantics?: string
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
    total_records?: number
    avg_salary?: number
    median_salary?: number
    salary_std_dev?: number
    avg_tenure?: number
    avg_rating?: number
    avg_age?: number
    observed_attrition_share?: number
    turnover_rate?: number // compatibility alias
}

export interface DepartmentList { departments: DepartmentStats[]; total_departments: number }
export interface PredictionSummary { distribution: { high_risk: number; medium_risk: number; low_risk: number; high_risk_pct: number; medium_risk_pct: number; low_risk_pct: number } }
export interface TenureDistribution { tenure_range: string; count: number; observed_attrition_share?: number; turnover_rate?: number }
export interface AgeDistribution { age_range: string; count: number }
export interface SalaryBand { band: string; lower: number; upper: number; count: number }
export interface DistributionsResponse { tenure: TenureDistribution[]; age: AgeDistribution[]; salary_bands: SalaryBand[] }
export interface CorrelationData { feature: string; correlation: number; abs_correlation: number }
export interface CorrelationsResponse { correlations: CorrelationData[]; target_column: string; metric_semantics?: string }
export interface HighRiskDepartment { dept: string; observed_attrition_share?: number; turnover_rate: number; headcount: number; avg_salary?: number; avg_rating?: number; reason?: string }
export interface HighRiskDepartmentsResponse { departments: HighRiskDepartment[]; threshold: number; threshold_semantics?: string }

export interface UploadStatus { has_data: boolean; employee_count: number; features_enabled: Record<string, boolean>; data?: { loaded: boolean; row_count: number }; engines?: Record<string, boolean> }
export interface UploadResponse { success: boolean; message: string; rows_loaded: number; columns: string[]; features_enabled: Record<string, boolean> }

export interface TeamHealth { dept: string; health_score: number; avg_tenure: number | null; avg_rating: number | null; headcount: number; attrition_rate: number | null; status: string }
export interface DiversityMetrics { dept: string; headcount: number; tenure_diversity: number | null; age_diversity: number | null; salary_equity: number | null; overall_diversity: number }
export interface TeamAnalysis { health: TeamHealth[]; diversity: DiversityMetrics[]; at_risk_teams: any[]; summary: any }
export interface NineBoxSummary { category: string; count: number; percentage: number }
export interface Session { session_name: string; filepath: string; created_at: string; row_count: number; features_enabled: Record<string, boolean> }
export interface SessionListResponse { sessions: Session[]; count: number }

export interface ModelMetrics { accuracy: number; f1: number; precision: number; recall: number; roc_auc?: number | null; brier_score?: number | null; calibration_error?: number | null; best_model: string; reliability: string; warnings: string[] }
export interface EmployeeRiskDetail { employee_id: string; dept: string; tenure: number; salary: number; last_rating: number; age: number; risk_score: number; risk_category: string; drivers: { feature: string; contribution: number; value: number; abs_contribution: number }[]; recommendations: string[]; base_value?: number; confidence?: Record<string, unknown> }

export interface FeatureImportance { feature: string; importance: number }
export interface FeatureImportanceResponse { features: FeatureImportance[]; model_name: string }

export interface SourceEffectiveness { HireSource: string; hire_count: number; pct_of_total: number; avg_performance?: number; high_performers?: number; high_performer_rate?: number; attrition_count?: number; retention_rate?: number; retention_rate_pct?: number; avg_promotions?: number; promoted_count?: number; promotion_rate?: number; avg_tenure?: number; avg_interview_score?: number; quality_score: number; grade: string; recommendation: string }
export interface QualityOfHireAnalysisResult { source_effectiveness: SourceEffectiveness[]; correlations?: any; retention_correlations?: any; insights?: any; cohort_analysis: any[]; new_hire_risks: any[]; summary: { total_employees: number; has_hire_source: boolean; has_interview_scores: boolean; has_assessment: boolean; prehire_signals_count: number; sources_analyzed: number; best_source?: string; top_predictor?: string; new_hires_at_risk: number }; recommendations: string[]; warnings: string[] }

export interface CohortInsight { cohort_name?: string; cohort_description?: string; cohort_size?: number; risk_level?: string; insight?: string; median_tenure?: number; avg_tenure_years?: number; survival_probability_12mo?: number }
export interface SurvivalAnalysisResult { kaplan_meier?: any; kaplan_meier_by_dept?: any; cox_model?: any; hazard_over_time?: any; cohort_insights?: CohortInsight[]; at_risk_employees?: any[]; summary?: { total_employees?: number; attrition_available?: boolean; attrition_count?: number; overall_attrition_rate?: number; cox_model_fitted?: boolean; covariates_used?: string[]; high_risk_count?: number; medium_risk_count?: number; median_tenure?: number; avg_12mo_risk?: number }; recommendations?: string[]; warnings?: string[] }

export interface ScenarioTemplate { name: string; type: string; description?: string; params?: Record<string, unknown> }
