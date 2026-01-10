/**
 * HR-Friendly Terminology Mappings
 *
 * This file provides consistent translations from technical/statistical terms
 * to HR-friendly language throughout the PeopleOS application.
 */

export const HR_TERMS = {
  // Navigation & Page Names
  survival_analysis: 'Retention Forecast',
  future_radar: 'Flight Risk',
  diagnostics: 'Workforce Health',
  nlp_analysis: 'Review Insights',

  // Attrition-related
  attrition: 'Departure',
  turnover: 'Turnover',
  flight_risk: 'Flight Risk',
  retention: 'Retention',
  churn: 'Turnover',

  // Risk levels
  high_risk: 'High Risk',
  at_risk: 'At Risk',
  medium_risk: 'Moderate Risk',
  low_risk: 'Low Risk',
  watch_list: 'Watch List',

  // Performance
  high_performer: 'High Performer',
  top_talent: 'Top Talent',
  solid_performer: 'Solid Performer',

  // Statistical terms
  hazard_ratio: 'Risk Multiplier',
  correlation: 'Relationship Strength',
  p_value: 'Statistical Confidence',
  concordance: 'Model Accuracy',
  coefficient: 'Impact Factor',

  // Model metrics
  f1_score: 'Model Accuracy',
  recall: 'Detection Rate',
  precision: 'Prediction Accuracy',
  auc: 'Overall Performance',

  // Survival Analysis terms
  cox_model: 'Risk Factors',
  kaplan_meier: 'Retention Over Time',
  cohorts: 'Employee Groups',
  median_survival: 'Median Time at Company',
  survival_probability: 'Retention Likelihood',

  // Structural terms
  span_of_control: 'Team Size',
  stagnation: 'Role Duration',
  promotion_velocity: 'Career Progression',
}

export const METRIC_EXPLANATIONS: Record<string, {
  name: string
  description: string
  interpretation: {
    good?: string
    warning?: string
    critical?: string
    above_1?: string
    below_1?: string
    example?: string
  }
  benchmark?: string
}> = {
  hazard_ratio: {
    name: 'Risk Multiplier',
    description: 'Shows how much a factor increases or decreases departure risk compared to baseline',
    interpretation: {
      above_1: 'Values above 1.0 mean higher risk of leaving',
      below_1: 'Values below 1.0 mean lower risk of leaving',
      example: 'A value of 1.35 means 35% more likely to leave'
    }
  },
  concordance: {
    name: 'Model Accuracy',
    description: 'How well the model correctly ranks employees by risk',
    interpretation: {
      good: '0.7 or higher = Good predictive power',
      warning: '0.6 - 0.7 = Moderate predictive power',
      critical: 'Below 0.6 = Weak predictive power'
    },
    benchmark: 'Industry standard is 0.65-0.75'
  },
  correlation: {
    name: 'Relationship Strength',
    description: 'Measures how closely two factors are related',
    interpretation: {
      good: '0.5 to 1.0 = Strong positive relationship',
      warning: '0.3 to 0.5 = Moderate relationship',
      critical: 'Below 0.3 = Weak or no relationship'
    }
  },
  f1_score: {
    name: 'Model Accuracy',
    description: 'Balance between catching departures and avoiding false alarms',
    interpretation: {
      good: '0.7 or higher = Reliable predictions',
      warning: '0.5 - 0.7 = Moderate reliability',
      critical: 'Below 0.5 = Needs improvement'
    }
  },
  recall: {
    name: 'Detection Rate',
    description: 'Percentage of actual departures that were correctly predicted',
    interpretation: {
      good: '80%+ = Catching most at-risk employees',
      warning: '60-80% = Missing some at-risk employees',
      critical: 'Below 60% = Missing many at-risk employees'
    }
  },
  precision: {
    name: 'Prediction Accuracy',
    description: 'When we flag someone as at-risk, how often we are correct',
    interpretation: {
      good: '70%+ = Highly accurate flags',
      warning: '50-70% = Some false alarms',
      critical: 'Below 50% = Many false alarms'
    }
  },
  turnover_rate: {
    name: 'Turnover Rate',
    description: 'Percentage of employees who left over a period',
    interpretation: {
      good: 'Below 10% = Healthy retention',
      warning: '10-20% = Moderate concern',
      critical: 'Above 20% = High turnover'
    },
    benchmark: 'Industry average is 15%'
  },
  enps: {
    name: 'Employee Net Promoter Score',
    description: 'Measures employee willingness to recommend the company',
    interpretation: {
      good: '+30 or higher = Excellent',
      warning: '0 to +30 = Good',
      critical: 'Below 0 = Needs attention'
    },
    benchmark: 'Top companies score +50 or higher'
  },
  retention_rate: {
    name: 'Retention Rate',
    description: 'Percentage of employees who stayed over a period',
    interpretation: {
      good: 'Above 90% = Strong retention',
      warning: '80-90% = Moderate retention',
      critical: 'Below 80% = Retention concerns'
    }
  },
  compa_ratio: {
    name: 'Compa Ratio',
    description: 'Salary compared to market midpoint (1.0 = at market)',
    interpretation: {
      good: '0.95 - 1.05 = At market rate',
      warning: '0.8 - 0.95 = Below market',
      critical: 'Below 0.8 = Significantly underpaid'
    }
  }
}

export const CORRELATION_GUIDE = {
  strong_positive: { min: 0.5, max: 1.0, label: 'Strong', color: 'text-success' },
  moderate: { min: 0.3, max: 0.5, label: 'Moderate', color: 'text-accent' },
  weak: { min: 0.1, max: 0.3, label: 'Weak', color: 'text-warning' },
  none: { min: 0, max: 0.1, label: 'None', color: 'text-text-muted' }
}

export function getCorrelationLabel(value: number): { label: string; color: string } {
  const absValue = Math.abs(value)
  if (absValue >= 0.5) return { label: 'Strong', color: 'text-success' }
  if (absValue >= 0.3) return { label: 'Moderate', color: 'text-accent' }
  if (absValue >= 0.1) return { label: 'Weak', color: 'text-warning' }
  return { label: 'None', color: 'text-text-muted' }
}

export function formatHazardRatio(value: number): {
  label: string
  interpretation: string
  color: string
} {
  if (value > 1) {
    const increase = Math.round((value - 1) * 100)
    return {
      label: value.toFixed(2),
      interpretation: `${increase}% more likely to leave`,
      color: 'text-danger'
    }
  } else if (value < 1) {
    const decrease = Math.round((1 - value) * 100)
    return {
      label: value.toFixed(2),
      interpretation: `${decrease}% less likely to leave`,
      color: 'text-success'
    }
  }
  return {
    label: '1.00',
    interpretation: 'No effect on departure risk',
    color: 'text-text-secondary'
  }
}

export function getMetricStatus(metric: string, value: number): 'good' | 'warning' | 'critical' {
  const explanation = METRIC_EXPLANATIONS[metric]
  if (!explanation) return 'warning'

  // Custom logic per metric
  switch (metric) {
    case 'concordance':
    case 'f1_score':
      if (value >= 0.7) return 'good'
      if (value >= 0.6) return 'warning'
      return 'critical'
    case 'recall':
    case 'precision':
      if (value >= 0.7) return 'good'
      if (value >= 0.5) return 'warning'
      return 'critical'
    case 'turnover_rate':
      if (value <= 0.1) return 'good'
      if (value <= 0.2) return 'warning'
      return 'critical'
    case 'retention_rate':
      if (value >= 0.9) return 'good'
      if (value >= 0.8) return 'warning'
      return 'critical'
    case 'enps':
      if (value >= 30) return 'good'
      if (value >= 0) return 'warning'
      return 'critical'
    default:
      return 'warning'
  }
}
