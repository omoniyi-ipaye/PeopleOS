'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Card } from '../../components/ui/card'
import { Badge } from '../../components/ui/badge'
import { api } from '../../lib/api-client'
import { cn } from '../../lib/utils'
import {
  Activity,
  ShieldAlert,
  AlertTriangle,
  Users,
  Clock,
  BarChart3,
  Info,
  HelpCircle,
  RefreshCw
} from 'lucide-react'
import {
  ExplanationBox,
  RiskFactorGuide,
  MetricExplainer,
} from '../../components/ui/metric-explainer'
import { RetentionCurveChart } from '../../components/charts/retention-curve-chart'
import { RiskAnalysisModal } from '../../components/risk-analysis-modal'
import type { SurvivalAnalysisResult, AtRiskEmployee, CohortInsight } from '@/types/api'

export default function RetentionForecastPage() {
  const [activeTab, setActiveTab] = useState<'summary' | 'risk-factors' | 'groups' | 'watch-list'>('summary')
  const [selectedEmployee, setSelectedEmployee] = useState<AtRiskEmployee | null>(null)
  const [isModalOpen, setIsModalOpen] = useState(false)

  const { data: analysisData, isLoading, isError, error, refetch } = useQuery<SurvivalAnalysisResult>({
    queryKey: ['survival', 'analysis'],
    queryFn: () => api.survival.getAnalysis() as Promise<SurvivalAnalysisResult>,
  })

  const { data: atRiskData, isError: isAtRiskError } = useQuery<AtRiskEmployee[]>({
    queryKey: ['survival', 'at-risk'],
    queryFn: () => api.survival.getAtRisk(100) as Promise<AtRiskEmployee[]>,
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-pulse-subtle text-text-secondary dark:text-text-dark-secondary">
          Calculating retention forecasts...
        </div>
      </div>
    )
  }

  if (isError) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4 text-center">
        <AlertTriangle className="w-12 h-12 text-danger" />
        <h2 className="text-xl font-semibold text-text-primary dark:text-text-dark-primary">
          Failed to Load Retention Data
        </h2>
        <p className="text-text-secondary dark:text-text-dark-secondary max-w-md">
          {error instanceof Error ? error.message : 'Unable to load retention forecast. Please try again.'}
        </p>
        <button
          onClick={() => refetch()}
          className="flex items-center gap-2 px-4 py-2 bg-accent text-white rounded-lg hover:bg-accent/90 transition-colors"
        >
          <RefreshCw className="w-4 h-4" />
          Retry
        </button>
      </div>
    )
  }

  const kmData = analysisData?.kaplan_meier
  const coxModel = analysisData?.cox_model
  const cohortInsights: CohortInsight[] = analysisData?.cohort_insights || []
  const atRiskEmployees: AtRiskEmployee[] = atRiskData || []
  const summary = analysisData?.summary
  const warnings: string[] = analysisData?.warnings || []
  const recommendations: string[] = analysisData?.recommendations || []

  // Convert Cox coefficients to array for mapping
  const coxCoefficients = coxModel?.coefficients
    ? Object.entries(coxModel.coefficients).map(([key, value]: [string, any]) => ({
      covariate: key,
      ...value
    }))
    : []

  return (
    <div className="space-y-6">
      {/* Page Title */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-text-primary dark:text-text-dark-primary">
            Retention Forecast
          </h1>
          <p className="text-text-secondary dark:text-text-dark-secondary mt-1">
            Predictive modeling of employee tenure and departure trends
          </p>
        </div>

        {/* Tab Switcher */}
        <div className="flex bg-surface dark:bg-surface-dark border border-border dark:border-border-dark p-1 rounded-lg shadow-sm overflow-x-auto no-scrollbar">
          {[
            { id: 'summary', name: 'Summary' },
            { id: 'risk-factors', name: 'Risk Factors' },
            { id: 'groups', name: 'Employee Groups' },
            { id: 'watch-list', name: 'Watch List' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={cn(
                'px-4 py-1.5 text-sm font-medium rounded-md transition-all whitespace-nowrap',
                activeTab === tab.id
                  ? 'bg-accent text-white shadow-sm'
                  : 'text-text-secondary dark:text-text-dark-secondary hover:text-text-primary dark:hover:text-text-dark-primary'
              )}
            >
              {tab.name}
            </button>
          ))}
        </div>
      </div>

      {/* Warnings */}
      {warnings.length > 0 && (
        <div className="bg-warning/10 border border-warning/20 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle className="w-4 h-4 text-warning" />
            <span className="font-medium text-warning">Warnings</span>
          </div>
          <ul className="text-sm text-text-secondary dark:text-text-dark-secondary space-y-1">
            {warnings.map((w: string, i: number) => (
              <li key={i}>- {w}</li>
            ))}
          </ul>
        </div>
      )}

      {activeTab === 'summary' && (
        <>
          {/* Explanation Box */}
          <ExplanationBox title="What is Retention Forecasting?">
            <p>
              Retention forecasting uses historical data to predict how long employees are likely to stay with the company.
              Instead of just looking at who left, it helps us anticipate future departures so we can intervene early.
            </p>
          </ExplanationBox>
          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card className="p-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-accent/10 rounded-lg">
                  <Users className="w-5 h-5 text-accent" />
                </div>
                <div>
                  <p className="text-xs text-text-muted dark:text-text-dark-muted uppercase tracking-wider">
                    Employees Analyzed
                  </p>
                  <p className="text-2xl font-bold text-text-primary dark:text-text-dark-primary">
                    {summary?.total_employees ?? 0}
                  </p>
                </div>
              </div>
            </Card>

            <Card className="p-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-success/10 rounded-lg">
                  <Activity className="w-5 h-5 text-success" />
                </div>
                <div>
                  <p className="text-xs text-text-muted dark:text-text-dark-muted uppercase tracking-wider">
                    Median Time at Company
                  </p>
                  <p className="text-2xl font-bold text-text-primary dark:text-text-dark-primary">
                    {summary?.median_tenure ? `${summary.median_tenure.toFixed(1)} years` : '> 15 years'}
                  </p>
                </div>
              </div>
            </Card>

            <Card className="p-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-danger/10 rounded-lg">
                  <ShieldAlert className="w-5 h-5 text-danger" />
                </div>
                <div>
                  <p className="text-xs text-text-muted dark:text-text-dark-muted uppercase tracking-wider">
                    Employees at High Risk
                  </p>
                  <p className="text-2xl font-bold text-danger">
                    {summary?.high_risk_count ?? 0}
                  </p>
                </div>
              </div>
            </Card>

            <Card className="p-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-warning/10 rounded-lg">
                  <Clock className="w-5 h-5 text-warning" />
                </div>
                <div>
                  <p className="text-xs text-text-muted dark:text-text-dark-muted uppercase tracking-wider">
                    12-Month Retention Chance
                  </p>
                  <p className="text-2xl font-bold text-text-primary dark:text-text-dark-primary">
                    {summary?.avg_12mo_risk !== undefined && summary?.avg_12mo_risk !== null
                      ? `${((1 - summary.avg_12mo_risk) * 100).toFixed(0)}%`
                      : 'N/A'}
                  </p>
                </div>
              </div>
            </Card>
          </div>

          <Card title="Retention Over Time" subtitle="Overall employee likelihood of staying with the company">
            {kmData?.overall?.survival_function && kmData.overall.survival_function.length > 0 ? (
              <div className="pt-4">
                <RetentionCurveChart data={kmData.overall.survival_function} />
                <div className="mt-4 grid grid-cols-2 md:grid-cols-5 gap-4">
                  {[
                    { mo: 6, key: 'survival_at_6mo' as const },
                    { mo: 12, key: 'survival_at_12mo' as const },
                    { mo: 24, key: 'survival_at_24mo' as const },
                    { mo: 36, key: 'survival_at_36mo' as const },
                    { mo: 60, key: 'survival_at_60mo' as const },
                  ].map(({ mo, key }) => {
                    const prob = kmData.overall?.[key]
                    return (
                      <div key={mo} className="text-center p-2 bg-surface-secondary dark:bg-background-dark rounded-lg">
                        <p className="text-[10px] text-text-muted uppercase tracking-tighter">{mo}mo Retention</p>
                        <p className="text-sm font-bold text-text-primary dark:text-text-dark-primary">
                          {prob !== undefined ? `${(prob * 100).toFixed(0)}%` : 'N/A'}
                        </p>
                      </div>
                    )
                  })}
                </div>
              </div>
            ) : (
              <div className="h-64 flex flex-col items-center justify-center text-text-secondary">
                <p>No survival curve data available</p>
                {kmData?.available === false && <p className="text-xs mt-2 text-danger">{kmData?.reason}</p>}
              </div>
            )}
          </Card>

          {/* Recommendations */}
          {recommendations.length > 0 && (
            <Card title="Recommendations" subtitle="Strategic actions based on survival analysis">
              <div className="space-y-3">
                {recommendations.map((rec: string, i: number) => (
                  <div key={i} className="flex items-start gap-3 p-3 bg-accent/5 rounded-lg">
                    <Info className="w-4 h-4 text-accent mt-0.5 shrink-0" />
                    <p className="text-sm text-text-secondary dark:text-text-dark-secondary">
                      {rec}
                    </p>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </>
      )}

      {activeTab === 'risk-factors' && (
        <Card title="Key Risk Factors" subtitle="Factors mathematically linked to employee departure">
          {coxModel && coxModel.coefficients ? (
            <div className="space-y-6">
              <RiskFactorGuide />

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 bg-surface-secondary dark:bg-background-dark rounded-lg">
                  <p className="text-xs text-text-muted uppercase tracking-wider mb-1">Model Accuracy</p>
                  <p className="text-xl font-bold text-text-primary dark:text-text-dark-primary">
                    {((coxModel.concordance || 0) * 100).toFixed(1)}%
                  </p>
                  <p className="text-xs text-text-muted mt-1">How well the model identifies at-risk employees</p>
                </div>
                <div className="p-4 bg-surface-secondary dark:bg-background-dark rounded-lg">
                  <p className="text-xs text-text-muted uppercase tracking-wider mb-1">Log Likelihood</p>
                  <p className="text-xl font-bold text-text-primary dark:text-text-dark-primary">
                    {coxModel.log_likelihood?.toFixed(2) || 'N/A'}
                  </p>
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-border dark:border-border-dark">
                      <th className="text-left py-3 px-4 text-xs font-semibold text-text-muted uppercase">
                        Factor
                      </th>
                      <th className="text-right py-3 px-4 text-xs font-semibold text-text-muted uppercase">
                        Risk Multiplier
                      </th>
                      <th className="text-right py-3 px-4 text-xs font-semibold text-text-muted uppercase">
                        Impact
                      </th>
                      <th className="text-right py-3 px-4 text-xs font-semibold text-text-muted uppercase">
                        Confidence
                      </th>
                      <th className="text-center py-3 px-4 text-xs font-semibold text-text-muted uppercase">
                        What to do
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {coxCoefficients.map((coef: any, i: number) => (
                      <tr
                        key={i}
                        className="border-b border-border/50 dark:border-border-dark/50 hover:bg-surface-hover dark:hover:bg-surface-dark-hover"
                      >
                        <td className="py-3 px-4 font-medium text-text-primary dark:text-text-dark-primary capitalize">
                          {coef.covariate?.replace(/_/g, ' ')}
                        </td>
                        <td className="py-3 px-4 text-right font-mono">
                          <span className={cn(
                            coef.hazard_ratio > 1 ? 'text-danger' : 'text-success'
                          )}>
                            {coef.hazard_ratio?.toFixed(2)}x
                          </span>
                        </td>
                        <td className="py-3 px-4 text-right text-sm">
                          <span className={coef.hazard_ratio > 1 ? 'text-danger' : 'text-success'}>
                            {coef.hazard_ratio > 1
                              ? `${((coef.hazard_ratio - 1) * 100).toFixed(0)}% more likely to leave`
                              : `${((1 - coef.hazard_ratio) * 100).toFixed(0)}% less likely to leave`
                            }
                          </span>
                        </td>
                        <td className="py-3 px-4 text-right text-sm text-text-secondary">
                          {coef.p_value < 0.05 ? 'High Confidence' : 'Low Confidence'}
                        </td>
                        <td className="py-3 px-4 text-center">
                          <Badge variant={coef.significant ? 'danger' : 'outline'}>
                            {coef.hazard_ratio > 1 ? 'Action Needed' : 'Monitor'}
                          </Badge>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="mt-4 p-4 bg-accent/5 border border-accent/20 rounded-lg">
                <p className="text-xs text-text-muted mb-2 font-bold uppercase tracking-wider">Why this matters</p>
                <p className="text-xs text-text-secondary leading-relaxed">
                  These factors represent the strongest drivers of retention in your organization. High multipliers (red) indicate areas where intervention could significantly reduce turnover, while low multipliers (green) highlight what's currently working well to keep employees engaged.
                </p>
              </div>
            </div>
          ) : (
            <div className="py-12 text-center text-text-secondary">
              Cox model data not available. Ensure data has Tenure and Attrition columns.
            </div>
          )}
        </Card>
      )}

      {activeTab === 'groups' && (
        <Card title="Employee Group Insights" subtitle="Understanding retention patterns across different segments">
          {cohortInsights.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {cohortInsights.map((cohort: any, i: number) => (
                <div
                  key={i}
                  className="p-4 border border-border dark:border-border-dark rounded-lg hover:shadow-md transition-shadow"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h4 className="font-bold text-text-primary dark:text-text-dark-primary">
                      {cohort.cohort_name}
                    </h4>
                    <Badge variant={
                      cohort.risk_level === 'High' ? 'danger' :
                        cohort.risk_level === 'Medium' ? 'warning' : 'success'
                    }>
                      {cohort.risk_level} Risk
                    </Badge>
                  </div>

                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-text-muted">Employees</span>
                      <span className="font-mono">{cohort.employee_count}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-text-muted">Median Tenure</span>
                      <span className="font-mono">{cohort.median_tenure?.toFixed(1)} yrs</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-text-muted">12-mo Retention Chance</span>
                      <span className="font-mono text-success">
                        {cohort.survival_probability_12mo
                          ? `${(cohort.survival_probability_12mo * 100).toFixed(0)}%`
                          : 'N/A'}
                      </span>
                    </div>
                  </div>

                  {cohort.insight && (
                    <p className="mt-3 text-xs text-text-secondary italic border-t border-border/50 pt-3">
                      {cohort.insight}
                    </p>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="py-12 text-center text-text-secondary">
              No cohort insights available
            </div>
          )}
        </Card>
      )}

      {activeTab === 'watch-list' && (
        <Card title="Employee Watch List" subtitle="Individuals with the highest predicted departure risk">
          {atRiskEmployees.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border dark:border-border-dark">
                    <th className="text-left py-3 px-4 text-xs font-semibold text-text-muted uppercase">
                      Employee
                    </th>
                    <th className="text-left py-3 px-4 text-xs font-semibold text-text-muted uppercase">
                      Department
                    </th>
                    <th className="text-right py-3 px-4 text-xs font-semibold text-text-muted uppercase">
                      Tenure
                    </th>
                    <th className="text-right py-3 px-4 text-xs font-semibold text-text-muted uppercase">
                      3-Month Chance
                    </th>
                    <th className="text-right py-3 px-4 text-xs font-semibold text-text-muted uppercase">
                      6-Month Chance
                    </th>
                    <th className="text-right py-3 px-4 text-xs font-semibold text-text-muted uppercase">
                      12-Month Chance
                    </th>
                    <th className="text-center py-3 px-4 text-xs font-semibold text-text-muted uppercase">
                      Category
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {atRiskEmployees.map((emp: any, i: number) => (
                    <tr
                      key={i}
                      onClick={() => {
                        setSelectedEmployee(emp)
                        setIsModalOpen(true)
                      }}
                      className="border-b border-border/50 dark:border-border-dark/50 hover:bg-surface-hover dark:hover:bg-surface-dark-hover cursor-pointer transition-colors group"
                    >
                      <td className="py-3 px-4 font-medium text-text-primary dark:text-text-dark-primary group-hover:text-accent">
                        {emp.EmployeeID}
                      </td>
                      <td className="py-3 px-4 text-text-secondary">
                        {emp.Dept || '-'}
                      </td>
                      <td className="py-3 px-4 text-right font-mono">
                        {emp.current_tenure_years?.toFixed(1)} yrs
                      </td>
                      <td className="py-3 px-4 text-right font-mono">
                        <span className={cn(
                          emp.attrition_risk_3mo > 0.3 ? 'text-danger' :
                            emp.attrition_risk_3mo > 0.15 ? 'text-warning' : 'text-success'
                        )}>
                          {emp.attrition_risk_3mo
                            ? `${((1 - emp.attrition_risk_3mo) * 100).toFixed(0)}%`
                            : '-'}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-right font-mono">
                        <span className={cn(
                          emp.attrition_risk_6mo > 0.4 ? 'text-danger' :
                            emp.attrition_risk_6mo > 0.2 ? 'text-warning' : 'text-success'
                        )}>
                          {emp.attrition_risk_6mo
                            ? `${((1 - emp.attrition_risk_6mo) * 100).toFixed(0)}%`
                            : '-'}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-right font-mono">
                        <span className={cn(
                          emp.attrition_risk_12mo > 0.5 ? 'text-danger' :
                            emp.attrition_risk_12mo > 0.3 ? 'text-warning' : 'text-success'
                        )}>
                          {emp.attrition_risk_12mo
                            ? `${((1 - emp.attrition_risk_12mo) * 100).toFixed(0)}%`
                            : '-'}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-center">
                        <Badge variant={
                          emp.risk_category === 'High' ? 'danger' :
                            emp.risk_category === 'Medium' ? 'warning' : 'success'
                        }>
                          {emp.risk_category}
                        </Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="py-12 text-center text-text-secondary">
              No at-risk employee data available
            </div>
          )}
        </Card>
      )}

      <RiskAnalysisModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        employee={selectedEmployee}
      />
    </div>
  )
}
