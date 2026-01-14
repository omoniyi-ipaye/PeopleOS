'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { GlassCard } from '@/components/ui/glass-card'
import { Badge } from '@/components/ui/badge'
import { BentoGrid, BentoGridItem } from '@/components/ui/bento-grid'
import { api } from '@/lib/api-client'
import { cn } from '@/lib/utils'
import {
  Activity,
  ShieldAlert,
  AlertTriangle,
  Users,
  Clock,
  BarChart3,
  Info,
  RefreshCw,
  TrendingDown,
  TrendingUp,
  Target,
  Layers,
  ChevronRight
} from 'lucide-react'
import { RetentionCurveChart } from '@/components/charts/retention-curve-chart'
import { RiskAnalysisModal } from '@/components/risk-analysis-modal'
import type { SurvivalAnalysisResult, AtRiskEmployee, CohortInsight } from '@/types/api'
import { ForecastTab } from '@/components/diagnostics/forecast-tab'

export default function RetentionForecastPage() {
  const [activeTab, setActiveTab] = useState<'overview' | 'risk-analysis' | 'watch-list' | 'trends'>('overview')
  const [selectedEmployee, setSelectedEmployee] = useState<AtRiskEmployee | null>(null)
  const [isModalOpen, setIsModalOpen] = useState(false)

  const { data: analysisData, isLoading, isError, error, refetch } = useQuery<SurvivalAnalysisResult>({
    queryKey: ['survival', 'analysis'],
    queryFn: () => api.survival.getAnalysis() as Promise<SurvivalAnalysisResult>,
  })

  // We limit the fetch to 100 for the premium view to keep it snappy
  const { data: atRiskData } = useQuery<AtRiskEmployee[]>({
    queryKey: ['survival', 'at-risk'],
    queryFn: () => api.survival.getAtRisk(100) as Promise<AtRiskEmployee[]>,
  })

  if (isLoading) {
    return (
      <div className="h-[calc(100vh-100px)] flex flex-col items-center justify-center">
        <div className="relative">
          <div className="absolute inset-0 bg-accent/20 blur-xl rounded-full animate-pulse" />
          <RefreshCw className="w-12 h-12 text-accent animate-spin relative z-10" />
        </div>
        <p className="mt-4 text-text-secondary animate-pulse font-medium">Forecasting Retention Trends...</p>
      </div>
    )
  }

  if (isError) {
    return (
      <div className="h-[calc(100vh-100px)] flex flex-col items-center justify-center text-center">
        <div className="p-6 rounded-full bg-danger/10 mb-4">
          <AlertTriangle className="w-12 h-12 text-danger" />
        </div>
        <h2 className="text-2xl font-bold text-text-primary dark:text-white">Analysis Failed</h2>
        <p className="text-text-secondary mt-2 max-w-md">
          {error instanceof Error ? error.message : 'Unable to generate retention model.'}
        </p>
        <button
          onClick={() => refetch()}
          className="mt-6 px-6 py-2 bg-text-primary dark:bg-white text-white dark:text-black rounded-xl font-bold hover:opacity-90 transition-opacity"
        >
          Retry Analysis
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
    <div className="space-y-6 h-[calc(100vh-100px)] flex flex-col animate-in fade-in duration-700 slide-in-from-bottom-4">
      {/* Header */}
      <div className="flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between flex-shrink-0">
        <div>
          <h1 className="text-4xl font-display font-bold text-gradient bg-clip-text text-transparent bg-gradient-to-r from-gray-900 to-gray-600 dark:from-white dark:to-gray-400">
            Retention Forecast
          </h1>
          <p className="text-text-secondary dark:text-text-dark-secondary mt-2 text-lg font-light flex items-center gap-2">
            Predictive modeling of employee tenure and turnover risk
          </p>
        </div>

        {/* Tab Controls */}
        <div className="glass p-1.5 rounded-2xl flex gap-1 overflow-x-auto no-scrollbar">
          {[
            { id: 'overview', icon: Activity, label: 'Overview' },
            { id: 'risk-analysis', icon: Target, label: 'Risk Analysis' },
            { id: 'watch-list', icon: ShieldAlert, label: 'Watch List' },
            { id: 'trends', icon: TrendingUp, label: 'Trends' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={cn(
                "flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-all duration-300 whitespace-nowrap",
                activeTab === tab.id
                  ? "bg-white dark:bg-slate-800 shadow-lg text-text-primary dark:text-white scale-105"
                  : "text-text-secondary dark:text-slate-400 hover:text-text-primary dark:hover:text-white hover:bg-white/10"
              )}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Warnings Banner */}
      {warnings.length > 0 && (
        <div className="flex-shrink-0 bg-warning/5 border border-warning/20 rounded-xl p-3 flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-warning flex-shrink-0 mt-0.5" />
          <div className="text-sm">
            <span className="font-bold text-warning block mb-1">Data Quality Warnings</span>
            <ul className="text-text-secondary space-y-0.5">
              {warnings.slice(0, 2).map((w, i) => <li key={i}>• {w}</li>)}
            </ul>
          </div>
        </div>
      )}

      {/* Main Content Area */}
      <div className="flex-1 min-h-0 overflow-y-auto pb-8 pr-2 custom-scrollbar">
        {activeTab === 'overview' && (
          <div className="space-y-6 animate-in fade-in slide-in-from-left-4 duration-500">
            {/* Summary Bento Grid */}
            <BentoGrid>
              <BentoGridItem
                title="Median Tenure"
                description="Average time employees stay"
                header={
                  <div className="flex items-center gap-2">
                    <span className="text-4xl font-display font-bold text-text-primary dark:text-white">
                      {summary?.median_tenure ? summary.median_tenure.toFixed(1) : '> 15'}
                    </span>
                    <span className="text-sm text-text-secondary self-end mb-1">years</span>
                  </div>
                }
                className="md:col-span-1"
                icon={<Clock className="w-4 h-4 text-accent" />}
              />
              <BentoGridItem
                title="12-Month Retention"
                description="Probability of staying 1 year"
                header={
                  <div className="flex items-center gap-2">
                    <span className={cn("text-4xl font-display font-bold", (1 - (summary?.avg_12mo_risk || 0)) > 0.8 ? "text-success" : "text-warning")}>
                      {summary?.avg_12mo_risk !== undefined ? `${((1 - (summary.avg_12mo_risk || 0)) * 100).toFixed(0)}%` : 'N/A'}
                    </span>
                  </div>
                }
                className="md:col-span-1"
                icon={<Activity className="w-4 h-4 text-success" />}
              />
              <BentoGridItem
                title="High Risk Volume"
                description="Employees likely to leave soon"
                header={
                  <div className="flex items-center gap-2">
                    <span className="text-4xl font-display font-bold text-danger">
                      {summary?.high_risk_count ?? 0}
                    </span>
                    <span className="text-sm text-text-secondary self-end mb-1">people</span>
                  </div>
                }
                className="md:col-span-1"
                icon={<ShieldAlert className="w-4 h-4 text-danger" />}
              />
              <BentoGridItem
                title="Strategic Actions"
                description="Recommended interventions"
                header={
                  <div className="space-y-2 mt-2">
                    {recommendations.slice(0, 2).map((rec, i) => (
                      <div key={i} className="flex items-center gap-2 text-xs text-text-secondary bg-surface-secondary/50 p-1.5 rounded-lg border border-white/5">
                        <div className="w-1.5 h-1.5 rounded-full bg-accent flex-shrink-0" />
                        <span className="line-clamp-1">{rec}</span>
                      </div>
                    ))}
                  </div>
                }
                className="md:col-span-1"
                icon={<Target className="w-4 h-4 text-purple-500" />}
              />
            </BentoGrid>

            {/* Main Retention Curve */}
            <GlassCard className="p-6">
              <div className="mb-6">
                <h3 className="text-lg font-bold flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-accent" />
                  Retention Survival Curve
                </h3>
                <p className="text-sm text-text-secondary">Kaplan-Meier estimate showing the probability of employee retention over time.</p>
              </div>

              {kmData?.overall?.survival_function && kmData.overall.survival_function.length > 0 ? (
                <div className="pl-2">
                  <RetentionCurveChart data={kmData.overall.survival_function} />

                  {/* Key Milestones */}
                  <div className="grid grid-cols-5 gap-4 mt-8">
                    {[
                      { mo: 6, label: '6 Months' },
                      { mo: 12, label: '1 Year' },
                      { mo: 24, label: '2 Years' },
                      { mo: 36, label: '3 Years' },
                      { mo: 60, label: '5 Years' },
                    ].map((milestone) => {
                      // Find closest key in data (simplified mapping)
                      const key = `survival_at_${milestone.mo}mo` as keyof typeof kmData.overall
                      const prob = kmData.overall?.[key] as number | undefined
                      return (
                        <div key={milestone.mo} className="text-center p-3 rounded-xl bg-surface-secondary/30 border border-white/5">
                          <p className="text-[10px] text-text-muted uppercase tracking-wider mb-1">{milestone.label}</p>
                          <p className="text-lg font-bold">
                            {prob !== undefined ? `${(prob * 100).toFixed(0)}%` : '-'}
                          </p>
                        </div>
                      )
                    })}
                  </div>
                </div>
              ) : (
                <div className="h-64 flex flex-col items-center justify-center text-text-muted opacity-50">
                  <BarChart3 className="w-12 h-12 mb-4" />
                  <p>No survival data available</p>
                </div>
              )}
            </GlassCard>
          </div>
        )}

        {activeTab === 'risk-analysis' && (
          <div className="space-y-6 animate-in fade-in slide-in-from-right-4 duration-500">
            {/* Cohort Insights */}
            <div className="mb-4">
              <h3 className="text-lg font-bold">Identified Risk Segments</h3>
              <p className="text-text-secondary text-sm">We've used AI to cluster your workforce into distinct groups with similar behavior. This helps you understand <i>who</i> is at risk and <i>why</i>.</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {cohortInsights.map((cohort, i) => (
                <GlassCard key={i} className="flex flex-col">
                  <div className="p-4 border-b border-white/5 flex items-center justify-between bg-surface-secondary/20">
                    <div>
                      <h4 className="font-bold text-lg">{cohort.cohort_name || 'Employee Segment ' + (i + 1)}</h4>
                      <p className="text-xs text-text-secondary mt-0.5 uppercase tracking-wide opacity-80">{cohort.cohort_description}</p>
                    </div>
                    <Badge variant={
                      cohort.risk_level === 'High' ? 'danger' :
                        cohort.risk_level === 'Medium' ? 'warning' : 'success'
                    }>
                      {cohort.risk_level || 'Unknown'} Risk
                    </Badge>
                  </div>
                  <div className="p-6 flex-1 flex flex-col gap-6">
                    {/* Insight First - The "Story" */}
                    {cohort.insight ? (
                      <div className="p-4 bg-accent/5 rounded-xl border border-accent/10">
                        <div className="flex gap-2 items-start">
                          <Info className="w-4 h-4 text-accent shrink-0 mt-0.5" />
                          <p className="text-sm text-text-primary dark:text-white italic">"{cohort.insight}"</p>
                        </div>
                      </div>
                    ) : (
                      <div className="p-4 bg-surface-secondary/30 rounded-xl border border-white/5">
                        <p className="text-sm text-text-secondary">This group shares similar tenure and risk characteristics.</p>
                      </div>
                    )}

                    {/* Metrics Grid */}
                    <div className="grid grid-cols-3 gap-4 text-center mt-auto border-t border-white/5 pt-4">
                      <div>
                        <p className="text-[10px] text-text-muted uppercase tracking-wider mb-1">Group Size</p>
                        <p className="text-xl font-display font-bold">{cohort.cohort_size}</p>
                      </div>
                      <div>
                        <p className="text-[10px] text-text-muted uppercase tracking-wider mb-1">Avg Tenure</p>
                        <p className="text-xl font-display font-bold">
                          {(cohort.median_tenure ?? cohort.avg_tenure_years)
                            ? `${(cohort.median_tenure ?? cohort.avg_tenure_years)?.toFixed(1)} yr`
                            : '-'}
                        </p>
                      </div>
                      <div>
                        <p className="text-[10px] text-text-muted uppercase tracking-wider mb-1">Projected Retention</p>
                        <p className={cn("text-xl font-display font-bold",
                          (cohort.survival_probability_12mo || 0) > 0.85 ? "text-success" :
                            (cohort.survival_probability_12mo || 0) > 0.70 ? "text-warning" : "text-danger"
                        )}>
                          {cohort.survival_probability_12mo ? `${(cohort.survival_probability_12mo * 100).toFixed(0)}%` : '-'}
                        </p>
                      </div>
                    </div>
                  </div>
                </GlassCard>
              ))}
              {cohortInsights.length === 0 && (
                <div className="col-span-full text-center py-12 opacity-50">
                  <Layers className="w-16 h-16 mx-auto mb-4 text-text-muted" />
                  <p>No cohort insights available.</p>
                </div>
              )}
            </div>

            {/* Risk Drivers Table */}
            <GlassCard>
              <div className="p-6 border-b border-border/50 flex items-center justify-between">
                <div>
                  <h3 className="text-xl font-bold flex items-center gap-2">
                    <Target className="w-6 h-6 text-accent" />
                    Key Risk Drivers (Cox Model)
                  </h3>
                  <p className="text-sm text-text-secondary mt-1">
                    Factors mathematically linked to employee departure. Model Accuracy: <span className="text-accent font-bold">{((coxModel?.concordance || 0) * 100).toFixed(1)}%</span>
                  </p>
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="border-b border-border/50 text-xs text-text-muted uppercase tracking-wider bg-surface-secondary/20">
                      <th className="p-4 rounded-tl-xl">Factor</th>
                      <th className="p-4 text-right">Impact Multiplier</th>
                      <th className="p-4 text-right">Probability Shift</th>
                      <th className="p-4 text-center">Confidence</th>
                      <th className="p-4 rounded-tr-xl text-center">Action</th>
                    </tr>
                  </thead>
                  <tbody className="text-sm">
                    {coxCoefficients.map((coef, i) => (
                      <tr key={i} className="border-b border-border/50 hover:bg-white/5 transition-colors group">
                        <td className="p-4 font-bold capitalize text-text-primary dark:text-white">
                          {coef.covariate?.replace(/_/g, ' ')}
                        </td>
                        <td className="p-4 text-right font-mono text-base">
                          <span className={cn(coef.hazard_ratio > 1 ? "text-danger" : "text-success")}>
                            {coef.hazard_ratio?.toFixed(2)}x
                          </span>
                        </td>
                        <td className="p-4 text-right">
                          <div className="flex items-center justify-end gap-2">
                            {coef.hazard_ratio > 1 ? <TrendingUp className="w-4 h-4 text-danger" /> : <TrendingDown className="w-4 h-4 text-success" />}
                            <span className={coef.hazard_ratio > 1 ? "text-danger" : "text-success"}>
                              {coef.hazard_ratio > 1
                                ? `${((coef.hazard_ratio - 1) * 100).toFixed(0)}% Higher Risk`
                                : `${((1 - coef.hazard_ratio) * 100).toFixed(0)}% Lower Risk`
                              }
                            </span>
                          </div>
                        </td>
                        <td className="p-4 text-center">
                          <Badge variant="outline" className={cn(coef.p_value < 0.05 ? "border-success/50 text-success bg-success/5" : "border-warning/50 text-warning bg-warning/5")}>
                            {coef.p_value < 0.05 ? 'High' : 'Low'}
                          </Badge>
                        </td>
                        <td className="p-4 text-center">
                          {coef.hazard_ratio > 1 && coef.p_value < 0.1 && (
                            <button className="text-xs font-bold text-accent hover:underline">Mitigate</button>
                          )}
                        </td>
                      </tr>
                    ))}
                    {coxCoefficients.length === 0 && (
                      <tr>
                        <td colSpan={5} className="p-8 text-center text-text-muted">No significant risk factors found in the model.</td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
              <div className="p-4 bg-accent/5 m-4 rounded-xl border border-accent/10">
                <p className="text-xs text-text-secondary leading-relaxed flex gap-2">
                  <Info className="w-4 h-4 text-accent shrink-0" />
                  A multiplier greater than 1.0 (Red) indicates this factor increases the risk of leaving. A multiplier less than 1.0 (Green) indicates it helps retain employees.
                </p>
              </div>
            </GlassCard>
          </div>
        )}

        {activeTab === 'watch-list' && (
          <GlassCard className="animate-in fade-in slide-in-from-left-4 duration-500">
            <div className="p-6 border-b border-border/50">
              <h3 className="text-xl font-bold flex items-center gap-2">
                <ShieldAlert className="w-6 h-6 text-danger" />
                High Risk Watch List
              </h3>
              <p className="text-sm text-text-secondary mt-1">Employees with the highest probability of departure in the next 12 months.</p>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left">
                <thead>
                  <tr className="text-xs text-text-muted uppercase tracking-wider bg-surface-secondary/20 border-b border-border/50">
                    <th className="p-4">Employee</th>
                    <th className="p-4">Department</th>
                    <th className="p-4 text-right">Tenure</th>
                    <th className="p-4 text-center">Departure Probability (12mo)</th>
                    <th className="p-4 text-center">Risk Level</th>
                    <th className="p-4"></th>
                  </tr>
                </thead>
                <tbody className="text-sm">
                  {atRiskEmployees.map((emp, i) => (
                    <tr key={i}
                      onClick={() => {
                        setSelectedEmployee(emp)
                        setIsModalOpen(true)
                      }}
                      className="border-b border-border/50 hover:bg-white/5 cursor-pointer transition-colors group"
                    >
                      <td className="p-4 font-medium text-text-primary dark:text-white group-hover:text-accent transition-colors">
                        {emp.EmployeeID}
                      </td>
                      <td className="p-4 text-text-secondary">{emp.Dept}</td>
                      <td className="p-4 text-right font-mono text-text-secondary">{emp.current_tenure_years?.toFixed(1)} yrs</td>
                      <td className="p-4 flex justify-center">
                        <div className="w-32 bg-slate-200 dark:bg-slate-800 rounded-full h-2 mt-2 relative overflow-hidden">
                          <div
                            className={cn("absolute left-0 top-0 bottom-0 transition-all duration-500", (emp.attrition_risk_12mo || 0) > 0.5 ? "bg-danger" : "bg-warning")}
                            style={{ width: `${(emp.attrition_risk_12mo || 0) * 100}%` }}
                          />
                        </div>
                        <span className="ml-3 font-bold text-xs w-8">{Math.round((emp.attrition_risk_12mo || 0) * 100)}%</span>
                      </td>
                      <td className="p-4 text-center">
                        <Badge variant={emp.risk_category === 'High' ? 'danger' : emp.risk_category === 'Medium' ? 'warning' : 'success'}>
                          {emp.risk_category}
                        </Badge>
                      </td>
                      <td className="p-4 text-right">
                        <ChevronRight className="w-4 h-4 text-text-muted group-hover:text-accent transition-colors" />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </GlassCard>
        )}

        {activeTab === 'trends' && (
          <ForecastTab />
        )}
      </div>

      <RiskAnalysisModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        employee={selectedEmployee}
      />
    </div>
  )
}
