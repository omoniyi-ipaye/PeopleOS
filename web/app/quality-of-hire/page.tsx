'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Card } from '../../components/ui/card'
import { Badge } from '../../components/ui/badge'
import { api } from '../../lib/api-client'
import { cn } from '../../lib/utils'
import {
  UserPlus,
  TrendingUp,
  Target,
  AlertTriangle,
  Award,
  BarChart3,
  Info,
  ArrowUpRight,
  ArrowDownRight,
  RefreshCw
} from 'lucide-react'
import { KPICard } from '../../components/dashboard/kpi-card'
import {
  ExplanationBox,
  CorrelationGuide,
  MetricExplainer,
} from '../../components/ui/metric-explainer'
import { RiskAnalysisModal } from '../../components/risk-analysis-modal'
import { TabGroup } from '../../components/ui/tab-group'
import type {
  QualityOfHireAnalysisResult,
  SourceEffectiveness,
  NewHireRisk,
  PreHireCorrelation,
} from '@/types/api'

type QoHTab = 'overview' | 'sources' | 'predictors' | 'new-hires'

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'sources', label: 'Hiring Sources' },
  { id: 'predictors', label: 'Success Drivers' },
  { id: 'new-hires', label: 'New Hire Watch' },
] as const

export default function QualityOfHirePage() {
  const [activeTab, setActiveTab] = useState<QoHTab>('overview')
  const [selectedEmployee, setSelectedEmployee] = useState<NewHireRisk | null>(null)
  const [isModalOpen, setIsModalOpen] = useState(false)

  const { data: analysisData, isLoading, isError, error, refetch } = useQuery<QualityOfHireAnalysisResult>({
    queryKey: ['quality-of-hire', 'analysis'],
    queryFn: () => api.qualityOfHire.getAnalysis() as Promise<QualityOfHireAnalysisResult>,
  })

  const { data: newHireRisks } = useQuery<NewHireRisk[]>({
    queryKey: ['quality-of-hire', 'new-hire-risks'],
    queryFn: () => api.qualityOfHire.getNewHireRisks(6) as Promise<NewHireRisk[]>,
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-pulse-subtle text-text-secondary dark:text-text-dark-secondary">
          Loading quality of hire analysis...
        </div>
      </div>
    )
  }

  if (isError) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4 text-center">
        <AlertTriangle className="w-12 h-12 text-danger" />
        <h2 className="text-xl font-semibold text-text-primary dark:text-text-dark-primary">
          Failed to Load Quality of Hire Data
        </h2>
        <p className="text-text-secondary dark:text-text-dark-secondary max-w-md">
          {error instanceof Error ? error.message : 'Unable to load hiring quality analysis. Please try again.'}
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

  const sourceEffectiveness: SourceEffectiveness[] = analysisData?.source_effectiveness || []
  const correlations = analysisData?.correlations
  const insights = analysisData?.insights
  const summary = analysisData?.summary
  const warnings: string[] = analysisData?.warnings || []
  const recommendations: string[] = analysisData?.recommendations || []
  const newHires: NewHireRisk[] = newHireRisks || []

  return (
    <div className="space-y-6">
      {/* Page Title */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-text-primary dark:text-text-dark-primary">
            Quality of Hire
          </h1>
          <p className="text-text-secondary dark:text-text-dark-secondary mt-1">
            Pre-hire to post-hire correlation analysis and hiring source effectiveness
          </p>
        </div>

        {/* Tab Switcher */}
        <TabGroup<QoHTab>
          tabs={TABS}
          activeTab={activeTab}
          onTabChange={setActiveTab}
        />
      </div>

      <ExplanationBox title={`About: ${activeTab === 'overview' ? 'Quality Overview' : activeTab === 'sources' ? 'Hiring Sources' : activeTab === 'predictors' ? 'Success Drivers' : 'New Hire Watch'}`}>
        {activeTab === 'overview' && "Evaluate how effectively your hiring process brings in high-performing, long-term talent."}
        {activeTab === 'sources' && "Compare different recruitment channels to see which ones deliver the best results."}
        {activeTab === 'predictors' && "Identify which pre-hire signals (like interview scores) actually predict future success."}
        {activeTab === 'new-hires' && "Monitor recent hires who might need extra support to ensure a successful onboarding."}
      </ExplanationBox>

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

      {activeTab === 'overview' && (
        <>
          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <KPICard
              title="Total Employees"
              value={summary?.total_employees ?? 0}
              icon={UserPlus}
              insight="The total workforce analyzed for hiring quality"
            />
            <KPICard
              title="Sources Analyzed"
              value={summary?.sources_analyzed ?? 0}
              icon={Target}
              insight="Recruitment channels where we mapped hiring success"
            />
            <KPICard
              title="Success Drivers"
              value={summary?.prehire_signals_count ?? 0}
              icon={BarChart3}
              insight="Pre-hire factors (like interview scores) found predictive of success"
            />
            <KPICard
              title="New Hire Watch"
              value={summary?.new_hires_at_risk ?? 0}
              icon={AlertTriangle}
              variant="danger"
              insight="Recent hires showing signs of low engagement or risk"
            />
          </div>

          {/* Best Source and Top Predictor */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card className="p-6">
              <div className="flex items-center gap-3 mb-4">
                <Award className="w-6 h-6 text-success" />
                <h3 className="text-lg font-bold text-text-primary dark:text-text-dark-primary">
                  Best Hiring Source
                </h3>
              </div>
              {summary?.best_source ? (
                <div className="text-center py-4">
                  <p className="text-3xl font-bold text-success mb-2">
                    {summary.best_source}
                  </p>
                  <p className="text-sm text-text-secondary">
                    Highest quality score based on performance and retention
                  </p>
                </div>
              ) : (
                <p className="text-text-muted text-center py-4">
                  No source data available
                </p>
              )}
            </Card>

            <Card className="p-6">
              <div className="flex items-center gap-3 mb-4">
                <TrendingUp className="w-6 h-6 text-accent" />
                <h3 className="text-lg font-bold text-text-primary dark:text-text-dark-primary">
                  Top Predictor
                </h3>
              </div>
              {summary?.top_predictor ? (
                <div className="text-center py-4">
                  <p className="text-3xl font-bold text-accent mb-2">
                    {summary.top_predictor}
                  </p>
                  <p className="text-sm text-text-secondary">
                    Strongest pre-hire signal for post-hire performance
                  </p>
                </div>
              ) : (
                <p className="text-text-muted text-center py-4">
                  No predictor data available
                </p>
              )}
            </Card>
          </div>

          {/* Recommendations */}
          {recommendations.length > 0 && (
            <Card title="Hiring Recommendations" subtitle="Strategic actions to improve quality of hire">
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

      {activeTab === 'sources' && (
        <Card title="Hiring Source Effectiveness" subtitle="Performance comparison across recruitment channels">
          {sourceEffectiveness.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border dark:border-border-dark">
                    <th className="text-left py-3 px-4 text-xs font-semibold text-text-muted uppercase">
                      Source
                    </th>
                    <th className="text-right py-3 px-4 text-xs font-semibold text-text-muted uppercase">
                      Hires
                    </th>
                    <th className="text-right py-3 px-4 text-xs font-semibold text-text-muted uppercase">
                      Avg Performance
                    </th>
                    <th className="text-right py-3 px-4 text-xs font-semibold text-text-muted uppercase">
                      Retention Rate
                    </th>
                    <th className="text-right py-3 px-4 text-xs font-semibold text-text-muted uppercase">
                      High Performers
                    </th>
                    <th className="text-right py-3 px-4 text-xs font-semibold text-text-muted uppercase">
                      Quality Score
                    </th>
                    <th className="text-center py-3 px-4 text-xs font-semibold text-text-muted uppercase">
                      Grade
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {sourceEffectiveness.map((source, i) => (
                    <tr
                      key={i}
                      className="border-b border-border/50 dark:border-border-dark/50 hover:bg-surface-hover dark:hover:bg-surface-dark-hover"
                    >
                      <td className="py-3 px-4 font-medium text-text-primary dark:text-text-dark-primary">
                        {source.HireSource}
                      </td>
                      <td className="py-3 px-4 text-right font-mono">
                        {source.hire_count}
                        <span className="text-xs text-text-muted ml-1">
                          ({source.pct_of_total?.toFixed(0)}%)
                        </span>
                      </td>
                      <td className="py-3 px-4 text-right font-mono">
                        {source.avg_performance?.toFixed(2) || '-'}
                      </td>
                      <td className="py-3 px-4 text-right font-mono">
                        <span className={cn(
                          (source.retention_rate_pct ?? 0) > 85 ? 'text-success' :
                            (source.retention_rate_pct ?? 0) > 70 ? 'text-warning' : 'text-danger'
                        )}>
                          {source.retention_rate_pct?.toFixed(0)}%
                        </span>
                      </td>
                      <td className="py-3 px-4 text-right font-mono">
                        {source.high_performer_rate?.toFixed(0)}%
                      </td>
                      <td className="py-3 px-4 text-right font-mono font-bold">
                        {source.quality_score?.toFixed(1)}
                      </td>
                      <td className="py-3 px-4 text-center">
                        <Badge variant={
                          source.grade === 'A' ? 'success' :
                            source.grade === 'B' ? 'info' :
                              source.grade === 'C' ? 'warning' : 'danger'
                        }>
                          {source.grade}
                        </Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>

              <div className="mt-4 p-4 bg-surface-secondary dark:bg-background-dark rounded-lg">
                <p className="text-xs text-text-muted mb-2">Source Recommendations:</p>
                <div className="space-y-2">
                  {sourceEffectiveness.slice(0, 3).map((source, i) => (
                    <div key={i} className="flex items-start gap-2 text-xs text-text-secondary">
                      <span className="font-bold">{source.HireSource}:</span>
                      <span>{source.recommendation}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="py-12 text-center text-text-secondary">
              No source effectiveness data available. Ensure data has HireSource column.
            </div>
          )}
        </Card>
      )}

      {activeTab === 'predictors' && (
        <Card title="Pre-hire Signal Correlations" subtitle="Which interview dimensions predict actual performance?">
          {correlations?.available && (correlations.correlations?.length ?? 0) > 0 ? (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                <div className="p-4 bg-success/10 border border-success/20 rounded-lg">
                  <p className="text-xs text-success uppercase tracking-wider mb-2 font-semibold">
                    Best Predictors
                  </p>
                  <div className="space-y-2">
                    {correlations.best_predictors?.slice(0, 3).map((pred, i) => (
                      <div key={i} className="flex items-center justify-between">
                        <span className="text-sm font-medium">{pred.display_name}</span>
                        <div className="flex items-center gap-2">
                          <ArrowUpRight className="w-3 h-3 text-success" />
                          <span className="font-mono text-sm">{pred.correlation?.toFixed(2)}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="p-4 bg-danger/10 border border-danger/20 rounded-lg">
                  <p className="text-xs text-danger uppercase tracking-wider mb-2 font-semibold">
                    Non-Predictive Signals
                  </p>
                  <div className="space-y-2">
                    {correlations.non_predictors?.slice(0, 3).map((pred, i) => (
                      <div key={i} className="flex items-center justify-between">
                        <span className="text-sm font-medium">{pred.display_name}</span>
                        <div className="flex items-center gap-2">
                          <ArrowDownRight className="w-3 h-3 text-text-muted" />
                          <span className="font-mono text-sm text-text-muted">
                            {pred.correlation?.toFixed(2)}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-border dark:border-border-dark">
                      <th className="text-left py-3 px-4 text-xs font-semibold text-text-muted uppercase">
                        Pre-hire Signal
                      </th>
                      <th className="text-right py-3 px-4 text-xs font-semibold text-text-muted uppercase">
                        Correlation
                      </th>
                      <th className="text-right py-3 px-4 text-xs font-semibold text-text-muted uppercase">
                        p-value
                      </th>
                      <th className="text-center py-3 px-4 text-xs font-semibold text-text-muted uppercase">
                        Strength
                      </th>
                      <th className="text-left py-3 px-4 text-xs font-semibold text-text-muted uppercase">
                        Interpretation
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {correlations.correlations?.map((corr, i) => (
                      <tr
                        key={i}
                        className="border-b border-border/50 dark:border-border-dark/50 hover:bg-surface-hover dark:hover:bg-surface-dark-hover"
                      >
                        <td className="py-3 px-4 font-medium text-text-primary dark:text-text-dark-primary">
                          {corr.display_name}
                        </td>
                        <td className="py-3 px-4 text-right font-mono">
                          <span className={cn(
                            corr.correlation > 0 ? 'text-success' : 'text-danger'
                          )}>
                            {corr.correlation?.toFixed(3)}
                          </span>
                        </td>
                        <td className="py-3 px-4 text-right font-mono text-sm">
                          {corr.p_value < 0.001 ? '<0.001' : corr.p_value?.toFixed(3)}
                        </td>
                        <td className="py-3 px-4 text-center">
                          <Badge variant={
                            corr.strength === 'Strong' ? 'success' :
                              corr.strength === 'Moderate' ? 'info' :
                                corr.strength === 'Weak' ? 'warning' : 'outline'
                          }>
                            {corr.strength}
                          </Badge>
                        </td>
                        <td className="py-3 px-4 text-sm text-text-secondary max-w-xs truncate">
                          {corr.interpretation}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {(correlations.recommendations?.length ?? 0) > 0 && (
                <div className="p-4 bg-accent/5 border border-accent/20 rounded-lg">
                  <p className="text-xs text-accent font-semibold mb-2 uppercase tracking-wider">
                    Recommendations
                  </p>
                  <ul className="text-sm text-text-secondary space-y-1">
                    {correlations.recommendations?.map((rec, i) => (
                      <li key={i}>- {rec}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ) : (
            <div className="py-12 text-center text-text-secondary">
              No correlation data available. Ensure data has interview score columns.
            </div>
          )}
        </Card>
      )}

      {activeTab === 'new-hires' && (
        <Card title="New Hire Risk Assessment" subtitle="Recent hires who may need additional support">
          {newHires.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border dark:border-border-dark">
                    <th className="text-left py-3 px-4 text-xs font-semibold text-text-muted uppercase">
                      Employee
                    </th>
                    <th className="text-left py-3 px-4 text-xs font-semibold text-text-muted uppercase">
                      Hire Date
                    </th>
                    <th className="text-left py-3 px-4 text-xs font-semibold text-text-muted uppercase">
                      Source
                    </th>
                    <th className="text-left py-3 px-4 text-xs font-semibold text-text-muted uppercase">
                      Department
                    </th>
                    <th className="text-right py-3 px-4 text-xs font-semibold text-text-muted uppercase">
                      Risk Score
                    </th>
                    <th className="text-center py-3 px-4 text-xs font-semibold text-text-muted uppercase">
                      Category
                    </th>
                    <th className="text-left py-3 px-4 text-xs font-semibold text-text-muted uppercase">
                      Risk Factors
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {newHires.map((hire, i) => (
                    <tr
                      key={i}
                      onClick={() => {
                        setSelectedEmployee(hire)
                        setIsModalOpen(true)
                      }}
                      className="border-b border-border/50 dark:border-border-dark/50 hover:bg-surface-hover dark:hover:bg-surface-dark-hover cursor-pointer"
                    >
                      <td className="py-3 px-4 font-medium text-text-primary dark:text-text-dark-primary">
                        {hire.EmployeeID}
                      </td>
                      <td className="py-3 px-4 text-text-secondary text-sm">
                        {hire.HireDate}
                      </td>
                      <td className="py-3 px-4 text-text-secondary">
                        {hire.HireSource || '-'}
                      </td>
                      <td className="py-3 px-4 text-text-secondary">
                        {hire.Dept || '-'}
                      </td>
                      <td className="py-3 px-4 text-right font-mono">
                        <span className={cn(
                          hire.risk_score > 0.7 ? 'text-danger' :
                            hire.risk_score > 0.4 ? 'text-warning' : 'text-success'
                        )}>
                          {(hire.risk_score * 100).toFixed(0)}%
                        </span>
                      </td>
                      <td className="py-3 px-4 text-center">
                        <Badge variant={
                          hire.risk_category === 'High' ? 'danger' :
                            hire.risk_category === 'Medium' ? 'warning' : 'success'
                        }>
                          {hire.risk_category}
                        </Badge>
                      </td>
                      <td className="py-3 px-4 text-sm text-text-secondary max-w-xs truncate">
                        {hire.risk_factors_text}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="py-12 text-center text-text-secondary">
              No new hire risk data available
            </div>
          )}
        </Card>
      )}

      {/* Risk Analysis Modal */}
      <RiskAnalysisModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        employee={selectedEmployee}
      />
    </div>
  )
}
