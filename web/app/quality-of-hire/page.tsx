'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { GlassCard } from '@/components/ui/glass-card'
import { Badge } from '@/components/ui/badge'
import { BentoGrid, BentoGridItem } from '@/components/ui/bento-grid'
import { api } from '@/lib/api-client'
import { cn } from '@/lib/utils'
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
  RefreshCw,
  Search,
  Sparkles,
  ChevronRight
} from 'lucide-react'
import { RiskAnalysisModal } from '@/components/risk-analysis-modal'
import type {
  QualityOfHireAnalysisResult,
  SourceEffectiveness,
  NewHireRisk,
} from '@/types/api'

type QoHTab = 'overview' | 'sources' | 'predictors' | 'new-hires'

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
      <div className="h-[calc(100vh-100px)] flex flex-col items-center justify-center">
        <div className="relative">
          <div className="absolute inset-0 bg-accent/20 blur-xl rounded-full animate-pulse" />
          <RefreshCw className="w-12 h-12 text-accent animate-spin relative z-10" />
        </div>
        <p className="mt-4 text-text-secondary animate-pulse font-medium">Analyzing Quality of Hire...</p>
      </div>
    )
  }

  if (isError) {
    return (
      <div className="h-[calc(100vh-100px)] flex flex-col items-center justify-center text-center">
        <div className="p-6 rounded-full bg-danger/10 mb-4">
          <AlertTriangle className="w-12 h-12 text-danger" />
        </div>
        <h2 className="text-2xl font-bold text-text-primary dark:text-text-dark-primary">Analysis Failed</h2>
        <p className="text-text-secondary dark:text-text-dark-secondary mt-2 max-w-md">
          {error instanceof Error ? error.message : 'Unable to load hiring quality analysis.'}
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

  const sourceEffectiveness: SourceEffectiveness[] = analysisData?.source_effectiveness || []
  const correlations = analysisData?.correlations
  const summary = analysisData?.summary
  const warnings: string[] = analysisData?.warnings || []
  const recommendations: string[] = analysisData?.recommendations || []
  const newHires: NewHireRisk[] = newHireRisks || []

  return (
    <div className="space-y-6 h-[calc(100vh-100px)] flex flex-col animate-in fade-in duration-700 slide-in-from-bottom-4">
      {/* Header */}
      <div className="flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between flex-shrink-0">
        <div>
          <h1 className="text-4xl font-display font-bold text-gradient bg-clip-text text-transparent bg-gradient-to-r from-gray-900 to-gray-600 dark:from-white dark:to-gray-400">
            Quality of Hire
          </h1>
          <p className="text-text-secondary dark:text-text-dark-secondary mt-2 text-lg font-light flex items-center gap-2">
            Optimize recruitment with data-driven insights
          </p>
        </div>

        {/* Tab Controls */}
        <div className="glass p-1.5 rounded-2xl flex gap-1 overflow-x-auto no-scrollbar">
          {[
            { id: 'overview', icon: Sparkles, label: 'Overview' },
            { id: 'sources', icon: Target, label: 'Sources' },
            { id: 'predictors', icon: BarChart3, label: 'Success Drivers' },
            { id: 'new-hires', icon: UserPlus, label: 'New Hires' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as QoHTab)}
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

      {/* Warnings */}
      {warnings.length > 0 && (
        <div className="flex-shrink-0 bg-warning/5 border border-warning/20 rounded-xl p-3 flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-warning flex-shrink-0 mt-0.5" />
          <div className="text-sm">
            <span className="font-bold text-warning block mb-1">Data Quality Warnings</span>
            <ul className="text-text-secondary space-y-0.5">
              {warnings.map((w, i) => <li key={i}>• {w}</li>)}
            </ul>
          </div>
        </div>
      )}

      {/* Main Content Area */}
      <div className="flex-1 min-h-0 overflow-y-auto pb-8 pr-2 custom-scrollbar">
        {activeTab === 'overview' && (
          <div className="space-y-6 animate-in fade-in slide-in-from-left-4 duration-500">
            <BentoGrid>
              <BentoGridItem
                title="Total Employees"
                description="Analyzed for quality"
                header={
                  <div className="flex items-center gap-2">
                    <span className="text-4xl font-display font-bold text-text-primary dark:text-white">
                      {summary?.total_employees ?? 0}
                    </span>
                    <span className="text-sm text-text-secondary self-end mb-1">people</span>
                  </div>
                }
                className="md:col-span-1"
                icon={<UserPlus className="w-4 h-4 text-accent" />}
              />
              <BentoGridItem
                title="Hiring Sources"
                description="Channels analyzed"
                header={
                  <div className="flex items-center gap-2">
                    <span className="text-4xl font-display font-bold text-text-primary dark:text-white">
                      {summary?.sources_analyzed ?? 0}
                    </span>
                  </div>
                }
                className="md:col-span-1"
                icon={<Target className="w-4 h-4 text-accent" />}
              />
              <BentoGridItem
                title="Success Drivers"
                description="Predictive factors"
                header={
                  <div className="flex items-center gap-2">
                    <span className="text-4xl font-display font-bold text-success">
                      {summary?.prehire_signals_count ?? 0}
                    </span>
                    <span className="text-sm text-text-secondary self-end mb-1">signals</span>
                  </div>
                }
                className="md:col-span-1"
                icon={<BarChart3 className="w-4 h-4 text-success" />}
              />
              <BentoGridItem
                title="New Hire Risk"
                description="At-risk new employees"
                header={
                  <div className="flex items-center gap-2">
                    <span className="text-4xl font-display font-bold text-danger">
                      {summary?.new_hires_at_risk ?? 0}
                    </span>
                  </div>
                }
                className="md:col-span-1"
                icon={<AlertTriangle className="w-4 h-4 text-danger" />}
              />

              {/* Best Source & Top Predictor */}
              <BentoGridItem
                title="Best Hiring Source"
                description="Highest quality candidates"
                header={
                  summary?.best_source ? (
                    <div>
                      <p className="text-2xl font-bold text-success mb-1">{summary.best_source}</p>
                      <p className="text-xs text-text-secondary">Maximizes retention & performance</p>
                    </div>
                  ) : (
                    <p className="text-text-muted italic">No data available</p>
                  )
                }
                className="md:col-span-2"
                icon={<Award className="w-4 h-4 text-success" />}
              />
              <BentoGridItem
                title="Top Predictor"
                description="Strongest pre-hire signal"
                header={
                  summary?.top_predictor ? (
                    <div>
                      <p className="text-2xl font-bold text-purple-500 mb-1">{summary.top_predictor}</p>
                      <p className="text-xs text-text-secondary">Best indicator of future success</p>
                    </div>
                  ) : (
                    <p className="text-text-muted italic">No data available</p>
                  )
                }
                className="md:col-span-2"
                icon={<TrendingUp className="w-4 h-4 text-purple-500" />}
              />
            </BentoGrid>

            {/* Recommendations */}
            {recommendations.length > 0 && (
              <GlassCard className="p-6">
                <h3 className="font-bold text-lg mb-4 flex items-center gap-2">
                  <Info className="w-5 h-5 text-accent" />
                  Strategic Recommendations
                </h3>
                <div className="grid gap-3">
                  {recommendations.map((rec, i) => (
                    <div key={i} className="flex items-start gap-3 p-3 bg-accent/5 rounded-xl border border-accent/10">
                      <div className="w-1.5 h-1.5 rounded-full bg-accent flex-shrink-0 mt-2" />
                      <p className="text-sm text-text-secondary">{rec}</p>
                    </div>
                  ))}
                </div>
              </GlassCard>
            )}
          </div>
        )}

        {activeTab === 'sources' && (
          <GlassCard className="animate-in fade-in slide-in-from-right-4 duration-500">
            <div className="p-6 border-b border-border/50">
              <h3 className="text-xl font-bold flex items-center gap-2">
                <Target className="w-6 h-6 text-accent" />
                Hiring Source Effectiveness
              </h3>
              <p className="text-sm text-text-secondary mt-1">Compare performance and retention across different recruitment channels.</p>
            </div>

            {sourceEffectiveness.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-left">
                  <thead>
                    <tr className="text-xs text-text-muted uppercase tracking-wider bg-surface-secondary/20 border-b border-border/50">
                      <th className="p-4 rounded-tl-xl">Source</th>
                      <th className="p-4 text-right">Hires</th>
                      <th className="p-4 text-right">Avg Rating</th>
                      <th className="p-4 text-right">Retention</th>
                      <th className="p-4 text-right">Quality Score</th>
                      <th className="p-4 text-center rounded-tr-xl">Grade</th>
                    </tr>
                  </thead>
                  <tbody className="text-sm">
                    {sourceEffectiveness.map((source, i) => (
                      <tr key={i} className="border-b border-border/50 hover:bg-white/5 transition-colors">
                        <td className="p-4 font-bold text-text-primary dark:text-white">
                          {source.HireSource}
                        </td>
                        <td className="p-4 text-right font-mono">
                          {source.hire_count}
                          <span className="text-xs text-text-muted ml-1 opacity-70">
                            ({source.pct_of_total?.toFixed(0)}%)
                          </span>
                        </td>
                        <td className="p-4 text-right font-mono">
                          {source.avg_performance?.toFixed(2) || '-'}
                        </td>
                        <td className="p-4 text-right font-mono">
                          <span className={cn(
                            (source.retention_rate_pct ?? 0) > 85 ? 'text-success' :
                              (source.retention_rate_pct ?? 0) > 70 ? 'text-warning' : 'text-danger'
                          )}>
                            {source.retention_rate_pct?.toFixed(0)}%
                          </span>
                        </td>
                        <td className="p-4 text-right font-mono font-bold">
                          {source.quality_score?.toFixed(1)}
                        </td>
                        <td className="p-4 text-center">
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
              </div>
            ) : (
              <div className="p-12 text-center text-text-secondary flex flex-col items-center">
                <Search className="w-12 h-12 mb-4 opacity-20" />
                <p>No source data available.</p>
              </div>
            )}

            <div className="p-4 bg-surface-secondary/30 m-6 rounded-xl border border-white/5">
              <p className="text-xs font-bold uppercase tracking-wider mb-3 text-text-muted">Recommendations</p>
              <div className="grid gap-2">
                {sourceEffectiveness.slice(0, 3).map((source, i) => (
                  <div key={i} className="flex items-start gap-2 text-sm text-text-secondary">
                    <span className="font-bold whitespace-nowrap text-text-primary dark:text-white">{source.HireSource}:</span>
                    <span>{source.recommendation}</span>
                  </div>
                ))}
              </div>
            </div>
          </GlassCard>
        )}

        {activeTab === 'predictors' && (
          <GlassCard className="animate-in fade-in slide-in-from-right-4 duration-500">
            <div className="p-6 border-b border-border/50">
              <h3 className="text-xl font-bold flex items-center gap-2">
                <BarChart3 className="w-6 h-6 text-accent" />
                Pre-Hire Signal Analysis
              </h3>
              <p className="text-sm text-text-secondary mt-1">Determine which interview scores and signals actually predict performance.</p>
            </div>

            {correlations?.available && (correlations.correlations?.length ?? 0) > 0 ? (
              <div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 p-6 border-b border-white/5">
                  <div className="p-4 bg-success/5 border border-success/10 rounded-xl">
                    <p className="text-xs font-bold text-success uppercase tracking-wider mb-3">Best Predictors</p>
                    <div className="space-y-3">
                      {correlations.best_predictors?.slice(0, 3).map((pred, i) => (
                        <div key={i} className="flex items-center justify-between">
                          <span className="font-medium text-sm">{pred.display_name}</span>
                          <Badge variant="success" className="font-mono">{pred.correlation?.toFixed(2)}</Badge>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="p-4 bg-surface-secondary/30 border border-white/5 rounded-xl opacity-80">
                    <p className="text-xs font-bold text-text-muted uppercase tracking-wider mb-3">Non-Predictive</p>
                    <div className="space-y-3">
                      {correlations.non_predictors?.slice(0, 3).map((pred, i) => (
                        <div key={i} className="flex items-center justify-between">
                          <span className="text-sm">{pred.display_name}</span>
                          <span className="font-mono text-xs text-text-muted">{pred.correlation?.toFixed(2)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left">
                    <thead>
                      <tr className="text-xs text-text-muted uppercase tracking-wider bg-surface-secondary/20 border-b border-border/50">
                        <th className="p-4">Signal</th>
                        <th className="p-4 text-right">Correlation</th>
                        <th className="p-4 text-right">P-Value</th>
                        <th className="p-4 text-center">Strength</th>
                        <th className="p-4">Interpretation</th>
                      </tr>
                    </thead>
                    <tbody className="text-sm">
                      {correlations.correlations?.map((corr, i) => (
                        <tr key={i} className="border-b border-border/50 hover:bg-white/5 transition-colors">
                          <td className="p-4 font-bold text-text-primary dark:text-white">{corr.display_name}</td>
                          <td className="p-4 text-right font-mono">
                            <span className={cn(corr.correlation > 0 ? "text-success" : "text-danger")}>
                              {corr.correlation?.toFixed(3)}
                            </span>
                          </td>
                          <td className="p-4 text-right font-mono text-text-secondary">
                            {corr.p_value < 0.001 ? '<0.001' : corr.p_value?.toFixed(3)}
                          </td>
                          <td className="p-4 text-center">
                            <Badge variant={
                              corr.strength === 'Strong' ? 'success' :
                                corr.strength === 'Moderate' ? 'info' :
                                  corr.strength === 'Weak' ? 'warning' : 'outline'
                            }>
                              {corr.strength}
                            </Badge>
                          </td>
                          <td className="p-4 text-text-secondary text-sm italic">
                            {corr.interpretation}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : (
              <div className="p-12 text-center text-text-secondary">No correlation data available.</div>
            )}
          </GlassCard>
        )}

        {activeTab === 'new-hires' && (
          <GlassCard className="animate-in fade-in slide-in-from-right-4 duration-500">
            <div className="p-6 border-b border-border/50">
              <h3 className="text-xl font-bold flex items-center gap-2">
                <UserPlus className="w-6 h-6 text-accent" />
                New Hire Risk Watch
              </h3>
              <p className="text-sm text-text-secondary mt-1">Monitoring recent hires for early warning signs of attrition or disengagement.</p>
            </div>

            {newHires.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-left">
                  <thead>
                    <tr className="text-xs text-text-muted uppercase tracking-wider bg-surface-secondary/20 border-b border-border/50">
                      <th className="p-4">Employee</th>
                      <th className="p-4">Hire Date</th>
                      <th className="p-4">Source</th>
                      <th className="p-4 text-right">Risk Score</th>
                      <th className="p-4 text-center">Category</th>
                      <th className="p-4">Risk Factors</th>
                      <th className="p-4"></th>
                    </tr>
                  </thead>
                  <tbody className="text-sm">
                    {newHires.map((hire, i) => (
                      <tr key={i}
                        onClick={() => {
                          setSelectedEmployee(hire)
                          setIsModalOpen(true)
                        }}
                        className="border-b border-border/50 hover:bg-white/5 transition-colors cursor-pointer group"
                      >
                        <td className="p-4 font-bold text-text-primary dark:text-white group-hover:text-accent transition-colors">
                          {hire.EmployeeID}
                        </td>
                        <td className="p-4 text-text-secondary">{hire.HireDate}</td>
                        <td className="p-4 text-text-secondary">{hire.HireSource || '-'}</td>
                        <td className="p-4 text-right font-mono">
                          <span className={cn(
                            hire.risk_score > 0.7 ? 'text-danger' :
                              hire.risk_score > 0.4 ? 'text-warning' : 'text-success'
                          )}>
                            {(hire.risk_score * 100).toFixed(0)}%
                          </span>
                        </td>
                        <td className="p-4 text-center">
                          <Badge variant={
                            hire.risk_category === 'High' ? 'danger' :
                              hire.risk_category === 'Medium' ? 'warning' : 'success'
                          }>
                            {hire.risk_category}
                          </Badge>
                        </td>
                        <td className="p-4 text-text-secondary max-w-xs truncate">{hire.risk_factors_text}</td>
                        <td className="p-4 text-right">
                          <ChevronRight className="w-4 h-4 text-text-muted group-hover:text-accent transition-colors" />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="p-12 text-center text-text-secondary">No new hire risk data available.</div>
            )}
          </GlassCard>
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
