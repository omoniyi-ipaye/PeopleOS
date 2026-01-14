'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { GlassCard } from '@/components/ui/glass-card'
import { BentoGrid, BentoGridItem } from '@/components/ui/bento-grid'
import { FeatureImportanceChart } from '@/components/charts/feature-importance-chart'
import { RiskDistributionPie } from '@/components/charts/risk-distribution-pie'
import { NineBoxGrid } from '@/components/charts/nine-box-grid'
import { HighRiskTable, type HighRiskEmployeeRow } from '@/components/dashboard/high-risk-table'
import { EmployeeDetailModal } from '@/components/dashboard/employee-detail-modal'
import { api } from '@/lib/api-client'
import { Brain, Target, CheckCircle, AlertTriangle, Info, ShieldAlert, RefreshCw } from 'lucide-react'
import { PredictionExplanationModal } from '@/components/diagnostics/prediction-explanation-modal'

import type {
  ModelMetrics,
  FeatureImportance,
  PredictionSummary,
  NineBoxSummary,
} from '@/types/api'

export default function FlightRiskPage() {
  const [activeTab, setActiveTab] = useState<'overview' | 'analysis' | 'employees'>('overview')
  const [selectedEmployee, setSelectedEmployee] = useState<string | null>(null)
  const [showExplanation, setShowExplanation] = useState(false)

  const { data: modelMetrics, isLoading, isError, error, refetch } = useQuery<ModelMetrics>({
    queryKey: ['predictions', 'model-metrics'],
    queryFn: () => api.predictions.getModelMetrics() as Promise<ModelMetrics>,
  })

  const { data: featureImportance } = useQuery<FeatureImportance>({
    queryKey: ['predictions', 'feature-importance'],
    queryFn: () => api.predictions.getFeatureImportance(10) as Promise<FeatureImportance>,
  })

  const { data: predictions } = useQuery<PredictionSummary>({
    queryKey: ['predictions', 'risk'],
    queryFn: () => api.predictions.getRisk(undefined, 100) as Promise<PredictionSummary>,
  })

  const { data: highRiskEmployees } = useQuery<{ employees: HighRiskEmployeeRow[] }>({
    queryKey: ['predictions', 'high-risk'],
    queryFn: () => api.predictions.getHighRisk(20) as Promise<{ employees: HighRiskEmployeeRow[] }>,
  })

  const { data: nineBox } = useQuery<NineBoxSummary[]>({
    queryKey: ['succession', '9box-summary'],
    queryFn: () => api.succession.get9BoxSummary() as Promise<NineBoxSummary[]>,
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-pulse-subtle text-text-secondary dark:text-text-dark-secondary">
          Loading predictions...
        </div>
      </div>
    )
  }

  if (isError) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4 text-center">
        <AlertTriangle className="w-12 h-12 text-danger" />
        <h2 className="text-xl font-semibold text-text-primary dark:text-text-dark-primary">
          Failed to Load Predictions
        </h2>
        <p className="text-text-secondary dark:text-text-dark-secondary max-w-md">
          {error instanceof Error ? error.message : 'Unable to load flight risk predictions. Please try again.'}
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

  if (!modelMetrics) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4 text-center">
        <AlertTriangle className="w-12 h-12 text-warning" />
        <h2 className="text-xl font-semibold text-text-primary dark:text-text-dark-primary">Predictive Analytics Unavailable</h2>
        <p className="text-text-secondary dark:text-text-dark-secondary max-w-md">
          Upload data with an Attrition column to enable ML-based risk predictions.
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-6 h-[calc(100vh-100px)] flex flex-col animate-in fade-in duration-700 slide-in-from-bottom-4">
      {/* Header with Tabs */}
      <div className="flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between flex-shrink-0">
        <div>
          <h1 className="text-4xl font-display font-bold text-gradient bg-clip-text text-transparent bg-gradient-to-r from-gray-900 to-gray-600 dark:from-white dark:to-gray-400">
            Flight Risk
          </h1>
          <p className="text-text-secondary dark:text-text-dark-secondary mt-2 text-lg font-light">
            AI-powered retention intelligence
          </p>
        </div>

        <div className="flex items-center gap-4">
          <button
            onClick={() => setShowExplanation(true)}
            className="flex items-center gap-2 px-4 py-2 bg-accent/10 text-accent rounded-xl font-medium hover:bg-accent/20 transition-all active:scale-95 text-sm"
          >
            <Info className="w-4 h-4" />
            How it works
          </button>

          {/* Premium Tab Navigation */}
          <div className="glass p-1.5 rounded-2xl flex gap-1">
            <button
              onClick={() => setActiveTab('overview')}
              className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-medium transition-all duration-300 ${activeTab === 'overview'
                ? 'bg-white dark:bg-slate-800 shadow-lg text-text-primary dark:text-white scale-105'
                : 'text-text-secondary dark:text-slate-400 hover:text-text-primary dark:hover:text-white hover:bg-white/10'
                }`}
            >
              <Target className="w-4 h-4" />
              Overview
            </button>
            <button
              onClick={() => setActiveTab('analysis')}
              className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-medium transition-all duration-300 ${activeTab === 'analysis'
                ? 'bg-white dark:bg-slate-800 shadow-lg text-text-primary dark:text-white scale-105'
                : 'text-text-secondary dark:text-slate-400 hover:text-text-primary dark:hover:text-white hover:bg-white/10'
                }`}
            >
              <Brain className="w-4 h-4" />
              Analysis
            </button>
            <button
              onClick={() => setActiveTab('employees')}
              className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-medium transition-all duration-300 ${activeTab === 'employees'
                ? 'bg-white dark:bg-slate-800 shadow-lg text-text-primary dark:text-white scale-105'
                : 'text-text-secondary dark:text-slate-400 hover:text-text-primary dark:hover:text-white hover:bg-white/10'
                }`}
            >
              <ShieldAlert className="w-4 h-4" />
              Employees
            </button>
          </div>
        </div>
      </div>

      <PredictionExplanationModal
        isOpen={showExplanation}
        onClose={() => setShowExplanation(false)}
      />

      {/* Tab Content Area */}
      <div className="flex-1 min-h-0 overflow-y-auto pr-2 pb-4">
        {activeTab === 'overview' && (
          <div className="space-y-6">
            <BentoGrid>
              <BentoGridItem
                title="Overall Accuracy"
                header={<div className="text-4xl font-display font-bold text-text-primary dark:text-white">{(modelMetrics.accuracy * 100).toFixed(1)}%</div>}
                icon={<Target className="w-5 h-5 text-accent" />}
                description="Prediction match rate"
                className={modelMetrics.accuracy > 0.8 ? "border-l-4 border-success" : "border-l-4 border-warning"}
              />
              <BentoGridItem
                title="Confidence Score"
                header={<div className="text-4xl font-display font-bold text-text-primary dark:text-white">{(modelMetrics.f1 * 100).toFixed(1)}%</div>}
                icon={<CheckCircle className="w-5 h-5 text-blue-500" />}
                description={`Algorithm: ${modelMetrics.best_model}`}
                className="border-l-4 border-blue-500"
              />
              <BentoGridItem
                title="Detection Rate"
                header={<div className="text-4xl font-display font-bold text-text-primary dark:text-white">{(modelMetrics.recall * 100).toFixed(1)}%</div>}
                icon={<Brain className="w-5 h-5 text-purple-500" />}
                description="Actual leavers identified"
                className="border-l-4 border-purple-500"
              />
              <BentoGridItem
                title="Reliability"
                header={<div className="text-4xl font-display font-bold text-text-primary dark:text-white">{modelMetrics.reliability}</div>}
                icon={<ShieldAlert className="w-5 h-5 text-emerald-500" />}
                description="Model stability"
                className="border-l-4 border-emerald-500"
              />
            </BentoGrid>

            {/* Risk Summary Grid */}
            {predictions?.distribution && (
              <GlassCard className="p-8">
                <h3 className="text-lg font-semibold mb-6 flex items-center gap-2">
                  <div className="w-1 h-6 bg-slate-500 rounded-full" />
                  Risk Distribution Summary
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                  <div className="bg-red-500/10 dark:bg-red-500/5 p-6 rounded-2xl border border-red-500/20 text-center relative overflow-hidden group">
                    <div className="absolute inset-0 bg-red-500/5 group-hover:bg-red-500/10 transition-colors" />
                    <div className="relative text-5xl font-bold text-red-500 mb-2">{predictions.distribution.high_risk}</div>
                    <div className="relative text-sm font-bold uppercase tracking-widest text-red-600/70 dark:text-red-400">High Risk</div>
                    <div className="mt-2 text-xs text-text-muted">{predictions.distribution.high_risk_pct}% of workforce</div>
                  </div>

                  <div className="bg-amber-500/10 dark:bg-amber-500/5 p-6 rounded-2xl border border-amber-500/20 text-center relative overflow-hidden group">
                    <div className="absolute inset-0 bg-amber-500/5 group-hover:bg-amber-500/10 transition-colors" />
                    <div className="relative text-5xl font-bold text-amber-500 mb-2">{predictions.distribution.medium_risk}</div>
                    <div className="relative text-sm font-bold uppercase tracking-widest text-amber-600/70 dark:text-amber-400">Medium Risk</div>
                    <div className="mt-2 text-xs text-text-muted">{predictions.distribution.medium_risk_pct}% of workforce</div>
                  </div>

                  <div className="bg-emerald-500/10 dark:bg-emerald-500/5 p-6 rounded-2xl border border-emerald-500/20 text-center relative overflow-hidden group">
                    <div className="absolute inset-0 bg-emerald-500/5 group-hover:bg-emerald-500/10 transition-colors" />
                    <div className="relative text-5xl font-bold text-emerald-500 mb-2">{predictions.distribution.low_risk}</div>
                    <div className="relative text-sm font-bold uppercase tracking-widest text-emerald-600/70 dark:text-emerald-400">Low Risk</div>
                    <div className="mt-2 text-xs text-text-muted">{predictions.distribution.low_risk_pct}% of workforce</div>
                  </div>
                </div>
              </GlassCard>
            )}

            {/* Warnings */}
            {modelMetrics.warnings && modelMetrics.warnings.length > 0 && (
              <div className="glass p-4 rounded-xl border-l-4 border-warning flex items-start gap-4">
                <AlertTriangle className="w-5 h-5 text-warning flex-shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-warning mb-1">Model Calibrations</h4>
                  <ul className="list-disc list-inside space-y-1 text-sm text-text-secondary">
                    {modelMetrics.warnings.map((warning, i) => (
                      <li key={i}>{warning}</li>
                    ))}
                  </ul>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'analysis' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 h-full">
            <GlassCard className="h-full flex flex-col min-h-[500px]">
              <h3 className="text-lg font-semibold mb-6 flex items-center gap-2">
                <div className="w-1 h-6 bg-teal-500 rounded-full" />
                Key Risk Drivers
              </h3>
              <div className="flex-1">
                {featureImportance?.features ? (
                  <FeatureImportanceChart data={featureImportance.features} />
                ) : (
                  <div className="h-full flex items-center justify-center text-text-muted">No data</div>
                )}
              </div>
            </GlassCard>

            <GlassCard className="h-full flex flex-col min-h-[500px]">
              <h3 className="text-lg font-semibold mb-6 flex items-center gap-2">
                <div className="w-1 h-6 bg-indigo-500 rounded-full" />
                Risk Distribution
              </h3>
              <div className="flex-1">
                {predictions?.distribution ? (
                  <RiskDistributionPie data={predictions.distribution} />
                ) : (
                  <div className="h-full flex items-center justify-center text-text-muted">No data</div>
                )}
              </div>
            </GlassCard>
          </div>
        )}

        {activeTab === 'employees' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 h-full">
            <GlassCard className="h-full flex flex-col min-h-[500px]">
              <h3 className="text-lg font-semibold mb-6 flex items-center gap-2">
                <div className="w-1 h-6 bg-purple-500 rounded-full" />
                9-Box Matrix (Performance vs Potential)
              </h3>
              <div className="flex-1">
                {nineBox && nineBox.length > 0 ? (
                  <NineBoxGrid data={nineBox} />
                ) : (
                  <div className="h-full flex items-center justify-center text-text-muted">No data</div>
                )}
              </div>
            </GlassCard>

            <GlassCard className="h-full flex flex-col min-h-[500px]">
              <h3 className="text-lg font-semibold mb-6 flex items-center gap-2">
                <div className="w-1 h-6 bg-rose-500 rounded-full" />
                High Risk Employees
              </h3>
              <div className="flex-1 overflow-auto">
                {highRiskEmployees?.employees ? (
                  <HighRiskTable
                    employees={highRiskEmployees.employees}
                    onEmployeeClick={setSelectedEmployee}
                  />
                ) : (
                  <div className="h-full flex items-center justify-center text-text-muted">No high-risk employees</div>
                )}
              </div>
            </GlassCard>
          </div>
        )}
      </div>

      {/* Employee Detail Modal */}
      {selectedEmployee && (
        <EmployeeDetailModal
          employeeId={selectedEmployee}
          onClose={() => setSelectedEmployee(null)}
        />
      )}
    </div>
  )
}
