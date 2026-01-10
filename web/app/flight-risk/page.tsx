'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { KPICard } from '@/components/dashboard/kpi-card'
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
    <div className="space-y-6">
      {/* Page Title */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-text-primary dark:text-text-dark-primary">Flight Risk</h1>
          <p className="text-text-secondary dark:text-text-dark-secondary mt-1">
            AI-powered predictions of potential departures and retention insights
          </p>
        </div>
        <button
          onClick={() => setShowExplanation(true)}
          className="flex items-center gap-2 px-4 py-2 bg-accent/10 text-accent rounded-lg font-medium hover:bg-accent/20 transition-all active:scale-95"
        >
          <Info className="w-4 h-4" />
          How it works
        </button>
      </div>

      <PredictionExplanationModal
        isOpen={showExplanation}
        onClose={() => setShowExplanation(false)}
      />

      {/* Model Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <KPICard
          title="Overall Accuracy"
          value={`${(modelMetrics.accuracy * 100).toFixed(1)}%`}
          icon={Target}
          variant={modelMetrics.accuracy > 0.8 ? 'success' : 'warning'}
          insight="How often our predictions match reality across all employees"
        />
        <KPICard
          title="Model Confidence"
          value={(modelMetrics.f1 * 100).toFixed(1) + '%'}
          icon={CheckCircle}
          insight={`Based on ${modelMetrics.best_model} algorithm`}
        />
        <KPICard
          title="Detection Rate"
          value={`${(modelMetrics.recall * 100).toFixed(1)}%`}
          icon={Brain}
          insight="The percentage of actual departures we correctly identified"
        />
        <KPICard
          title="Reliability"
          value={modelMetrics.reliability}
          icon={ShieldAlert}
          variant={
            modelMetrics.reliability === 'High'
              ? 'success'
              : modelMetrics.reliability === 'Medium'
                ? 'warning'
                : 'danger'
          }
          insight="Current confidence level in the model's predictions"
        />
      </div>

      {/* Model Warnings */}
      {modelMetrics.warnings && modelMetrics.warnings.length > 0 && (
        <div className="bg-warning/10 border border-warning/20 rounded-xl p-4">
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle className="w-5 h-5 text-warning" />
            <h3 className="font-medium text-warning">Model Warnings</h3>
          </div>
          <ul className="list-disc list-inside space-y-1 text-sm text-text-secondary">
            {modelMetrics.warnings.map((warning: string, i: number) => (
              <li key={i}>{warning}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Feature Importance */}
        <Card title="What Drives Departures" subtitle="Top factors influencing employee flight risk">
          {featureImportance?.features ? (
            <FeatureImportanceChart data={featureImportance.features} />
          ) : (
            <div className="h-64 flex items-center justify-center text-text-secondary dark:text-text-dark-secondary">
              No feature importance data
            </div>
          )}
        </Card>

        {/* Risk Distribution */}
        <Card title="Risk Distribution" subtitle="Employee risk categories">
          {predictions?.distribution ? (
            <RiskDistributionPie data={predictions.distribution} />
          ) : (
            <div className="h-64 flex items-center justify-center text-text-secondary dark:text-text-dark-secondary">
              No predictions available
            </div>
          )}
        </Card>
      </div>

      {/* Second Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 9-Box Grid */}
        <Card title="9-Box Matrix" subtitle="Performance vs Potential distribution">
          {nineBox && nineBox.length > 0 ? (
            <NineBoxGrid data={nineBox} />
          ) : (
            <div className="h-64 flex items-center justify-center text-text-secondary dark:text-text-dark-secondary">
              No 9-box data available
            </div>
          )}
        </Card>

        {/* High Risk Employees */}
        <Card title="High Risk Employees" subtitle="Click on an employee to view details">
          {highRiskEmployees?.employees ? (
            <HighRiskTable
              employees={highRiskEmployees.employees}
              onEmployeeClick={setSelectedEmployee}
            />
          ) : (
            <div className="h-64 flex items-center justify-center text-text-secondary dark:text-text-dark-secondary">
              No high-risk employees
            </div>
          )}
        </Card>
      </div>

      {/* Employee Detail Modal */}
      {selectedEmployee && (
        <EmployeeDetailModal
          employeeId={selectedEmployee}
          onClose={() => setSelectedEmployee(null)}
        />
      )}

      {/* Risk Stats Summary */}
      {predictions?.distribution && (
        <Card title="Risk Summary" padding="lg">
          <div className="grid grid-cols-3 gap-8 text-center pb-2">
            <div>
              <div className="text-3xl font-bold text-danger drop-shadow-sm">
                {predictions.distribution.high_risk}
              </div>
              <div className="text-xs font-semibold uppercase tracking-wider text-text-secondary dark:text-text-dark-secondary mt-2">
                High Risk ({predictions.distribution.high_risk_pct}%)
              </div>
            </div>
            <div>
              <div className="text-3xl font-bold text-warning drop-shadow-sm">
                {predictions.distribution.medium_risk}
              </div>
              <div className="text-xs font-semibold uppercase tracking-wider text-text-secondary dark:text-text-dark-secondary mt-2">
                Medium Risk ({predictions.distribution.medium_risk_pct}%)
              </div>
            </div>
            <div>
              <div className="text-3xl font-bold text-success drop-shadow-sm">
                {predictions.distribution.low_risk}
              </div>
              <div className="text-xs font-semibold uppercase tracking-wider text-text-secondary dark:text-text-dark-secondary mt-2">
                Low Risk ({predictions.distribution.low_risk_pct}%)
              </div>
            </div>
          </div>
        </Card>
      )}
    </div>
  )
}
