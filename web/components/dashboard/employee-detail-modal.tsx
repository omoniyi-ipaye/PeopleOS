'use client'

import { useQuery } from '@tanstack/react-query'
import { X, User, Briefcase, Clock, DollarSign, Star, AlertTriangle } from 'lucide-react'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { ShapWaterfallChart } from '@/components/charts/shap-waterfall-chart'
import { api } from '@/lib/api-client'
import { getRiskColor } from '@/lib/utils'
import { EmployeeRiskDetail } from '@/types/api'

interface EmployeeDetailModalProps {
  employeeId: string
  onClose: () => void
}

export function EmployeeDetailModal({
  employeeId,
  onClose,
}: EmployeeDetailModalProps) {
  const { data: detail, isLoading, error } = useQuery<EmployeeRiskDetail>({
    queryKey: ['employee', employeeId],
    queryFn: () => api.predictions.getEmployeeDetail(employeeId) as any,
  })

  if (isLoading) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
        <Card className="w-full max-w-2xl max-h-[90vh] overflow-auto m-4">
          <div className="flex items-center justify-center h-64">
            <div className="animate-pulse text-text-secondary dark:text-text-dark-secondary">
              Loading employee details...
            </div>
          </div>
        </Card>
      </div>
    )
  }

  if (error || !detail) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
        <Card className="w-full max-w-md m-4">
          <div className="flex flex-col items-center gap-4 py-8">
            <AlertTriangle className="w-12 h-12 text-danger" />
            <h3 className="text-lg font-medium text-text-primary dark:text-text-dark-primary">Error Loading Employee</h3>
            <p className="text-text-secondary dark:text-text-dark-secondary text-center">
              Could not load details for employee {employeeId}
            </p>
            <Button onClick={onClose}>Close</Button>
          </div>
        </Card>
      </div>
    )
  }

  const riskColor = getRiskColor(detail.risk_category)
  const riskLabel =
    detail.risk_score >= 0.7
      ? 'High Risk'
      : detail.risk_score >= 0.4
        ? 'Medium Risk'
        : 'Low Risk'

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
      onClick={onClose}
    >
      <Card
        className="w-full max-w-3xl max-h-[90vh] overflow-auto m-4"
        onClick={(e: React.MouseEvent) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-start justify-between mb-6">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-full bg-accent/10 dark:bg-accent/20 flex items-center justify-center">
              <User className="w-6 h-6 text-accent" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-text-primary dark:text-text-dark-primary">{detail.employee_id}</h2>
              <div className="flex items-center gap-2 mt-1">
                <Badge variant="info">{detail.dept}</Badge>
                <Badge
                  variant={
                    detail.risk_score >= 0.7
                      ? 'danger'
                      : detail.risk_score >= 0.4
                        ? 'warning'
                        : 'success'
                  }
                >
                  {riskLabel}
                </Badge>
              </div>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-surface-hover dark:hover:bg-surface-dark-hover transition-colors"
          >
            <X className="w-5 h-5 text-text-muted dark:text-text-dark-muted" />
          </button>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="p-4 rounded-lg bg-surface-hover dark:bg-surface-dark-hover border border-border/50 dark:border-border-dark/50">
            <div className="flex items-center gap-2 text-text-muted dark:text-text-dark-muted mb-1">
              <Clock className="w-4 h-4" />
              <span className="text-xs">Tenure</span>
            </div>
            <div className="text-lg font-semibold text-text-primary dark:text-text-dark-primary">
              {detail.tenure ? `${detail.tenure.toFixed(1)} yrs` : 'N/A'}
            </div>
          </div>
          <div className="p-4 rounded-lg bg-surface-hover dark:bg-surface-dark-hover border border-border/50 dark:border-border-dark/50">
            <div className="flex items-center gap-2 text-text-muted dark:text-text-dark-muted mb-1">
              <DollarSign className="w-4 h-4" />
              <span className="text-xs">Salary</span>
            </div>
            <div className="text-lg font-semibold text-text-primary dark:text-text-dark-primary">
              {detail.salary
                ? `$${Math.round(detail.salary).toLocaleString()}`
                : 'N/A'}
            </div>
          </div>
          <div className="p-4 rounded-lg bg-surface-hover dark:bg-surface-dark-hover border border-border/50 dark:border-border-dark/50">
            <div className="flex items-center gap-2 text-text-muted dark:text-text-dark-muted mb-1">
              <Star className="w-4 h-4" />
              <span className="text-xs">Last Rating</span>
            </div>
            <div className="text-lg font-semibold text-text-primary dark:text-text-dark-primary">
              {detail.last_rating?.toFixed(1) ?? 'N/A'}
            </div>
          </div>
          <div className="p-4 rounded-lg bg-surface-hover dark:bg-surface-dark-hover border border-border/50 dark:border-border-dark/50">
            <div className="flex items-center gap-2 text-text-muted dark:text-text-dark-muted mb-1">
              <Briefcase className="w-4 h-4" />
              <span className="text-xs">Age</span>
            </div>
            <div className="text-lg font-semibold text-text-primary dark:text-text-dark-primary">{detail.age ?? 'N/A'}</div>
          </div>
        </div>

        {/* Risk Score */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-text-secondary dark:text-text-dark-secondary mb-3">
            Attrition Risk Score
          </h3>
          <div className="flex items-center gap-4">
            <div className="flex-1 h-3 bg-surface-hover dark:bg-surface-dark rounded-full overflow-hidden border border-border/50 dark:border-border-dark/50">
              <div
                className={`h-full rounded-full transition-all ${detail.risk_score >= 0.7
                  ? 'bg-danger'
                  : detail.risk_score >= 0.4
                    ? 'bg-warning'
                    : 'bg-success'
                  }`}
                style={{ width: `${detail.risk_score * 100}%` }}
              />
            </div>
            <span className="text-lg font-mono font-bold" style={{ color: riskColor }}>
              {(detail.risk_score * 100).toFixed(1)}%
            </span>
          </div>
        </div>

        {/* SHAP Explanation */}
        {detail.drivers && detail.drivers.length > 0 && (
          <div>
            <h3 className="text-sm font-medium text-text-secondary dark:text-text-dark-secondary mb-3">
              Risk Drivers (SHAP Analysis)
            </h3>
            <div className="p-4 rounded-lg bg-surface-hover dark:bg-surface-dark-hover border border-border/50 dark:border-border-dark/50">
              <ShapWaterfallChart
                features={detail.drivers.slice(0, 8).map((sv: any) => ({
                  feature: sv.feature,
                  value: sv.value,
                  contribution: sv.contribution,
                }))}
                baseValue={detail.base_value || 0.5}
                prediction={detail.risk_score}
              />
            </div>
          </div>
        )}

        {/* Recommendations */}
        {detail.recommendations && detail.recommendations.length > 0 && (
          <div className="mt-6">
            <h3 className="text-sm font-medium text-text-secondary dark:text-text-dark-secondary mb-3">
              Recommended Actions
            </h3>
            <ul className="space-y-2">
              {detail.recommendations.map((action: string, index: number) => (
                <li
                  key={index}
                  className="flex items-start gap-2 text-sm text-text-secondary dark:text-text-dark-secondary"
                >
                  <span className="w-1.5 h-1.5 rounded-full bg-accent mt-1.5 flex-shrink-0" />
                  {action}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Actions */}
        <div className="flex justify-end gap-3 mt-6 pt-6 border-t border-border dark:border-border-dark">
          <Button variant="secondary" onClick={onClose}>
            Close
          </Button>
        </div>
      </Card>
    </div>
  )
}
