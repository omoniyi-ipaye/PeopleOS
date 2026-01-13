'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Card } from '../../components/ui/card'
import { Badge } from '../../components/ui/badge'
import { api } from '../../lib/api-client'
import { cn } from '../../lib/utils'
import {
  Heart,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Users,
  RefreshCw,
  ChevronRight,
} from 'lucide-react'
import { ExplanationBox } from '../../components/ui/metric-explainer'
import { TabGroup } from '../../components/ui/tab-group'

type ExperienceTab = 'overview' | 'segments' | 'drivers' | 'watchlist' | 'lifecycle'

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'segments', label: 'Segments' },
  { id: 'drivers', label: 'Drivers' },
  { id: 'watchlist', label: 'Watch List' },
  { id: 'lifecycle', label: 'Lifecycle' },
] as const

interface ExperienceAnalysis {
  experience_index: {
    available: boolean
    overall_exi?: number
    exi_std?: number
    total_employees?: number
    interpretation?: string
    benchmark?: string
  }
  segments: {
    available: boolean
    segments?: Array<{
      segment: string
      count: number
      percentage: number
      avg_exi: number
    }>
    health_indicator?: string
    thriving_percentage?: number
    at_risk_percentage?: number
  }
  drivers: {
    available: boolean
    drivers?: Array<{
      factor: string
      correlation: number
      impact: string
      direction: string
    }>
  }
  at_risk: {
    available: boolean
    total_at_risk?: number
    employees?: Array<{
      EmployeeID: string
      Dept?: string
      current_exi: number
      segment: string
      risk_factors: string[]
      recommended_actions: string[]
    }>
  }
  lifecycle: {
    available: boolean
    stages?: Array<{
      stage: string
      count: number
      avg_exi: number
      at_risk_count: number
    }>
  }
  manager_impact: {
    available: boolean
    managers_analyzed?: number
    overall_avg_exi?: number
    bottom_managers?: Array<{
      ManagerID: string
      team_size: number
      avg_team_exi: number
    }>
  }
  signals: {
    has_enps: boolean
    has_pulse: boolean
    total_signals: number
    recommendations: string[]
  }
  summary: {
    overall_exi?: number
    health_indicator: string
    total_employees: number
    at_risk_count: number
  }
  recommendations: string[]
  warnings: string[]
}

function EXIGauge({ value, size = 'large' }: { value: number; size?: 'large' | 'small' }) {
  const getColor = (exi: number) => {
    if (exi >= 80) return 'text-success'
    if (exi >= 60) return 'text-accent'
    if (exi >= 40) return 'text-warning'
    return 'text-danger'
  }

  const getLabel = (exi: number) => {
    if (exi >= 80) return 'Thriving'
    if (exi >= 60) return 'Content'
    if (exi >= 40) return 'Neutral'
    if (exi >= 20) return 'Disengaged'
    return 'Critical'
  }

  return (
    <div className={cn("flex flex-col items-center", size === 'large' ? 'gap-2' : 'gap-1')}>
      <div className={cn(
        "relative flex items-center justify-center rounded-full border-8",
        size === 'large' ? 'w-32 h-32' : 'w-20 h-20',
        value >= 80 ? 'border-success/30 bg-success/5' :
        value >= 60 ? 'border-accent/30 bg-accent/5' :
        value >= 40 ? 'border-warning/30 bg-warning/5' :
        'border-danger/30 bg-danger/5'
      )}>
        <span className={cn(
          "font-bold",
          size === 'large' ? 'text-3xl' : 'text-xl',
          getColor(value)
        )}>
          {Math.round(value)}
        </span>
      </div>
      <span className={cn(
        "font-semibold",
        size === 'large' ? 'text-sm' : 'text-xs',
        getColor(value)
      )}>
        {getLabel(value)}
      </span>
    </div>
  )
}

function SegmentBar({ segments }: { segments: Array<{ segment: string; percentage: number }> }) {
  const colors: Record<string, string> = {
    'Thriving': 'bg-success',
    'Content': 'bg-accent',
    'Neutral': 'bg-gray-400',
    'Disengaged': 'bg-warning',
    'Critical': 'bg-danger',
  }

  return (
    <div className="w-full">
      <div className="h-6 rounded-full overflow-hidden flex">
        {segments.map((seg) => (
          <div
            key={seg.segment}
            className={cn("h-full", colors[seg.segment] || 'bg-gray-300')}
            style={{ width: `${seg.percentage}%` }}
            title={`${seg.segment}: ${seg.percentage.toFixed(1)}%`}
          />
        ))}
      </div>
      <div className="flex justify-between mt-2 text-xs text-text-muted">
        {segments.map((seg) => (
          <div key={seg.segment} className="flex items-center gap-1">
            <div className={cn("w-2 h-2 rounded-full", colors[seg.segment])} />
            <span>{seg.segment} ({seg.percentage.toFixed(0)}%)</span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function EmployeeExperiencePage() {
  const [activeTab, setActiveTab] = useState<ExperienceTab>('overview')

  const { data, isLoading, isError, error, refetch } = useQuery<ExperienceAnalysis>({
    queryKey: ['experience', 'analysis'],
    queryFn: () => api.experience.getAnalysis() as Promise<ExperienceAnalysis>,
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-pulse-subtle text-text-secondary dark:text-text-dark-secondary">
          Analyzing employee experience...
        </div>
      </div>
    )
  }

  if (isError) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4 text-center">
        <AlertTriangle className="w-12 h-12 text-danger" />
        <h2 className="text-xl font-semibold text-text-primary dark:text-text-dark-primary">
          Failed to Load Experience Data
        </h2>
        <p className="text-text-secondary dark:text-text-dark-secondary max-w-md">
          {error instanceof Error ? error.message : 'Unable to load experience metrics. Please try again.'}
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

  const exi = data?.experience_index
  const segments = data?.segments
  const drivers = data?.drivers
  const atRisk = data?.at_risk
  const lifecycle = data?.lifecycle
  const signals = data?.signals
  const summary = data?.summary

  return (
    <div className="space-y-6">
      {/* Page Title */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-text-primary dark:text-text-dark-primary">Employee Experience</h1>
          <p className="text-text-secondary dark:text-text-dark-secondary mt-1">
            Unified experience index and engagement insights
          </p>
        </div>

        <TabGroup<ExperienceTab>
          tabs={TABS}
          activeTab={activeTab}
          onTabChange={setActiveTab}
        />
      </div>

      <ExplanationBox title={`About: ${activeTab === 'overview' ? 'Experience Index' : activeTab === 'segments' ? 'Engagement Segments' : activeTab === 'drivers' ? 'Experience Drivers' : activeTab === 'watchlist' ? 'At-Risk Employees' : 'Lifecycle Analysis'}`}>
        {activeTab === 'overview' && "The Employee Experience Index (EXI) combines multiple signals into a single 0-100 score measuring overall employee experience."}
        {activeTab === 'segments' && "Employees are segmented by engagement level: Thriving, Content, Neutral, Disengaged, and Critical."}
        {activeTab === 'drivers' && "Factors that statistically correlate with higher or lower experience scores."}
        {activeTab === 'watchlist' && "Employees with low experience scores who may need intervention."}
        {activeTab === 'lifecycle' && "How experience varies by tenure stage, from new hires to veterans."}
      </ExplanationBox>

      {/* Warnings */}
      {data?.warnings && data.warnings.length > 0 && (
        <div className="bg-warning/10 border border-warning/30 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-warning mt-0.5" />
            <div>
              <p className="font-medium text-warning">Limited Data Available</p>
              <ul className="text-sm text-text-secondary mt-1 space-y-1">
                {data.warnings.map((w, i) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'overview' && (
        <>
          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card className="p-6 flex flex-col items-center justify-center">
              {exi?.available && exi.overall_exi !== undefined ? (
                <EXIGauge value={exi.overall_exi} />
              ) : (
                <div className="text-center text-text-muted">No EXI data</div>
              )}
              <p className="text-xs text-text-muted mt-2">Experience Index</p>
            </Card>

            <Card className="p-6">
              <div className="flex items-center gap-3 mb-2">
                <div className="p-2 bg-accent/10 rounded-lg">
                  <Users className="w-5 h-5 text-accent" />
                </div>
                <span className="text-sm text-text-secondary">Total Employees</span>
              </div>
              <p className="text-2xl font-bold">{summary?.total_employees || 0}</p>
            </Card>

            <Card className="p-6">
              <div className="flex items-center gap-3 mb-2">
                <div className={cn(
                  "p-2 rounded-lg",
                  summary?.health_indicator === 'Healthy' ? 'bg-success/10' :
                  summary?.health_indicator === 'Moderate' ? 'bg-warning/10' :
                  'bg-danger/10'
                )}>
                  <Heart className={cn(
                    "w-5 h-5",
                    summary?.health_indicator === 'Healthy' ? 'text-success' :
                    summary?.health_indicator === 'Moderate' ? 'text-warning' :
                    'text-danger'
                  )} />
                </div>
                <span className="text-sm text-text-secondary">Health Status</span>
              </div>
              <p className="text-xl font-bold">{summary?.health_indicator || 'Unknown'}</p>
            </Card>

            <Card className="p-6">
              <div className="flex items-center gap-3 mb-2">
                <div className="p-2 bg-danger/10 rounded-lg">
                  <AlertTriangle className="w-5 h-5 text-danger" />
                </div>
                <span className="text-sm text-text-secondary">At Risk</span>
              </div>
              <p className="text-2xl font-bold text-danger">{summary?.at_risk_count || 0}</p>
            </Card>
          </div>

          {/* Segment Distribution */}
          {segments?.available && segments.segments && (
            <Card title="Engagement Distribution" subtitle="Workforce segmented by experience level">
              <div className="p-4">
                <SegmentBar segments={segments.segments} />
              </div>
            </Card>
          )}

          {/* Signals Available */}
          <Card title="Available Signals" subtitle="Experience data sources detected in your data">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-4">
              <div className={cn(
                "p-3 rounded-lg border",
                signals?.has_enps ? 'border-success/30 bg-success/5' : 'border-border bg-surface-secondary'
              )}>
                <p className="text-sm font-medium">eNPS</p>
                <Badge variant={signals?.has_enps ? 'success' : 'outline'}>
                  {signals?.has_enps ? 'Available' : 'Not Found'}
                </Badge>
              </div>
              <div className={cn(
                "p-3 rounded-lg border",
                signals?.has_pulse ? 'border-success/30 bg-success/5' : 'border-border bg-surface-secondary'
              )}>
                <p className="text-sm font-medium">Pulse Survey</p>
                <Badge variant={signals?.has_pulse ? 'success' : 'outline'}>
                  {signals?.has_pulse ? 'Available' : 'Not Found'}
                </Badge>
              </div>
              <div className="p-3 rounded-lg border border-border bg-surface-secondary">
                <p className="text-sm font-medium">Total Signals</p>
                <Badge variant="info">{signals?.total_signals || 0}</Badge>
              </div>
            </div>
            {signals?.recommendations && signals.recommendations.length > 0 && (
              <div className="px-4 pb-4">
                <p className="text-xs text-text-muted mb-2">Recommendations:</p>
                <ul className="text-sm text-text-secondary space-y-1">
                  {signals.recommendations.map((r, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <ChevronRight className="w-4 h-4 text-accent mt-0.5" />
                      {r}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </Card>
        </>
      )}

      {activeTab === 'segments' && segments?.available && segments.segments && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {segments.segments.map((seg) => (
            <Card key={seg.segment} className="p-6">
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h3 className="font-bold text-lg">{seg.segment}</h3>
                  <p className="text-sm text-text-muted">
                    {seg.segment === 'Thriving' && 'Highly engaged advocates (80-100)'}
                    {seg.segment === 'Content' && 'Satisfied employees (60-79)'}
                    {seg.segment === 'Neutral' && 'Neither engaged nor disengaged (40-59)'}
                    {seg.segment === 'Disengaged' && 'At risk, showing warning signs (20-39)'}
                    {seg.segment === 'Critical' && 'Immediate intervention needed (0-19)'}
                  </p>
                </div>
                <Badge variant={
                  seg.segment === 'Thriving' ? 'success' :
                  seg.segment === 'Content' ? 'info' :
                  seg.segment === 'Neutral' ? 'outline' :
                  seg.segment === 'Disengaged' ? 'warning' : 'danger'
                }>
                  {seg.percentage.toFixed(1)}%
                </Badge>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-text-secondary">Count</span>
                  <span className="font-medium">{seg.count}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-text-secondary">Avg EXI</span>
                  <span className="font-medium">{seg.avg_exi.toFixed(1)}</span>
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}

      {activeTab === 'drivers' && drivers?.available && drivers.drivers && (
        <Card title="Experience Drivers" subtitle="Factors that correlate with experience scores">
          <div className="divide-y divide-border">
            {drivers.drivers.map((driver, idx) => (
              <div key={driver.factor} className="flex items-center justify-between p-4">
                <div className="flex items-center gap-4">
                  <div className={cn(
                    "w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm",
                    idx < 3 ? 'bg-accent/20 text-accent' : 'bg-surface-secondary text-text-muted'
                  )}>
                    {idx + 1}
                  </div>
                  <div>
                    <p className="font-medium">{driver.factor.replace(/_/g, ' ')}</p>
                    <p className="text-xs text-text-muted">
                      Correlation: {driver.correlation.toFixed(3)}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {driver.direction === 'Positive' ? (
                    <TrendingUp className="w-4 h-4 text-success" />
                  ) : (
                    <TrendingDown className="w-4 h-4 text-danger" />
                  )}
                  <Badge variant={
                    driver.impact === 'High' ? 'danger' :
                    driver.impact === 'Medium' ? 'warning' : 'outline'
                  }>
                    {driver.impact} Impact
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {activeTab === 'watchlist' && atRisk?.available && (
        <Card title={`At-Risk Employees (${atRisk.total_at_risk || 0})`} subtitle="Employees with low experience scores needing attention">
          {atRisk.employees && atRisk.employees.length > 0 ? (
            <div className="divide-y divide-border">
              {atRisk.employees.map((emp) => (
                <div key={emp.EmployeeID} className="p-4">
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <p className="font-medium">{emp.EmployeeID}</p>
                      <p className="text-sm text-text-muted">{emp.Dept || 'Unknown Dept'}</p>
                    </div>
                    <div className="flex items-center gap-2">
                      <EXIGauge value={emp.current_exi} size="small" />
                    </div>
                  </div>
                  {emp.risk_factors.length > 0 && (
                    <div className="mb-2">
                      <p className="text-xs font-medium text-text-muted mb-1">Risk Factors:</p>
                      <div className="flex flex-wrap gap-1">
                        {emp.risk_factors.map((rf, i) => (
                          <Badge key={i} variant="danger">{rf}</Badge>
                        ))}
                      </div>
                    </div>
                  )}
                  {emp.recommended_actions.length > 0 && (
                    <div>
                      <p className="text-xs font-medium text-text-muted mb-1">Recommended Actions:</p>
                      <ul className="text-sm text-text-secondary space-y-1">
                        {emp.recommended_actions.map((action, i) => (
                          <li key={i} className="flex items-start gap-2">
                            <ChevronRight className="w-4 h-4 text-accent mt-0.5 flex-shrink-0" />
                            {action}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="p-8 text-center text-text-muted">
              No at-risk employees found
            </div>
          )}
        </Card>
      )}

      {activeTab === 'lifecycle' && lifecycle?.available && lifecycle.stages && (
        <Card title="Experience by Lifecycle Stage" subtitle="How experience varies by tenure">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 p-4">
            {lifecycle.stages.map((stage) => (
              <div key={stage.stage} className="p-4 rounded-lg border border-border bg-surface-secondary">
                <h4 className="font-bold mb-2">{stage.stage}</h4>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-text-muted">Employees</span>
                    <span className="font-medium">{stage.count}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-text-muted">Avg EXI</span>
                    <span className={cn(
                      "font-medium",
                      stage.avg_exi >= 60 ? 'text-success' :
                      stage.avg_exi >= 40 ? 'text-warning' : 'text-danger'
                    )}>
                      {stage.avg_exi.toFixed(1)}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-text-muted">At Risk</span>
                    <span className={cn(
                      "font-medium",
                      stage.at_risk_count > 0 ? 'text-danger' : 'text-success'
                    )}>
                      {stage.at_risk_count}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Recommendations */}
      {data?.recommendations && data.recommendations.length > 0 && (
        <Card title="Recommendations" subtitle="Data-driven suggestions to improve experience">
          <ul className="divide-y divide-border">
            {data.recommendations.map((rec, i) => (
              <li key={i} className="flex items-start gap-3 p-4">
                <div className="w-6 h-6 rounded-full bg-accent/20 text-accent flex items-center justify-center flex-shrink-0 text-sm font-bold">
                  {i + 1}
                </div>
                <p className="text-sm text-text-secondary">{rec}</p>
              </li>
            ))}
          </ul>
        </Card>
      )}
    </div>
  )
}
