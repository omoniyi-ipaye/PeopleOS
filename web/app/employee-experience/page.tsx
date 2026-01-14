'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { GlassCard } from '@/components/ui/glass-card'
import { BentoGrid, BentoGridItem } from '@/components/ui/bento-grid'
import { Badge } from '@/components/ui/badge'
import { api } from '@/lib/api-client'
import { cn } from '@/lib/utils'
import {
  Heart,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Users,
  RefreshCw,
  ChevronRight,
  Target,
  BarChart,
  Eye,
  Clock,
  Sparkles,
  PieChart,
  Layers
} from 'lucide-react'

// Consolidated Tab definitions
type ExperienceTab = 'overview' | 'analysis'

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
  warnings: string[]
  recommendations: string[]
}

export default function EmployeeExperiencePage() {
  const [activeTab, setActiveTab] = useState<ExperienceTab>('overview')

  const { data, isLoading, isError, error, refetch } = useQuery<ExperienceAnalysis>({
    queryKey: ['experience', 'analysis'],
    queryFn: () => api.experience.getAnalysis() as Promise<ExperienceAnalysis>,
  })

  // Helper components
  const EXIGauge = ({ value, size = 'large' }: { value: number; size?: 'large' | 'small' }) => {
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
          "relative flex items-center justify-center rounded-full border-8 transition-all duration-500",
          size === 'large' ? 'w-40 h-40' : 'w-20 h-20',
          value >= 80 ? 'border-success/30 bg-success/5 shadow-[0_0_20px_rgba(34,197,94,0.2)]' :
            value >= 60 ? 'border-accent/30 bg-accent/5 shadow-[0_0_20px_rgba(59,130,246,0.2)]' :
              value >= 40 ? 'border-warning/30 bg-warning/5 shadow-[0_0_20px_rgba(234,179,8,0.2)]' :
                'border-danger/30 bg-danger/5 shadow-[0_0_20px_rgba(239,68,68,0.2)]'
        )}>
          <div className="absolute inset-0 rounded-full border-4 border-white/5 animate-pulse-slow"></div>
          <span className={cn(
            "font-display font-bold",
            size === 'large' ? 'text-5xl' : 'text-xl',
            getColor(value)
          )}>
            {Math.round(value)}
          </span>
        </div>
        <span className={cn(
          "font-semibold uppercase tracking-widest",
          size === 'large' ? 'text-sm' : 'text-[10px]',
          getColor(value)
        )}>
          {getLabel(value)}
        </span>
      </div>
    )
  }

  const SegmentBar = ({ segments }: { segments: Array<{ segment: string; percentage: number }> }) => {
    const colors: Record<string, string> = {
      'Thriving': 'bg-success shadow-[0_0_10px_rgba(34,197,94,0.4)]',
      'Content': 'bg-accent shadow-[0_0_10px_rgba(59,130,246,0.4)]',
      'Neutral': 'bg-slate-400 dark:bg-slate-600',
      'Disengaged': 'bg-warning shadow-[0_0_10px_rgba(234,179,8,0.4)]',
      'Critical': 'bg-danger shadow-[0_0_10px_rgba(239,68,68,0.4)]',
    }

    return (
      <div className="w-full">
        <div className="h-4 rounded-full overflow-hidden flex bg-white/5 backdrop-blur-sm">
          {segments.map((seg) => (
            <div
              key={seg.segment}
              className={cn("h-full transition-all duration-1000 ease-out", colors[seg.segment])}
              style={{ width: `${seg.percentage}%` }}
              title={`${seg.segment}: ${seg.percentage.toFixed(1)}%`}
            />
          ))}
        </div>
        <div className="flex justify-between mt-4">
          {segments.map((seg) => (
            <div key={seg.segment} className="flex flex-col items-center">
              <div className={cn("w-2 h-2 rounded-full mb-1", colors[seg.segment].split(' ')[0])} />
              <span className="text-xs font-bold text-text-primary dark:text-white">{seg.percentage.toFixed(0)}%</span>
              <span className="text-[10px] text-text-muted uppercase tracking-wide">{seg.segment}</span>
            </div>
          ))}
        </div>
      </div>
    )
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-pulse-subtle text-text-secondary dark:text-text-dark-secondary">
          Calculating experience index...
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
    <div className="space-y-6 h-[calc(100vh-100px)] flex flex-col animate-in fade-in duration-700 slide-in-from-bottom-4">
      {/* Header with Tabs */}
      <div className="flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between flex-shrink-0">
        <div>
          <h1 className="text-4xl font-display font-bold text-gradient bg-clip-text text-transparent bg-gradient-to-r from-gray-900 to-gray-600 dark:from-white dark:to-gray-400">
            Employee Experience
          </h1>
          <p className="text-text-secondary dark:text-text-dark-secondary mt-2 text-lg font-light flex items-center gap-2">
            Unified Experience Index (EXI) & Engagement Analysis
          </p>
        </div>

        {/* Premium Tab Navigation */}
        <div className="glass p-1.5 rounded-2xl flex gap-1">
          <button
            onClick={() => setActiveTab('overview')}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-medium transition-all duration-300 ${activeTab === 'overview'
              ? 'bg-white dark:bg-slate-800 shadow-lg text-text-primary dark:text-white scale-105'
              : 'text-text-secondary dark:text-slate-400 hover:text-text-primary dark:hover:text-white hover:bg-white/10'
              }`}
          >
            <PieChart className="w-4 h-4" />
            Overview
          </button>
          <button
            onClick={() => setActiveTab('analysis')}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-medium transition-all duration-300 ${activeTab === 'analysis'
              ? 'bg-white dark:bg-slate-800 shadow-lg text-text-primary dark:text-white scale-105'
              : 'text-text-secondary dark:text-slate-400 hover:text-text-primary dark:hover:text-white hover:bg-white/10'
              }`}
          >
            <Layers className="w-4 h-4" />
            Deep-Dive Analysis
          </button>
        </div>
      </div>

      {/* Tab Content */}
      <div className="flex-1 min-h-0 overflow-y-auto pr-2 pb-4">
        {activeTab === 'overview' && (
          <div className="space-y-6">
            <GlassCard className="p-8">
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
                <div className="lg:col-span-4 flex justify-center border-r border-white/10">
                  {exi?.overall_exi !== undefined ? (
                    <div className="text-center">
                      <EXIGauge value={exi.overall_exi} />
                      <p className="mt-4 text-sm text-text-muted">Overall Experience Index</p>
                    </div>
                  ) : (
                    <div className="text-center text-text-muted">No EXI Data</div>
                  )}
                </div>
                <div className="lg:col-span-8 space-y-8">
                  <div className="grid grid-cols-3 gap-8">
                    <div className="text-center">
                      <h4 className="text-sm font-medium text-text-muted uppercase tracking-widest mb-1">Interpretation</h4>
                      <p className="text-xl font-bold text-text-primary dark:text-white">{exi?.interpretation || 'N/A'}</p>
                    </div>
                    <div className="text-center">
                      <h4 className="text-sm font-medium text-text-muted uppercase tracking-widest mb-1">Health Status</h4>
                      <div className="inline-flex items-center gap-2">
                        <div className={cn("w-2 h-2 rounded-full", summary?.health_indicator === 'Healthy' ? 'bg-success' : 'bg-warning')} />
                        <p className="text-xl font-bold text-text-primary dark:text-white">{summary?.health_indicator || 'N/A'}</p>
                      </div>
                    </div>
                    <div className="text-center">
                      <h4 className="text-sm font-medium text-text-muted uppercase tracking-widest mb-1">Total Signals</h4>
                      <p className="text-xl font-bold text-text-primary dark:text-white">{signals?.total_signals || 0}</p>
                    </div>
                  </div>

                  {/* Segments moved here per request */}
                  <div>
                    <h4 className="text-sm font-medium text-text-muted uppercase tracking-widest mb-4">Engagement Segments</h4>
                    {segments?.segments && <SegmentBar segments={segments.segments} />}
                  </div>
                </div>
              </div>
            </GlassCard>

            {/* Detailed Segments Cards (Moved from standalone tab) */}
            {segments?.available && segments.segments && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {segments.segments.map((seg) => (
                  <GlassCard key={seg.segment} className="relative overflow-hidden group">
                    <div className={cn(
                      "absolute top-0 left-0 w-1 h-full",
                      seg.segment === 'Thriving' ? 'bg-success' :
                        seg.segment === 'Content' ? 'bg-accent' :
                          seg.segment === 'Disengaged' ? 'bg-warning' :
                            seg.segment === 'Critical' ? 'bg-danger' : 'bg-slate-400'
                    )} />
                    <div className="p-2">
                      <div className="flex justify-between items-start mb-4">
                        <h3 className="text-lg font-bold">{seg.segment}</h3>
                        <Badge variant={
                          seg.segment === 'Thriving' ? 'success' :
                            seg.segment === 'Content' ? 'info' :
                              seg.segment === 'Disengaged' ? 'warning' :
                                seg.segment === 'Critical' ? 'danger' : 'outline'
                        }>{seg.percentage.toFixed(0)}%</Badge>
                      </div>
                      <div className="flex justify-between items-end">
                        <div>
                          <p className="text-xs text-text-muted uppercase tracking-wide">Count</p>
                          <p className="text-lg font-semibold">{seg.count}</p>
                        </div>
                        <div>
                          <p className="text-xs text-text-muted uppercase tracking-wide text-right">Avg EXI</p>
                          <p className="text-lg font-semibold text-right">{seg.avg_exi.toFixed(1)}</p>
                        </div>
                      </div>
                    </div>
                  </GlassCard>
                ))}
              </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <GlassCard>
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-accent" />
                  Key Recommendations
                </h3>
                <ul className="space-y-4">
                  {data?.recommendations?.slice(0, 3).map((rec, i) => (
                    <li key={i} className="flex items-start gap-4 p-4 rounded-xl bg-white/5 border border-white/10">
                      <div className="w-6 h-6 rounded-full bg-accent/20 text-accent flex items-center justify-center flex-shrink-0 text-xs font-bold mt-0.5">
                        {i + 1}
                      </div>
                      <p className="text-sm text-text-secondary">{rec}</p>
                    </li>
                  ))}
                </ul>
              </GlassCard>

              <GlassCard>
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Eye className="w-5 h-5 text-red-500" />
                  Attention Required
                </h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-4 rounded-xl bg-red-500/5 border border-red-500/10">
                    <div>
                      <p className="text-sm font-medium text-red-500">At-Risk Employees</p>
                      <p className="text-xs text-text-muted">High flight risk & low engagement</p>
                    </div>
                    <p className="text-2xl font-bold text-red-500">{summary?.at_risk_count || 0}</p>
                  </div>
                  <div className="flex items-center justify-between p-4 rounded-xl bg-amber-500/5 border border-amber-500/10">
                    <div>
                      <p className="text-sm font-medium text-amber-500">Warning Signals</p>
                      <p className="text-xs text-text-muted">Anomalies detected in pulse data</p>
                    </div>
                    <p className="text-2xl font-bold text-amber-500">{data?.warnings?.length || 0}</p>
                  </div>
                  {data?.warnings && data.warnings.length > 0 && (
                    <div className="mt-2 text-xs text-text-secondary dark:text-text-dark-secondary space-y-1 pl-1">
                      {data.warnings.map((w, i) => (
                        <p key={i} className="flex items-start gap-2">
                          <span className="text-amber-500 mt-0.5">•</span>
                          {w}
                        </p>
                      ))}
                    </div>
                  )}
                </div>
              </GlassCard>
            </div>
          </div>
        )}

        {/* Combined Analysis Tab (Drivers + Lifecycle) */}
        {activeTab === 'analysis' && (
          <div className="space-y-6">
            {/* Drivers Section */}
            <div>
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <Target className="w-5 h-5 text-accent" />
                Experience Drivers
              </h3>
              {drivers?.available && drivers.drivers && (
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
                  {drivers.drivers.map((driver, idx) => (
                    <GlassCard key={driver.factor} className="flex flex-col p-4 items-center text-center hover:scale-[1.02] transition-transform aspect-square justify-center group relative overflow-hidden">
                      <div className={cn(
                        "absolute top-0 right-0 p-2 opacity-50 transition-opacity",
                        driver.direction === 'Positive' ? 'text-success' : 'text-danger'
                      )}>
                        {driver.direction === 'Positive' ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                      </div>
                      <div className={cn(
                        "w-10 h-10 rounded-xl flex items-center justify-center text-sm font-bold shadow-glow mb-3 transition-colors",
                        idx < 3 ? 'bg-accent text-white' : 'bg-white/5 text-text-muted group-hover:bg-white/10'
                      )}>
                        {idx + 1}
                      </div>

                      <h4 className="text-sm font-bold text-text-primary dark:text-white capitalize leading-tight mb-2">
                        {driver.factor.replace(/_/g, ' ')}
                      </h4>

                      <div className="mt-auto space-y-2 w-full">
                        <Badge variant={driver.impact === 'High' ? 'danger' : 'warning'} className="text-[10px] px-2 py-0.5 w-full justify-center">
                          {driver.impact} Impact
                        </Badge>
                        <div className="text-[10px] text-text-muted">
                          Corr: {driver.correlation.toFixed(2)}
                        </div>
                      </div>
                    </GlassCard>
                  ))}
                </div>
              )}
            </div>

            {/* Lifecycle Section */}
            <div>
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <Clock className="w-5 h-5 text-accent" />
                Employee Experience by Lifecycle
              </h3>
              {lifecycle?.available && lifecycle.stages && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                  {lifecycle.stages.map((stage, idx) => (
                    <GlassCard key={stage.stage} className="relative group">
                      <div className="absolute top-2 right-2 text-4xl font-display font-bold text-white/5 z-0 group-hover:text-white/10 transition-colors">
                        {idx + 1}
                      </div>
                      <div className="relative z-10">
                        <h3 className="text-lg font-bold mb-4 shadow-glow inline-block px-2 py-1 bg-white/5 rounded-lg">{stage.stage}</h3>
                        <div className="space-y-3">
                          <div className="flex justify-between items-center text-sm">
                            <span className="text-text-muted">EXI Score</span>
                            <span className={cn(
                              "font-bold",
                              stage.avg_exi >= 60 ? 'text-success' : 'text-warning'
                            )}>{stage.avg_exi.toFixed(1)}</span>
                          </div>
                          <div className="flex justify-between items-center text-sm">
                            <span className="text-text-muted">Population</span>
                            <span className="font-bold">{stage.count}</span>
                          </div>
                          <div className="flex justify-between items-center text-sm">
                            <span className="text-text-muted">At Risk</span>
                            <span className={cn(
                              "font-bold",
                              stage.at_risk_count > 0 ? 'text-danger' : 'text-text-secondary'
                            )}>{stage.at_risk_count}</span>
                          </div>
                        </div>
                      </div>
                    </GlassCard>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
