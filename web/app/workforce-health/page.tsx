'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { GlassCard } from '@/components/ui/glass-card'
import { BentoGrid, BentoGridItem } from '@/components/ui/bento-grid'
import { DepartmentBarChart } from '@/components/charts/department-bar-chart'
import { CorrelationHeatmap } from '@/components/charts/correlation-heatmap'
import { api } from '@/lib/api-client'
import { cn } from '@/lib/utils'
import {
  AlertTriangle,
  TrendingDown,
  Activity,
  DollarSign,
  Users,
  MessageSquare,
  Heart,
  Target,
  ArrowRight
} from 'lucide-react'

import { TabGroup } from '@/components/ui/tab-group'
import type {
  DepartmentStats,
  DepartmentList,
  CorrelationData,
  CorrelationsResponse,
  HighRiskDepartment,
  HighRiskDepartmentsResponse,
} from '@/types/api'

// Import existing tabs content components - we will keep using these logic units but wrap them
import { CompensationTab } from '@/components/diagnostics/compensation-tab'
import { SuccessionTab } from '@/components/diagnostics/succession-tab'
import { NLPTab } from '@/components/diagnostics/nlp-tab'
import { TeamDynamicsTab } from '@/components/diagnostics/team-dynamics-tab'
import { SegmentsTab } from '@/components/diagnostics/segments-tab'

// Mapped to Premium Tabs
type WorkforceTab = 'pulse' | 'compensation' | 'succession' | 'insights' | 'segments'

export default function WorkforceHealthPage() {
  const [activeTab, setActiveTab] = useState<WorkforceTab>('pulse')

  const { data: departmentData, isLoading: deptLoading, isError, error, refetch } = useQuery<DepartmentList>({
    queryKey: ['analytics', 'departments'],
    queryFn: () => api.analytics.getDepartments() as Promise<DepartmentList>,
  })
  const departments: DepartmentStats[] = departmentData?.departments || []

  const { data: correlationData } = useQuery<CorrelationsResponse>({
    queryKey: ['analytics', 'correlations'],
    queryFn: () => api.analytics.getCorrelations(15) as Promise<CorrelationsResponse>,
  })
  const correlations: CorrelationData[] = correlationData?.correlations || []

  const { data: highRiskData } = useQuery<HighRiskDepartmentsResponse>({
    queryKey: ['analytics', 'high-risk-departments'],
    queryFn: () => api.analytics.getHighRiskDepartments() as Promise<HighRiskDepartmentsResponse>,
  })
  const highRiskDepts: HighRiskDepartment[] = highRiskData?.departments || []
  const threshold = highRiskData?.threshold || 0


  if (deptLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-pulse-subtle text-text-secondary dark:text-text-dark-secondary">
          Analyzing organizational health...
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6 h-[calc(100vh-100px)] flex flex-col animate-in fade-in duration-700 slide-in-from-bottom-4">
      {/* Header with Tabs */}
      <div className="flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between flex-shrink-0">
        <div>
          <h1 className="text-4xl font-display font-bold text-gradient bg-clip-text text-transparent bg-gradient-to-r from-gray-900 to-gray-600 dark:from-white dark:to-gray-400">
            Workforce Health
          </h1>
          <p className="text-text-secondary dark:text-text-dark-secondary mt-2 text-lg font-light flex items-center gap-2">
            Real-time organizational pulse & diagnostics
          </p>
        </div>

        {/* Premium Tab Navigation */}
        <div className="glass p-1.5 rounded-2xl flex gap-1">
          <button
            onClick={() => setActiveTab('pulse')}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-medium transition-all duration-300 ${activeTab === 'pulse'
              ? 'bg-white dark:bg-slate-800 shadow-lg text-text-primary dark:text-white scale-105'
              : 'text-text-secondary dark:text-slate-400 hover:text-text-primary dark:hover:text-white hover:bg-white/10'
              }`}
          >
            <Activity className="w-4 h-4" />
            Pulse
          </button>
          <button
            onClick={() => setActiveTab('compensation')}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-medium transition-all duration-300 ${activeTab === 'compensation'
              ? 'bg-white dark:bg-slate-800 shadow-lg text-text-primary dark:text-white scale-105'
              : 'text-text-secondary dark:text-slate-400 hover:text-text-primary dark:hover:text-white hover:bg-white/10'
              }`}
          >
            <DollarSign className="w-4 h-4" />
            Comp
          </button>
          <button
            onClick={() => setActiveTab('succession')}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-medium transition-all duration-300 ${activeTab === 'succession'
              ? 'bg-white dark:bg-slate-800 shadow-lg text-text-primary dark:text-white scale-105'
              : 'text-text-secondary dark:text-slate-400 hover:text-text-primary dark:hover:text-white hover:bg-white/10'
              }`}
          >
            <Users className="w-4 h-4" />
            Succession
          </button>
          <button
            onClick={() => setActiveTab('insights')}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-medium transition-all duration-300 ${activeTab === 'insights'
              ? 'bg-white dark:bg-slate-800 shadow-lg text-text-primary dark:text-white scale-105'
              : 'text-text-secondary dark:text-slate-400 hover:text-text-primary dark:hover:text-white hover:bg-white/10'
              }`}
          >
            <MessageSquare className="w-4 h-4" />
            Insights
          </button>
          <button
            onClick={() => setActiveTab('segments')}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-medium transition-all duration-300 ${activeTab === 'segments'
              ? 'bg-white dark:bg-slate-800 shadow-lg text-text-primary dark:text-white scale-105'
              : 'text-text-secondary dark:text-slate-400 hover:text-text-primary dark:hover:text-white hover:bg-white/10'
              }`}
          >
            <Target className="w-4 h-4" />
            Personas
          </button>
        </div>
      </div>

      {/* Tab Content */}
      <div className="flex-1 min-h-0 overflow-y-auto pr-2 pb-4">
        {activeTab === 'pulse' && (
          <div className="space-y-6">
            {/* High Risk Alert */}
            {highRiskDepts && highRiskDepts.length > 0 && (
              <GlassCard className="border-l-4 border-l-red-500 relative overflow-hidden group">
                <div className="absolute inset-0 bg-red-500/5 group-hover:bg-red-500/10 transition-colors pointer-events-none" />
                <div className="flex items-start gap-4 relative z-10">
                  <div className="p-3 bg-red-500/20 rounded-full text-red-500">
                    <AlertTriangle className="w-6 h-6" />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-text-primary dark:text-white">
                      {highRiskDepts.length} Critical Areas Detected
                    </h3>
                    <p className="text-text-secondary dark:text-slate-300 mt-1">
                      Departments exceeding the {(threshold * 100).toFixed(0)}% turnover threshold require immediate attention.
                    </p>
                    <div className="flex flex-wrap gap-2 mt-4">
                      {highRiskDepts.map(d => (
                        <div key={d.dept} className="flex items-center gap-2 px-3 py-1 bg-red-500/10 border border-red-500/20 rounded-full text-xs font-semibold text-red-500 uppercase tracking-wide">
                          {d.dept}
                          <span className="w-1 h-1 rounded-full bg-red-500" />
                          {(d.turnover_rate * 100).toFixed(0)}%
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </GlassCard>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Turnover Chart */}
              <GlassCard className="col-span-2 h-[450px] flex flex-col">
                <h3 className="text-lg font-semibold mb-6 flex items-center gap-2">
                  <div className="w-1 h-6 bg-blue-500 rounded-full" />
                  Turnover by Department
                </h3>
                <div className="flex-1">
                  {departments && departments.length > 0 ? (
                    <DepartmentBarChart
                      data={departments.filter((d) => d.turnover_rate != null && d.turnover_rate !== undefined)}
                      dataKey="turnover_rate"
                    />
                  ) : (
                    <div className="h-full flex items-center justify-center text-text-muted">No data</div>
                  )}
                </div>
              </GlassCard>

              {/* Correlations */}
              <GlassCard className="col-span-1 h-[450px] flex flex-col">
                <h3 className="text-lg font-semibold mb-6 flex items-center gap-2">
                  <div className="w-1 h-6 bg-purple-500 rounded-full" />
                  Retention Drivers
                </h3>
                <div className="flex-1">
                  {correlations && correlations.length > 0 ? (
                    <CorrelationHeatmap data={correlations} />
                  ) : (
                    <div className="h-full flex items-center justify-center text-text-muted">No data</div>
                  )}
                </div>
              </GlassCard>
            </div>

            {/* Departments Table */}
            <GlassCard>
              <h3 className="text-lg font-semibold mb-6 flex items-center gap-2">
                <div className="w-1 h-6 bg-slate-500 rounded-full" />
                Department Analytics
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-white/10">
                      <th className="text-left py-4 px-4 text-xs font-bold text-text-muted uppercase tracking-wider">Department</th>
                      <th className="text-right py-4 px-4 text-xs font-bold text-text-muted uppercase tracking-wider">Headcount</th>
                      <th className="text-right py-4 px-4 text-xs font-bold text-text-muted uppercase tracking-wider">Avg Salary</th>
                      <th className="text-right py-4 px-4 text-xs font-bold text-text-muted uppercase tracking-wider">Avg Tenure</th>
                      <th className="text-right py-4 px-4 text-xs font-bold text-text-muted uppercase tracking-wider">Rating</th>
                      <th className="text-right py-4 px-4 text-xs font-bold text-text-muted uppercase tracking-wider">Risk</th>
                    </tr>
                  </thead>
                  <tbody>
                    {departments.map((dept) => (
                      <tr key={dept.dept} className="border-b border-white/5 hover:bg-white/5 transition-colors group">
                        <td className="py-3 px-4 font-medium text-text-primary dark:text-white group-hover:text-accent transition-colors">{dept.dept}</td>
                        <td className="py-3 px-4 text-right text-text-secondary">{dept.headcount}</td>
                        <td className="py-3 px-4 text-right text-text-secondary font-mono text-xs">
                          {dept.avg_salary ? `$${Math.round(dept.avg_salary).toLocaleString()}` : '-'}
                        </td>
                        <td className="py-3 px-4 text-right text-text-secondary">{dept.avg_tenure ? `${dept.avg_tenure.toFixed(1)}y` : '-'}</td>
                        <td className="py-3 px-4 text-right text-text-secondary">{dept.avg_rating ? dept.avg_rating.toFixed(2) : '-'}</td>
                        <td className="py-3 px-4 text-right">
                          {dept.turnover_rate !== undefined && dept.turnover_rate !== null ? (
                            <span className={cn(
                              "px-2 py-1 rounded-full text-xs font-bold",
                              dept.turnover_rate > 0.2 ? 'bg-red-500/10 text-red-500' :
                                dept.turnover_rate > 0.15 ? 'bg-amber-500/10 text-amber-500' :
                                  'bg-emerald-500/10 text-emerald-500'
                            )}>
                              {(dept.turnover_rate * 100).toFixed(1)}%
                            </span>
                          ) : '-'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </GlassCard>
            <div className="h-8"></div>
          </div>
        )}

        {activeTab === 'compensation' && (
          <GlassCard className="min-h-full">
            <CompensationTab />
          </GlassCard>
        )}

        {activeTab === 'succession' && (
          <GlassCard className="min-h-full">
            <SuccessionTab />
          </GlassCard>
        )}

        {activeTab === 'insights' && (
          <div className="space-y-6">
            <GlassCard>
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <Heart className="w-5 h-5 text-accent" />
                Team Dynamics
              </h3>
              <TeamDynamicsTab />
            </GlassCard>
            <GlassCard>
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <MessageSquare className="w-5 h-5 text-accent" />
                AI Sentiment Analysis
              </h3>
              <NLPTab />
            </GlassCard>
          </div>
        )}

        {activeTab === 'segments' && (
          <SegmentsTab />
        )}
      </div>
    </div>
  )
}
