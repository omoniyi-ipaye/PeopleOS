'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { KPICard } from '../components/dashboard/kpi-card'
import { DepartmentBarChart } from '../components/charts/department-bar-chart'
import { RiskDistributionPie } from '../components/charts/risk-distribution-pie'
import { TenureDistributionChart } from '../components/charts/tenure-distribution-chart'
import { HighRiskTable } from '../components/dashboard/high-risk-table'
import { EmployeeDetailModal } from '../components/dashboard/employee-detail-modal'
import { PredictiveMathModal } from '../components/dashboard/predictive-math-modal'
import { WorldMap } from '../components/dashboard/world-map'
import { api } from '../lib/api-client'
import { GlassCard } from '../components/ui/glass-card'
import { BentoGrid, BentoGridItem } from '../components/ui/bento-grid'
import { AnimatedNumber } from '../components/ui/animated-number'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '../components/ui/tooltip'
import {
  Users,
  TrendingDown,
  Clock,
  DollarSign,
  AlertTriangle,
  Info,
  Star,
  Brain,
  Calculator,
  ShieldCheck,
  Globe,
  LayoutDashboard,
} from 'lucide-react'
import {
  AnalyticsSummary,
  DepartmentList,
  PredictionSummary,
  DistributionsResponse,
} from '@/types/api'

export default function DashboardPage() {
  const [selectedEmployeeId, setSelectedEmployeeId] = useState<string | null>(null)
  const [showMathModal, setShowMathModal] = useState(false)
  const [activeTab, setActiveTab] = useState<'dashboard' | 'worldmap'>('dashboard')

  const { data: summary, isLoading: summaryLoading } = useQuery<AnalyticsSummary>({
    queryKey: ['analytics', 'summary'],
    queryFn: api.analytics.getSummary as any,
  })

  const { data: departmentData } = useQuery<DepartmentList>({
    queryKey: ['analytics', 'departments'],
    queryFn: api.analytics.getDepartments as any,
  })
  const departments = departmentData?.departments || []

  const { data: predictions } = useQuery<PredictionSummary>({
    queryKey: ['predictions', 'risk'],
    queryFn: () => api.predictions.getRisk() as any,
  })

  const { data: distributions } = useQuery<DistributionsResponse>({
    queryKey: ['analytics', 'distributions'],
    queryFn: api.analytics.getDistributions as any,
  })

  const { data: highRiskEmployees } = useQuery<{ employees: any[] }>({
    queryKey: ['predictions', 'high-risk'],
    queryFn: () => api.predictions.getHighRisk() as any,
  })

  if (summaryLoading) {
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] gap-6">
        <div className="relative">
          <div className="absolute inset-0 bg-accent/20 blur-xl rounded-full animate-pulse" />
          <div className="relative bg-slate-900/50 border border-slate-800 p-4 rounded-2xl backdrop-blur-sm">
            <Brain className="w-10 h-10 text-accent animate-pulse" />
          </div>
        </div>
        <div className="flex flex-col items-center gap-2">
          <h3 className="text-lg font-medium text-text-primary dark:text-text-dark-primary">
            Analyzing Workforce Intelligence
          </h3>
          <div className="flex items-center gap-1.5">
            <div className="h-1 w-1 bg-accent rounded-full animate-bounce [animation-delay:-0.3s]" />
            <div className="h-1 w-1 bg-accent rounded-full animate-bounce [animation-delay:-0.15s]" />
            <div className="h-1 w-1 bg-accent rounded-full animate-bounce" />
          </div>
          <p className="text-xs text-text-muted dark:text-text-dark-muted font-medium uppercase tracking-[0.2em] mt-2">
            PeopleOS Strategic Engine
          </p>
        </div>
      </div>
    )
  }

  if (!summary) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4">
        <AlertTriangle className="w-12 h-12 text-warning" />
        <h2 className="text-xl font-semibold text-text-primary dark:text-text-dark-primary">No Data Loaded</h2>
        <p className="text-text-secondary dark:text-text-dark-secondary text-center">
          Upload a CSV file or load sample data to get started.
        </p>
      </div>
    )
  }

  return (
    <TooltipProvider>
      <div className="space-y-8 animate-in fade-in duration-700 slide-in-from-bottom-4">
        {/* Header with Tabs */}
        <div className="flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h1 className="text-4xl font-display font-bold text-gradient bg-clip-text text-transparent bg-gradient-to-r from-gray-900 to-gray-600 dark:from-white dark:to-gray-400">
              Overview
            </h1>
            <p className="text-text-secondary dark:text-text-dark-secondary mt-2 text-lg font-light">
              Workforce intelligence at a glance
            </p>
          </div>

          {/* Premium Tab Navigation */}
          <div className="glass p-1.5 rounded-2xl flex gap-1">
            <button
              onClick={() => setActiveTab('dashboard')}
              className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-medium transition-all duration-300 ${activeTab === 'dashboard'
                ? 'bg-white dark:bg-slate-800 shadow-lg text-text-primary dark:text-white scale-105'
                : 'text-text-secondary dark:text-slate-400 hover:text-text-primary dark:hover:text-white hover:bg-white/10'
                }`}
            >
              <LayoutDashboard className="w-4 h-4" />
              Dashboard
            </button>
            <button
              onClick={() => setActiveTab('worldmap')}
              className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-medium transition-all duration-300 ${activeTab === 'worldmap'
                ? 'bg-white dark:bg-slate-800 shadow-lg text-text-primary dark:text-white scale-105'
                : 'text-text-secondary dark:text-slate-400 hover:text-text-primary dark:hover:text-white hover:bg-white/10'
                }`}
            >
              <Globe className="w-4 h-4" />
              World Map
            </button>
          </div>
        </div>

        {/* Tab Content */}
        {activeTab === 'worldmap' ? (
          <div className="glass-card p-6 min-h-[600px] flex flex-col">
            <WorldMap />
          </div>
        ) : (
          <div className="space-y-8">
            {/* Key Takeaways - Glass Panel */}
            {summary.takeaways && summary.takeaways.length > 0 && (
              <div className="glass p-6 rounded-2xl border-l-4 border-accent shadow-shine">
                <div className="flex justify-between items-start mb-4">
                  <div className="text-xs font-bold text-accent uppercase tracking-widest flex items-center gap-2">
                    <Brain className="w-4 h-4 animate-pulse" />
                    Strategic Insights
                  </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-4">
                  {summary.takeaways.map((takeaway, i) => (
                    <div key={i} className="text-text-secondary dark:text-slate-200 text-sm flex gap-3 items-start group">
                      <span className="text-accent mt-1.5 h-1.5 w-1.5 rounded-full bg-accent group-hover:scale-150 transition-transform" />
                      <span className="leading-relaxed">{takeaway}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Bento Grid Layout for Metrics */}
            <BentoGrid>
              {/* Headcount - Large Card */}
              <BentoGridItem
                title="Active Headcount"
                header={<AnimatedNumber value={summary.active_count || summary.headcount} className="text-5xl font-display font-bold text-text-primary dark:text-white" />}
                icon={<Users className="w-6 h-6 text-accent" />}
                description={`Currently active workforce (${summary.headcount} total records including attrition)`}
                className="md:col-span-1 border-l-4 border-accent"
              />

              {/* Turnover - Critical Metric */}
              <BentoGridItem
                title="Departure Rate"
                header={
                  <div className="flex items-baseline gap-2">
                    <span className="text-5xl font-display font-bold text-text-primary dark:text-white">
                      {summary.turnover_rate ? (summary.turnover_rate * 100).toFixed(1) : '0'}
                    </span>
                    <span className="text-xl text-text-muted">%</span>
                  </div>
                }
                icon={<TrendingDown className={summary.turnover_rate && summary.turnover_rate > 0.15 ? "w-6 h-6 text-red-500" : "w-6 h-6 text-green-500"} />}
                description={summary.insights?.turnover_rate || "Annualized turnover"}
                className={summary.turnover_rate && summary.turnover_rate > 0.15 ? "border-l-4 border-red-500 bg-red-50/10" : "border-l-4 border-green-500"}
              />

              {/* Risk Prediction - AI Feature */}
              <BentoGridItem
                title="At-Risk Employees"
                header={
                  <div className="flex items-baseline gap-2">
                    <AnimatedNumber value={predictions?.distribution?.high_risk || 0} className="text-5xl font-display font-bold text-text-primary dark:text-white" />
                    <span className="text-sm font-medium px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-500 border border-amber-500/20">
                      AI Forecast
                    </span>
                  </div>
                }
                icon={<AlertTriangle className="w-6 h-6 text-amber-500" />}
                description="High probability of departure"
                className="md:col-span-1 border-l-4 border-amber-500"
              />

              {/* Secondary Metrics Row */}
              <BentoGridItem
                title="Avg Tenure"
                header={<div className="text-3xl font-display font-bold">{summary.tenure_mean ? summary.tenure_mean.toFixed(1) : '0'} <span className="text-sm text-muted-foreground font-normal">years</span></div>}
                icon={<Clock className="w-5 h-5 text-purple-500" />}
                description="Average loyalty"
                className="md:col-span-1"
              />

              <BentoGridItem
                title="Avg Salary"
                header={<div className="text-3xl font-display font-bold">${summary.salary_mean ? Math.round(summary.salary_mean / 1000) : '0'}k</div>}
                icon={<DollarSign className="w-5 h-5 text-emerald-500" />}
                description="Mean compensation"
                className="md:col-span-1"
              />

              <BentoGridItem
                title="Performance"
                header={<div className="text-3xl font-display font-bold">{summary.lastrating_mean ? summary.lastrating_mean.toFixed(1) : '0.0'} <span className="text-sm text-muted-foreground font-normal">/ 5.0</span></div>}
                icon={<Star className="w-5 h-5 text-yellow-500" />}
                description="Average rating"
                className="md:col-span-1"
              />
            </BentoGrid>

            {/* Charts Section - Two Column Glass Layout */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <GlassCard className="min-h-[400px]">
                <h3 className="text-lg font-semibold mb-6 flex items-center gap-2">
                  <div className="w-1 h-6 bg-blue-500 rounded-full" />
                  Headcount by Department
                </h3>
                {departments.length > 0 ? (
                  <DepartmentBarChart data={departments} />
                ) : (
                  <div className="h-64 flex items-center justify-center text-text-muted">No data available</div>
                )}
              </GlassCard>

              <GlassCard className="min-h-[400px]">
                <h3 className="text-lg font-semibold mb-6 flex items-center gap-2">
                  <div className="w-1 h-6 bg-indigo-500 rounded-full" />
                  Risk Distribution (AI)
                </h3>
                {predictions?.distribution ? (
                  <RiskDistributionPie data={predictions.distribution} />
                ) : (
                  <div className="h-64 flex items-center justify-center text-text-muted text-sm">
                    ML Training required for risk distribution
                  </div>
                )}
              </GlassCard>
            </div>

            {/* Deep Dives */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              <GlassCard className="lg:col-span-1 min-h-[350px]">
                <h3 className="text-lg font-semibold mb-6 flex items-center gap-2">
                  <div className="w-1 h-6 bg-teal-500 rounded-full" />
                  Tenure Profile
                </h3>
                {distributions?.tenure ? (
                  <TenureDistributionChart data={distributions.tenure} />
                ) : (
                  <div className="h-64 flex items-center justify-center text-text-muted">No tenure data</div>
                )}
              </GlassCard>

              <GlassCard className="lg:col-span-2 min-h-[350px]">
                <h3 className="text-lg font-semibold mb-6 flex items-center gap-2">
                  <div className="w-1 h-6 bg-rose-500 rounded-full" />
                  High Risk Employees
                </h3>
                {highRiskEmployees?.employees ? (
                  <HighRiskTable
                    employees={highRiskEmployees.employees}
                    onEmployeeClick={(id) => setSelectedEmployeeId(id)}
                  />
                ) : (
                  <div className="h-64 flex items-center justify-center text-text-muted text-sm px-12 text-center">
                    Predictive analysis ready once data with Attrition is provided
                  </div>
                )}
              </GlassCard>
            </div>

            {/* Modals */}
            {selectedEmployeeId && (
              <EmployeeDetailModal
                employeeId={selectedEmployeeId}
                onClose={() => setSelectedEmployeeId(null)}
              />
            )}

            {showMathModal && (
              <PredictiveMathModal onClose={() => setShowMathModal(false)} />
            )}
          </div>
        )}
      </div>
    </TooltipProvider>
  )
}
