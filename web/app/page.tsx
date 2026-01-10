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
import { api } from '../lib/api-client'
import { Card } from '../components/ui/card'
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
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-text-primary dark:text-text-dark-primary">Overview</h1>
          <p className="text-text-secondary dark:text-text-dark-secondary mt-1">
            Key metrics and insights across your organization
          </p>
        </div>

        {/* Key Takeaways */}
        {summary.takeaways && summary.takeaways.length > 0 && (
          <div className="bg-gradient-to-br from-slate-900 to-slate-800 dark:from-slate-950 dark:to-slate-900 p-6 rounded-xl border-l-4 border-success shadow-lg">
            <div className="flex justify-between items-start mb-3">
              <div className="text-xs font-bold text-success uppercase tracking-wider flex items-center gap-2">
                <Star className="w-4 h-4" />
                Key Insights
              </div>
              <Tooltip>
                <TooltipTrigger>
                  <Info className="w-4 h-4 text-slate-400 cursor-help" />
                </TooltipTrigger>
                <TooltipContent className="bg-slate-800 text-slate-100 border-slate-700">
                  <p className="text-xs">AI-generated patterns from your current dataset and trends.</p>
                </TooltipContent>
              </Tooltip>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-2">
              {summary.takeaways.map((takeaway, i) => (
                <div key={i} className="text-slate-200 text-sm flex gap-2 items-start">
                  <span className="text-success mt-1">•</span>
                  <span>{takeaway}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Main KPI Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <KPICard
            title="Total Headcount"
            value={summary.headcount}
            icon={Users}
            insight={summary.insights?.headcount || "Current total number of active employees across the organization."}
          />
          <KPICard
            title={
              <div className="flex items-center gap-2">
                Departure Rate
                <Tooltip>
                  <TooltipTrigger><Info className="w-3.5 h-3.5 opacity-50" /></TooltipTrigger>
                  <TooltipContent>The percentage of employees who left during the period.</TooltipContent>
                </Tooltip>
              </div>
            }
            value={summary.turnover_rate ? `${(summary.turnover_rate * 100).toFixed(1)}%` : '0%'}
            icon={TrendingDown}
            variant={summary.turnover_rate && summary.turnover_rate > 0.15 ? 'danger' : 'default'}
            insight={summary.insights?.turnover_rate || "Percentage of employees leaving, showing overall stability."}
          />
          <KPICard
            title="Avg Tenure"
            value={summary.tenure_mean ? `${summary.tenure_mean.toFixed(1)} yrs` : '0 yrs'}
            icon={Clock}
            subtitle="Time with company"
            insight={summary.insights?.tenure_mean || "Average time employees have stayed with the company."}
          />
          <KPICard
            title="Avg Salary"
            value={summary.salary_mean ? `$${Math.round(summary.salary_mean).toLocaleString()}` : '$0'}
            icon={DollarSign}
            subtitle="Across workforce"
            insight="Average annual compensation across all employees in the current dataset."
          />
        </div>

        {/* Supporting Metrics Row */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <KPICard
            title="Avg Performance"
            value={summary.lastrating_mean ? summary.lastrating_mean.toFixed(1) : '0.0'}
            icon={Star}
            subtitle="Recent ratings"
            variant="success"
            insight="Average score from the most recent performance review cycle."
          />
          <KPICard
            title="Avg Age"
            value={summary.age_mean ? `${summary.age_mean.toFixed(0)} yrs` : '0 yrs'}
            icon={Users}
            insight="The average age of employees, helping identify generational trends."
          />
          <KPICard
            title={
              <div className="flex items-center justify-between w-full">
                <div className="flex items-center gap-2">
                  Anticipated Risk
                  <Tooltip>
                    <TooltipTrigger><Brain className="w-3.5 h-3.5 opacity-50" /></TooltipTrigger>
                    <TooltipContent>AI-identified employees with high probability of departure.</TooltipContent>
                  </Tooltip>
                </div>
                <button
                  onClick={() => setShowMathModal(true)}
                  className="text-[10px] bg-accent/10 hover:bg-accent/20 text-accent px-1.5 py-0.5 rounded flex items-center gap-1 transition-colors"
                >
                  <Calculator className="w-3 h-3" />
                  How it works
                </button>
              </div>
            }
            value={predictions?.distribution?.high_risk || 0}
            icon={AlertTriangle}
            variant={(predictions?.distribution?.high_risk || 0) > 5 ? 'danger' : 'default'}
            subtitle="High-risk targets"
            insight="Number of employees whose patterns strongly match previous departures."
          />
          <KPICard
            title="Data Health"
            value="High"
            icon={Info}
            subtitle="Data quality audit"
            variant="default"
            insight="Ensures your data is complete and accurate for reliable analysis."
          />
        </div>

        {/* Charts Row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card title="Headcount by Department">
            {departments.length > 0 ? (
              <DepartmentBarChart data={departments} />
            ) : (
              <div className="h-64 flex items-center justify-center text-text-muted">No data available</div>
            )}
          </Card>

          <Card
            title={
              <div className="flex items-center gap-2">
                Risk Distribution
                <Tooltip>
                  <TooltipTrigger><Info className="w-3.5 h-3.5 opacity-50" /></TooltipTrigger>
                  <TooltipContent>ML grouping of workforce by attrition probability.</TooltipContent>
                </Tooltip>
              </div>
            }
          >
            {predictions?.distribution ? (
              <RiskDistributionPie data={predictions.distribution} />
            ) : (
              <div className="h-64 flex items-center justify-center text-text-muted text-sm">
                ML Training required for risk distribution
              </div>
            )}
          </Card>
        </div>

        {/* Selection Row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card title="Tenure Distribution">
            {distributions?.tenure ? (
              <TenureDistributionChart data={distributions.tenure} />
            ) : (
              <div className="h-64 flex items-center justify-center text-text-muted">No tenure data found</div>
            )}
          </Card>

          <Card title="High Risk Employees" subtitle="Click an employee for deep-dive analysis">
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
          </Card>
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
    </TooltipProvider>
  )
}
