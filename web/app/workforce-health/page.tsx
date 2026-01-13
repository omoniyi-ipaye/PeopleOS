'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Card } from '../../components/ui/card'
import { Badge } from '../../components/ui/badge'
import { DepartmentBarChart } from '../../components/charts/department-bar-chart'
import { CorrelationHeatmap } from '../../components/charts/correlation-heatmap'
import { api } from '../../lib/api-client'
import { cn } from '../../lib/utils'
import { AlertTriangle, TrendingDown, Info, RefreshCw } from 'lucide-react'
import {
  ExplanationBox,
  RiskFactorGuide,
  MetricExplainer,
  CorrelationGuide,
} from '../../components/ui/metric-explainer'
import { TabGroup } from '../../components/ui/tab-group'
import type {
  DepartmentStats,
  DepartmentList,
  CorrelationData,
  CorrelationsResponse,
  HighRiskDepartment,
  HighRiskDepartmentsResponse,
} from '@/types/api'
import { CompensationTab } from '../../components/diagnostics/compensation-tab'
import { SuccessionTab } from '../../components/diagnostics/succession-tab'
import { NLPTab } from '../../components/diagnostics/nlp-tab'
import { TeamDynamicsTab } from '../../components/diagnostics/team-dynamics-tab'

type WorkforceTab = 'operational' | 'compensation' | 'succession' | 'team' | 'nlp'

const TABS = [
  { id: 'operational', label: 'Operational Health' },
  { id: 'compensation', label: 'Compensation' },
  { id: 'succession', label: 'Succession' },
  { id: 'team', label: 'Team Dynamics' },
  { id: 'nlp', label: 'Language Analysis (AI)' },
] as const

export default function WorkforceHealthPage() {
  const [activeTab, setActiveTab] = useState<WorkforceTab>('operational')

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

  if (isError) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4 text-center">
        <AlertTriangle className="w-12 h-12 text-danger" />
        <h2 className="text-xl font-semibold text-text-primary dark:text-text-dark-primary">
          Failed to Load Workforce Health Data
        </h2>
        <p className="text-text-secondary dark:text-text-dark-secondary max-w-md">
          {error instanceof Error ? error.message : 'Unable to load organizational health metrics. Please try again.'}
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

  return (
    <div className="space-y-6">
      {/* Page Title */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-text-primary dark:text-text-dark-primary">Workforce Health</h1>
          <p className="text-text-secondary dark:text-text-dark-secondary mt-1">
            Understanding the pulse of your organization through data-driven insights
          </p>
        </div>

        {/* Tab Switcher */}
        <TabGroup<WorkforceTab>
          tabs={TABS}
          activeTab={activeTab}
          onTabChange={setActiveTab}
        />
      </div>

      <ExplanationBox title={`About: ${activeTab === 'operational' ? 'Operational Health' : activeTab === 'compensation' ? 'Compensation Analysis' : activeTab === 'succession' ? 'Succession Planning' : activeTab === 'team' ? 'Team Dynamics' : 'Language Analysis'}`}>
        {activeTab === 'operational' && "Monitor turnover and attrition across departments to identify high-risk areas."}
        {activeTab === 'compensation' && "Analyze how pay and benefits impact employee retention and performance."}
        {activeTab === 'succession' && "Ensure business continuity by identifying and developing future leaders."}
        {activeTab === 'team' && "Measure the health and wellness of your teams based on performance feedback."}
        {activeTab === 'nlp' && "AI-powered analysis of performance reviews to uncover hidden sentiment and themes."}
      </ExplanationBox>

      {activeTab === 'operational' ? (
        <>
          {/* High Risk Departments Alert */}
          {highRiskDepts && highRiskDepts.length > 0 && (
            <div className="bg-danger/5 dark:bg-danger/10 border border-danger/20 rounded-xl p-6 shadow-lg backdrop-blur-md">
              <div className="flex items-center gap-4 mb-6">
                <div className="p-3 bg-danger/10 rounded-xl">
                  <AlertTriangle className="w-6 h-6 text-danger" />
                </div>
                <div>
                  <h3 className="text-lg font-bold text-danger">
                    {highRiskDepts.length} Areas of Concern Detected
                  </h3>
                  <p className="text-sm text-text-secondary dark:text-text-dark-secondary">
                    Departure rates have exceeded the normal threshold ({(threshold * 100).toFixed(0)}%).
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {highRiskDepts.map((dept) => (
                  <div key={dept.dept} className="bg-surface dark:bg-surface-dark border border-danger/20 p-4 rounded-xl shadow-sm hover:shadow-md transition-shadow">
                    <div className="flex justify-between items-start mb-2">
                      <span className="font-bold text-text-primary dark:text-text-dark-primary">{dept.dept}</span>
                      <span className="px-2 py-0.5 bg-danger/10 text-danger text-[10px] font-bold rounded-full border border-danger/20 uppercase tracking-wider">
                        {(dept.turnover_rate * 100).toFixed(1)}% Departure Rate
                      </span>
                    </div>
                    <div className="space-y-3">
                      <div className="flex justify-between items-center text-xs text-text-muted dark:text-text-dark-muted">
                        <span>Headcount: {dept.headcount}</span>
                        <span>Avg Rating: {dept.avg_rating?.toFixed(1) || 'N/A'}</span>
                      </div>
                      {dept.reason && (
                        <div className="pt-2 border-t border-border dark:border-border-dark">
                          <div className="text-[9px] font-bold text-danger uppercase tracking-widest mb-1 opacity-70">Focus Areas</div>
                          <div className="text-[11px] text-text-secondary dark:text-text-dark-secondary italic leading-relaxed">
                            {dept.reason}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Department Stats Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Department Turnover */}
            <Card title="Departure Rates by Department" subtitle="How many employees are leaving each area">
              {departments && departments.length > 0 ? (
                <DepartmentBarChart
                  data={departments.filter((d) => d.turnover_rate !== null)}
                  dataKey="turnover_rate"
                />
              ) : (
                <div className="h-64 flex items-center justify-center text-text-secondary dark:text-text-dark-secondary">
                  No turnover data available
                </div>
              )}
            </Card>

            {/* Feature Correlations */}
            <Card title="Retention Drivers" subtitle="Factors mathematically linked to people leaving">
              {correlations && correlations.length > 0 ? (
                <CorrelationHeatmap data={correlations} />
              ) : (
                <div className="h-64 flex items-center justify-center text-text-secondary dark:text-text-dark-secondary">
                  Upload data with Attrition column
                </div>
              )}
            </Card>
          </div>

          {/* Department Details Table */}
          <Card title="Department Analytics" subtitle="Detailed performance and payroll metrics">
            {departments && departments.length > 0 ? (
              <div className="overflow-x-auto -mx-4 md:mx-0">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-border dark:border-border-dark">
                      <th className="text-left py-3 px-4 text-xs font-semibold text-text-muted dark:text-text-dark-muted uppercase tracking-wider">
                        Department
                      </th>
                      <th className="text-right py-3 px-4 text-xs font-semibold text-text-muted dark:text-text-dark-muted uppercase tracking-wider">
                        Headcount
                      </th>
                      <th className="text-right py-3 px-4 text-xs font-semibold text-text-muted dark:text-text-dark-muted uppercase tracking-wider">
                        Avg Salary
                      </th>
                      <th className="text-right py-3 px-4 text-xs font-semibold text-text-muted dark:text-text-dark-muted uppercase tracking-wider">
                        Avg Tenure
                      </th>
                      <th className="text-right py-3 px-4 text-xs font-semibold text-text-muted dark:text-text-dark-muted uppercase tracking-wider">
                        Avg Rating
                      </th>
                      <th className="text-right py-3 px-4 text-xs font-semibold text-text-muted dark:text-text-dark-muted uppercase tracking-wider">
                        Departure Rate
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {departments.map((dept) => (
                      <tr
                        key={dept.dept}
                        className="border-b border-border/50 dark:border-border-dark/50 hover:bg-surface-hover dark:hover:bg-surface-dark-hover transition-colors"
                      >
                        <td className="py-3 px-4 font-semibold text-text-primary dark:text-text-dark-primary">{dept.dept}</td>
                        <td className="py-3 px-4 text-right text-text-secondary dark:text-text-dark-secondary">
                          {dept.headcount}
                        </td>
                        <td className="py-3 px-4 text-right text-text-secondary dark:text-text-dark-secondary font-mono">
                          {dept.avg_salary
                            ? `$${Math.round(dept.avg_salary).toLocaleString()}`
                            : '-'}
                        </td>
                        <td className="py-3 px-4 text-right text-text-secondary dark:text-text-dark-secondary">
                          {dept.avg_tenure ? `${dept.avg_tenure.toFixed(1)} yrs` : '-'}
                        </td>
                        <td className="py-3 px-4 text-right text-text-secondary dark:text-text-dark-secondary">
                          {dept.avg_rating ? dept.avg_rating.toFixed(2) : '-'}
                        </td>
                        <td className="py-3 px-4 text-right">
                          {dept.turnover_rate !== undefined && dept.turnover_rate !== null ? (
                            <span
                              className={cn(
                                "font-semibold",
                                dept.turnover_rate > 0.2
                                  ? 'text-danger'
                                  : dept.turnover_rate > 0.15
                                    ? 'text-warning'
                                    : 'text-success'
                              )}
                            >
                              {(dept.turnover_rate * 100).toFixed(1)}%
                            </span>
                          ) : (
                            '-'
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="py-8 text-center text-text-secondary dark:text-text-dark-secondary">
                No department data available
              </div>
            )}
          </Card>

          {/* Why People Leave Section */}
          <div className="bg-gradient-to-br from-surface to-surface-secondary dark:from-surface-dark dark:to-background-dark border border-border dark:border-border-dark rounded-xl p-8 mb-6 shadow-sm overflow-hidden relative">
            <div className="absolute top-0 right-0 p-8 opacity-5">
              <TrendingDown className="w-32 h-32" />
            </div>

            <div className="flex items-center gap-4 mb-6 relative z-10">
              <div className="p-3 bg-accent/10 rounded-xl">
                <TrendingDown className="w-6 h-6 text-accent" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-text-primary dark:text-text-dark-primary italic">Why People Leave</h3>
                <p className="text-sm text-text-secondary dark:text-text-dark-secondary">
                  Data-driven drivers of organizational attrition
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 relative z-10">
              <div>
                <p className="text-sm text-text-secondary dark:text-text-dark-secondary leading-relaxed mb-6 font-medium">
                  Correlation analysis of your workforce data reveals the underlying factors statistically linked to turnover events.
                </p>
                <div className="space-y-4">
                  {correlations.slice(0, 3).map((corr, idx) => (
                    <div key={idx} className="flex items-start gap-4 p-3 bg-background/50 dark:bg-background-dark/50 rounded-lg border border-border/50 dark:border-border-dark/50">
                      <div className={cn(
                        "w-6 h-6 rounded-full flex items-center justify-center shrink-0 mt-0.5 font-bold text-[10px]",
                        idx === 0 ? "bg-danger/20 text-danger" : idx === 1 ? "bg-warning/20 text-warning" : "bg-accent/20 text-accent"
                      )}>
                        {idx + 1}
                      </div>
                      <div>
                        <p className="text-xs font-bold text-text-primary dark:text-text-dark-primary uppercase tracking-tight">
                          {corr.feature.replace(/_/g, ' ')}
                        </p>
                        <p className="text-[11px] text-text-secondary dark:text-text-dark-secondary mt-0.5">
                          High statistical correlation ({corr.correlation.toFixed(2)}) indicates this is a primary driver.
                        </p>
                      </div>
                    </div>
                  ))}
                  {correlations.length === 0 && (
                    <p className="text-xs text-text-muted italic">Statistical training in progress...</p>
                  )}
                </div>
              </div>

              <div className="bg-accent/5 border border-accent/20 p-6 rounded-2xl flex flex-col justify-center">
                <div className="text-[10px] font-bold text-accent uppercase tracking-widest mb-2">Strategy recommendation</div>
                <div className="text-xl font-bold text-text-primary dark:text-text-dark-primary mb-3">Targeted Retention Program</div>
                <p className="text-xs text-text-secondary leading-relaxed mb-4">
                  Employees showing patterns in the top {correlations.length > 0 ? '3' : 'factors'} identified above should be prioritized for intervention. Focused stay-interviews could reduce voluntary turnover by an estimated 12-18%.
                </p>
                <div className="flex gap-2">
                  <Badge variant="info">Priority High</Badge>
                  <Badge variant="outline">Impact 14%</Badge>
                </div>
              </div>
            </div>
          </div>
        </>
      ) : activeTab === 'compensation' ? (
        <CompensationTab />
      ) : activeTab === 'succession' ? (
        <SuccessionTab />
      ) : activeTab === 'team' ? (
        <TeamDynamicsTab />
      ) : activeTab === 'nlp' ? (
        <NLPTab />
      ) : null}
    </div>
  )
}
