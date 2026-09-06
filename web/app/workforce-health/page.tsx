'use client'

import { useQuery } from '@tanstack/react-query'
import Link from 'next/link'
import { api } from '@/lib/api-client'
import { EmptyState, MetricCard, Page, PageHeader, SectionHeader, StateSummary, StatusBadge, Surface } from '@/components/ui'
import { Activity, AlertTriangle, ArrowRight, HeartPulse, Target, Users } from 'lucide-react'
import type { CorrelationsResponse, DepartmentList, HighRiskDepartmentsResponse } from '@/types/api'

export default function WorkforceHealthPage() {
  const { data: departmentData, isLoading, isError, error } = useQuery<DepartmentList>({
    queryKey: ['analytics', 'departments'],
    queryFn: () => api.analytics.getDepartments() as Promise<DepartmentList>,
  })
  const { data: correlationData } = useQuery<CorrelationsResponse>({
    queryKey: ['analytics', 'correlations'],
    queryFn: () => api.analytics.getCorrelations(10) as Promise<CorrelationsResponse>,
  })
  const { data: riskData } = useQuery<HighRiskDepartmentsResponse>({
    queryKey: ['analytics', 'high-risk-departments'],
    queryFn: () => api.analytics.getHighRiskDepartments() as Promise<HighRiskDepartmentsResponse>,
  })

  const header = <PageHeader eyebrow="Understand · Workforce Health" title="Where is organisational pressure visible in the current workforce?" description="Current-state department evidence using active headcount, observed attrition share, tenure and ratings. Association is kept separate from causation." />
  if (isLoading) return <Page>{header}<StateSummary title="Reading workforce health" description="Comparing current department structure and observed workforce outcomes." tone="info" /></Page>
  if (isError) return <Page>{header}<EmptyState title="Workforce Health is unavailable" description={error instanceof Error ? error.message : 'Unable to load department analytics.'} /></Page>

  const departments = departmentData?.departments ?? []
  const highRisk = riskData?.departments ?? []
  const correlations = correlationData?.correlations ?? []
  const totalHeadcount = departments.reduce((sum, item) => sum + (item.headcount || 0), 0)
  const avgAttritionShare = departments.length ? departments.reduce((sum, item) => sum + (item.turnover_rate ?? 0), 0) / departments.length : 0

  return (
    <Page>
      {header}
      <StateSummary title="Metric boundary" description="Observed attrition share is the share of current employee records marked as departed in the available outcome data. It is not a period turnover rate unless a defined time window and at-risk denominator are available." tone="info" />

      {highRisk.length > 0 ? <StateSummary title={`${highRisk.length} department${highRisk.length === 1 ? '' : 's'} above the configured attrition-share threshold`} description="Use this only to prioritize aggregate investigation. Validate time period, local context and data completeness before intervention." tone="warning" /> : <StateSummary title="No department exceeds the configured observed attrition-share threshold" description="This does not prove the absence of retention risk; it only describes the current recorded outcome mix." tone="success" />}

      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="Active people represented" value={totalHeadcount.toLocaleString()} detail="Current active department aggregates" icon={Users} />
        <MetricCard label="Departments" value={departments.length.toLocaleString()} detail="Current analytical coverage" icon={Target} />
        <MetricCard label="Avg department attrition share" value={`${(avgAttritionShare * 100).toFixed(1)}%`} detail="Unweighted descriptive average across departments" icon={HeartPulse} tone={avgAttritionShare > .2 ? 'danger' : avgAttritionShare > .15 ? 'warning' : 'neutral'} />
        <MetricCard label="Priority aggregates" value={highRisk.length.toLocaleString()} detail={`Configured threshold ${((riskData?.threshold ?? 0) * 100).toFixed(0)}%`} icon={AlertTriangle} tone={highRisk.length ? 'warning' : 'neutral'} />
      </section>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.35fr)_minmax(320px,0.65fr)]">
        <Surface padding="lg">
          <SectionHeader title="Department current-state evidence" description="Compare active headcount, observed attrition share, tenure and rating without hiding population semantics." />
          <div className="mt-5 overflow-x-auto">
            <table className="w-full min-w-[720px] text-sm">
              <thead><tr className="border-b border-border text-left text-[11px] uppercase tracking-wider text-text-muted"><th className="py-3 pr-4">Department</th><th className="py-3 px-4 text-right">Active people</th><th className="py-3 px-4 text-right">Observed attrition share</th><th className="py-3 px-4 text-right">Avg tenure</th><th className="py-3 pl-4 text-right">Avg rating</th></tr></thead>
              <tbody>{departments.map((dept) => <tr key={dept.dept} className="border-b border-border last:border-0"><td className="py-3 pr-4 font-medium">{dept.dept}</td><td className="px-4 py-3 text-right">{dept.headcount}</td><td className="px-4 py-3 text-right"><StatusBadge tone={(dept.turnover_rate ?? 0) > .2 ? 'danger' : (dept.turnover_rate ?? 0) > .15 ? 'warning' : 'neutral'}>{dept.turnover_rate == null ? '—' : `${(dept.turnover_rate * 100).toFixed(1)}%`}</StatusBadge></td><td className="px-4 py-3 text-right">{dept.avg_tenure == null ? '—' : `${dept.avg_tenure.toFixed(1)}y`}</td><td className="py-3 pl-4 text-right">{dept.avg_rating == null ? '—' : dept.avg_rating.toFixed(2)}</td></tr>)}</tbody>
            </table>
          </div>
        </Surface>

        <Surface padding="lg">
          <SectionHeader title="Observed associations" description="Correlation with the recorded Attrition outcome. These are not causal drivers or intervention effects." />
          <div className="mt-5 space-y-3">{correlations.length ? correlations.slice(0, 8).map((item) => <div key={item.feature} className="flex items-center justify-between gap-4 border-b border-border py-3 last:border-0"><div><div className="font-medium">{item.feature}</div><div className="text-xs text-text-muted">Association with {correlationData?.target_column ?? 'recorded outcome'}</div></div><StatusBadge tone={Math.abs(item.correlation) >= .4 ? 'info' : 'neutral'}>r={item.correlation.toFixed(2)}</StatusBadge></div>) : <EmptyState title="No association data available" />}</div>
        </Surface>
      </div>

      <Surface padding="md" className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div><div className="font-semibold">Need another analytical lens?</div><div className="text-sm text-text-secondary">Use People Intelligence for governed compensation, structure, fairness or retention evidence. Individual ranking panels are not part of Workforce Health.</div></div>
        <Link href="/advisor" className="inline-flex items-center gap-2 text-sm font-semibold text-accent"><Activity className="h-4 w-4" />Investigate aggregate evidence <ArrowRight className="h-4 w-4" /></Link>
      </Surface>
    </Page>
  )
}
