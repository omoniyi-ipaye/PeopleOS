'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api-client'
import { Button, EmptyState, MetricCard, Page, PageHeader, SectionHeader, StateSummary, StatusBadge, Surface } from '@/components/ui'
import { Activity, AlertTriangle, DollarSign, HeartPulse, MessageSquare, Target, Users } from 'lucide-react'
import type { DepartmentList, CorrelationsResponse, HighRiskDepartmentsResponse } from '@/types/api'
import { CompensationTab } from '@/components/diagnostics/compensation-tab'
import { SuccessionTab } from '@/components/diagnostics/succession-tab'
import { NLPTab } from '@/components/diagnostics/nlp-tab'

type WorkforceTab = 'pulse' | 'compensation' | 'succession' | 'insights'

export default function WorkforceHealthPage() {
  const [tab, setTab] = useState<WorkforceTab>('pulse')
  const { data: departmentData, isLoading, isError, error, refetch } = useQuery<DepartmentList>({ queryKey: ['analytics', 'departments'], queryFn: () => api.analytics.getDepartments() as Promise<DepartmentList> })
  const { data: correlationData } = useQuery<CorrelationsResponse>({ queryKey: ['analytics', 'correlations'], queryFn: () => api.analytics.getCorrelations(10) as Promise<CorrelationsResponse> })
  const { data: riskData } = useQuery<HighRiskDepartmentsResponse>({ queryKey: ['analytics', 'high-risk-departments'], queryFn: () => api.analytics.getHighRiskDepartments() as Promise<HighRiskDepartmentsResponse> })

  if (isLoading) return <Page><StateSummary title="Reading workforce health" description="Comparing department structure, turnover and observed workforce drivers." tone="info" /></Page>
  if (isError) return <Page><EmptyState title="Workforce Health is unavailable" description={error instanceof Error ? error.message : 'Unable to load department analytics.'} action={<Button onClick={() => refetch()}>Retry</Button>} /></Page>

  const departments = departmentData?.departments ?? []
  const highRisk = riskData?.departments ?? []
  const correlations = correlationData?.correlations ?? []
  const totalHeadcount = departments.reduce((sum, item) => sum + (item.headcount || 0), 0)
  const avgTurnover = departments.length ? departments.reduce((sum, item) => sum + (item.turnover_rate ?? 0), 0) / departments.length : 0

  const tabs: Array<{ id: WorkforceTab; label: string; icon: typeof Activity }> = [
    { id: 'pulse', label: 'Workforce pulse', icon: Activity },
    { id: 'compensation', label: 'Compensation', icon: DollarSign },
    { id: 'succession', label: 'Succession', icon: Users },
    { id: 'insights', label: 'Narrative evidence', icon: MessageSquare },
  ]

  return (
    <Page>
      <PageHeader eyebrow="Understand · Workforce Health" title="Where is organisational pressure building?" description="A portfolio view of turnover, department health, workforce drivers and supporting governed lenses." />

      <div className="flex flex-wrap gap-2">{tabs.map(({ id, label, icon: Icon }) => <Button key={id} variant={tab === id ? 'primary' : 'secondary'} size="sm" onClick={() => setTab(id)}><Icon className="h-4 w-4" />{label}</Button>)}</div>

      {tab === 'pulse' ? (
        <>
          {highRisk.length > 0 ? <StateSummary title={`${highRisk.length} department${highRisk.length === 1 ? '' : 's'} above the configured turnover threshold`} description="Treat this as a prioritisation signal. Validate local context before deciding an intervention." tone="warning" /> : <StateSummary title="No department currently exceeds the configured turnover threshold" description="Continue monitoring trend direction and local context rather than assuming risk is absent." tone="success" />}

          <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <MetricCard label="People represented" value={totalHeadcount.toLocaleString()} detail="Across department aggregates" icon={Users} />
            <MetricCard label="Departments" value={departments.length.toLocaleString()} detail="Current analytical coverage" icon={Target} />
            <MetricCard label="Average department turnover" value={`${(avgTurnover * 100).toFixed(1)}%`} detail="Unweighted department average" icon={HeartPulse} tone={avgTurnover > .2 ? 'danger' : avgTurnover > .15 ? 'warning' : 'success'} />
            <MetricCard label="Priority areas" value={highRisk.length.toLocaleString()} detail={`Threshold ${((riskData?.threshold ?? 0) * 100).toFixed(0)}%`} icon={AlertTriangle} tone={highRisk.length ? 'warning' : 'success'} />
          </section>

          <div className="grid gap-6 xl:grid-cols-[minmax(0,1.35fr)_minmax(320px,0.65fr)]">
            <Surface padding="lg">
              <SectionHeader title="Department health" description="Compare turnover, headcount, tenure and performance without hiding the denominator." />
              <div className="mt-5 overflow-x-auto">
                <table className="w-full min-w-[720px] text-sm">
                  <thead><tr className="border-b border-border text-left text-[11px] uppercase tracking-wider text-text-muted"><th className="py-3 pr-4">Department</th><th className="py-3 px-4 text-right">People</th><th className="py-3 px-4 text-right">Turnover</th><th className="py-3 px-4 text-right">Tenure</th><th className="py-3 pl-4 text-right">Rating</th></tr></thead>
                  <tbody>{departments.map((dept) => <tr key={dept.dept} className="border-b border-border last:border-0"><td className="py-3 pr-4 font-medium">{dept.dept}</td><td className="px-4 py-3 text-right">{dept.headcount}</td><td className="px-4 py-3 text-right"><StatusBadge tone={(dept.turnover_rate ?? 0) > .2 ? 'danger' : (dept.turnover_rate ?? 0) > .15 ? 'warning' : 'success'}>{dept.turnover_rate == null ? '—' : `${(dept.turnover_rate * 100).toFixed(1)}%`}</StatusBadge></td><td className="px-4 py-3 text-right">{dept.avg_tenure == null ? '—' : `${dept.avg_tenure.toFixed(1)}y`}</td><td className="py-3 pl-4 text-right">{dept.avg_rating == null ? '—' : dept.avg_rating.toFixed(2)}</td></tr>)}</tbody>
                </table>
              </div>
            </Surface>

            <Surface padding="lg">
              <SectionHeader title="Observed drivers" description="Strongest correlations available in the current dataset." />
              <div className="mt-5 space-y-3">{correlations.length ? correlations.slice(0, 8).map((item) => <div key={item.feature} className="flex items-center justify-between gap-4 border-b border-border py-3 last:border-0"><div><div className="font-medium">{item.feature}</div><div className="text-xs text-text-muted">Association with {correlationData?.target_column ?? 'target'}</div></div><StatusBadge tone={Math.abs(item.correlation) >= .4 ? 'info' : 'neutral'}>{item.correlation.toFixed(2)}</StatusBadge></div>) : <EmptyState title="No driver data available" />}</div>
            </Surface>
          </div>
        </>
      ) : (
        <Surface padding="lg">
          <SectionHeader title={tabs.find((item) => item.id === tab)?.label ?? 'Diagnostic view'} description="This diagnostic uses the same governed dataset and should be read as supporting evidence, not an isolated decision engine." />
          <div className="mt-6">
            {tab === 'compensation' && <CompensationTab />}
            {tab === 'succession' && <SuccessionTab />}
            {tab === 'insights' && <NLPTab />}
          </div>
        </Surface>
      )}
    </Page>
  )
}
