'use client'

import { useQuery } from '@tanstack/react-query'
import Link from 'next/link'
import { api } from '@/lib/api-client'
import { EmptyState, MetricCard, Page, PageHeader, SectionHeader, StateSummary, StatusBadge, Surface, TrustDisclosure } from '@/components/ui'
import { Activity, AlertTriangle, ArrowRight, HeartPulse, Target, Users } from 'lucide-react'
import type { CorrelationsResponse, DepartmentList, HighRiskDepartmentsResponse } from '@/types/api'

type CorrelationItem = { feature: string; correlation: number; abs_correlation: number; p_value?: number; observations?: number }
type RiskResponse = HighRiskDepartmentsResponse & { evidence_available?: boolean; unavailable_reason?: string | null; outcome_observations?: number }
type SummaryResponse = { observed_attrition_share?: number | null; attrition_known_count?: number; active_count?: number }

function humanizeField(value: string) {
  return value.replaceAll('_', ' ').replace(/([a-z0-9])([A-Z])/g, '$1 $2').replace(/\b\w/g, char => char.toUpperCase())
}

function pValue(value?: number) {
  if (value == null || !Number.isFinite(value)) return null
  return value < 0.001 ? 'p<0.001' : `p=${value.toFixed(3)}`
}

function askHref(question: string) {
  return `/advisor?q=${encodeURIComponent(question)}`
}

export default function WorkforceHealthPage() {
  const { data: departmentData, isLoading, isError, error } = useQuery<DepartmentList>({ queryKey: ['analytics', 'departments'], queryFn: () => api.analytics.getDepartments() as Promise<DepartmentList> })
  const { data: summaryData } = useQuery<SummaryResponse>({ queryKey: ['analytics', 'summary'], queryFn: () => api.analytics.getSummary() as Promise<SummaryResponse> })
  const { data: correlationData, isError: correlationError } = useQuery<CorrelationsResponse>({ queryKey: ['analytics', 'correlations'], queryFn: () => api.analytics.getCorrelations(10) as Promise<CorrelationsResponse> })
  const { data: riskData, isError: riskError } = useQuery<RiskResponse>({ queryKey: ['analytics', 'high-risk-departments'], queryFn: () => api.analytics.getHighRiskDepartments() as Promise<RiskResponse> })

  const header = <PageHeader eyebrow="Insights · Workforce" title="What is happening across your workforce?" description="See the clearest current workforce patterns first, then open the calculation details only when you need them." />
  if (isLoading) return <Page>{header}<StateSummary title="Preparing workforce insights" description="Reading your current workforce and department patterns." tone="info" /></Page>
  if (isError) return <Page>{header}<EmptyState title="Workforce insights are unavailable" description={error instanceof Error ? error.message : 'PeopleOS could not read your workforce data.'} /></Page>

  const departments = departmentData?.departments ?? []
  const highRisk = riskData?.departments ?? []
  const correlations = (correlationData?.correlations ?? []) as CorrelationItem[]
  const totalHeadcount = departments.reduce((sum, item) => sum + (item.headcount || 0), 0)
  const overallAttrition = typeof summaryData?.observed_attrition_share === 'number' ? summaryData.observed_attrition_share : null
  const largest = [...departments].sort((a, b) => b.headcount - a.headcount)[0]

  return <Page>
    {header}

    {(correlationError || riskError) && <StateSummary title="Some supporting analysis is temporarily unavailable" description="Your core workforce measures are still available. Retry later for the missing supporting analysis." tone="warning" />}
    {!departments.length && <EmptyState title="No workforce insight is available yet" description="PeopleOS did not receive enough department information to build this view." />}

    <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <MetricCard label="Active people" value={departments.length ? totalHeadcount.toLocaleString() : 'Unavailable'} detail="Current active workforce represented" icon={Users} />
      <MetricCard label="Departments" value={departments.length ? departments.length.toLocaleString() : 'Unavailable'} detail={largest ? `${largest.dept} is the largest team` : 'Current structure'} icon={Target} />
      <MetricCard label="Recorded attrition" value={overallAttrition == null ? 'Unavailable' : `${(overallAttrition * 100).toFixed(1)}%`} detail={summaryData?.attrition_known_count ? `${summaryData.attrition_known_count.toLocaleString()} known employee outcomes` : 'Known employee outcomes'} icon={HeartPulse} tone={overallAttrition == null ? 'neutral' : overallAttrition > .2 ? 'danger' : overallAttrition > .15 ? 'warning' : 'neutral'} />
      <MetricCard label="Areas to review" value={riskData?.evidence_available === false ? 'Unavailable' : riskData ? highRisk.length.toLocaleString() : 'Unavailable'} detail={riskData?.evidence_available === false ? 'More recorded outcomes are needed' : highRisk.length ? 'Above your configured review threshold' : 'No department exceeds the current threshold'} icon={AlertTriangle} tone={highRisk.length ? 'warning' : 'neutral'} />
    </section>

    <Surface padding="md" className="border-violet-100/80 bg-violet-50/30 dark:border-violet-500/15 dark:bg-violet-500/[0.03]">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div><div className="text-sm font-semibold">Explore this workforce</div><div className="mt-1 text-xs text-text-secondary">Open another verified cut without building a report.</div></div>
        <div className="flex flex-wrap gap-2">
          <Link href={askHref('Headcount by department')} className="rounded-xl border border-border bg-surface px-3 py-2 text-xs font-semibold hover:border-violet-300 hover:text-violet-700 dark:hover:border-violet-500/40 dark:hover:text-violet-300">People by department</Link>
          <Link href={askHref('Headcount by location')} className="rounded-xl border border-border bg-surface px-3 py-2 text-xs font-semibold hover:border-violet-300 hover:text-violet-700 dark:hover:border-violet-500/40 dark:hover:text-violet-300">People by location</Link>
          <Link href={askHref('Recorded attrition share by department')} className="rounded-xl border border-border bg-surface px-3 py-2 text-xs font-semibold hover:border-violet-300 hover:text-violet-700 dark:hover:border-violet-500/40 dark:hover:text-violet-300">Attrition by department</Link>
          <Link href={askHref('Average salary by department')} className="rounded-xl border border-border bg-surface px-3 py-2 text-xs font-semibold hover:border-violet-300 hover:text-violet-700 dark:hover:border-violet-500/40 dark:hover:text-violet-300">Pay by department</Link>
        </div>
      </div>
    </Surface>

    {highRisk.length > 0 && <Surface padding="lg" className="border-amber-200/80 bg-amber-50/50 dark:border-amber-500/20 dark:bg-amber-500/[0.04]">
      <div className="text-xs font-bold uppercase tracking-[0.16em] text-amber-700 dark:text-amber-300">Worth a closer look</div>
      <div className="mt-3 grid gap-3 md:grid-cols-2 xl:grid-cols-3">{highRisk.slice(0, 6).map((dept) => <div key={dept.dept} className="rounded-2xl border border-amber-200/80 bg-white p-4 dark:border-amber-500/20 dark:bg-slate-950"><div className="font-semibold">{dept.dept}</div><div className="mt-1 text-2xl font-semibold">{(dept.turnover_rate * 100).toFixed(1)}%</div><div className="mt-1 text-xs text-slate-500 dark:text-slate-400">recorded attrition share</div></div>)}</div>
      <div className="mt-4 text-sm text-slate-600 dark:text-slate-400">These are investigation priorities, not conclusions about why people left.</div>
    </Surface>}

    <div className="grid gap-6 xl:grid-cols-[minmax(0,1.35fr)_minmax(320px,0.65fr)]">
      <Surface padding="lg">
        <SectionHeader title="Department picture" description="A clean comparison of the workforce measures available now." />
        <div className="mt-5 overflow-x-auto">
          <table className="w-full min-w-[640px] text-sm">
            <thead><tr className="border-b border-border text-left text-[11px] uppercase tracking-wider text-text-muted"><th className="py-3 pr-3">Department</th><th className="px-3 py-3 text-right">People</th><th className="px-3 py-3 text-right">Recorded attrition</th><th className="px-3 py-3 text-right">Avg tenure</th><th className="py-3 pl-3 text-right">Avg rating</th></tr></thead>
            <tbody>{departments.map((dept) => <tr key={dept.dept} className="border-b border-border last:border-0"><td className="py-3 pr-3 font-medium">{dept.dept}</td><td className="px-3 py-3 text-right">{dept.headcount}</td><td className="px-3 py-3 text-right"><StatusBadge tone={(dept.turnover_rate ?? 0) > .2 ? 'danger' : (dept.turnover_rate ?? 0) > .15 ? 'warning' : 'neutral'}>{dept.turnover_rate == null ? '—' : `${(dept.turnover_rate * 100).toFixed(1)}%`}</StatusBadge></td><td className="px-3 py-3 text-right">{dept.avg_tenure == null ? '—' : `${dept.avg_tenure.toFixed(1)}y`}</td><td className="py-3 pl-3 text-right">{dept.avg_rating == null ? '—' : dept.avg_rating.toFixed(2)}</td></tr>)}</tbody>
          </table>
        </div>
      </Surface>

      <Surface padding="lg">
        <SectionHeader title="Related patterns" description="Relationships in recorded outcomes that may be worth investigating further." />
        <div className="mt-5 space-y-2">{correlations.length ? correlations.slice(0, 8).map((item) => <div key={item.feature} className="rounded-xl border border-border px-4 py-3"><div className="flex items-center justify-between gap-4"><div className="font-medium">{humanizeField(item.feature)}</div><StatusBadge tone={Math.abs(item.correlation) >= .4 ? 'info' : 'neutral'}>r={item.correlation.toFixed(2)}</StatusBadge></div>{item.observations != null && <div className="mt-1 text-xs text-text-muted">{item.observations.toLocaleString()} paired observations{pValue(item.p_value) ? ` · ${pValue(item.p_value)}` : ''}</div>}</div>) : <EmptyState title="No reliable related patterns available" description="PeopleOS needs enough valid paired measurements before showing a relationship." />}</div>
      </Surface>
    </div>

    <TrustDisclosure title="How to read these numbers" summary="Calculated from your current dataset">
      <div className="space-y-2"><p><strong>Recorded attrition</strong> is the workforce-wide share of known employee outcomes marked as departed. It is not automatically an annual turnover rate.</p><p><strong>Department shares</strong> are shown separately so different team sizes do not change the overall workforce metric above.</p><p><strong>Related patterns</strong> are pairwise statistical associations. They do not prove that one factor caused another.</p><p>PeopleOS keeps small or unsupported results unavailable rather than turning missing evidence into zero.</p></div>
    </TrustDisclosure>

    <Surface padding="md" className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
      <div><div className="font-semibold">Want to understand one of these patterns?</div><div className="text-sm text-text-secondary">Ask PeopleOS to compare teams, examine another lens or explain what the available evidence supports.</div></div>
      <Link href="/advisor" className="inline-flex items-center gap-2 text-sm font-semibold text-accent"><Activity className="h-4 w-4" />Ask PeopleOS <ArrowRight className="h-4 w-4" /></Link>
    </Surface>
  </Page>
}
