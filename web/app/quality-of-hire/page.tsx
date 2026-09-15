'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import Link from 'next/link'
import { api } from '@/lib/api-client'
import { Button, EmptyState, MetricCard, Page, PageHeader, SectionHeader, StateSummary, StatusBadge, Surface, TrustDisclosure } from '@/components/ui'
import { RelationshipInsight } from '@/components/relationship-insight'
import { Award, ArrowRight, BarChart3, Target, UserPlus } from 'lucide-react'
import type { PreHireCorrelation, QualityOfHireAnalysisResult, SourceEffectiveness } from '@/types/api'

type Tab = 'overview' | 'sources' | 'associations'

function humanize(value?: string | null) {
  if (!value) return 'Not available'
  return value.replaceAll('_', ' ').replace(/([a-z0-9])([A-Z])/g, '$1 $2').replace(/\b\w/g, char => char.toUpperCase())
}

function followUpLabel(value?: string | null) {
  const normalized = value?.replaceAll('_', ' ')
  if (normalized === 'duration qualified observed') return '12-month follow-up available'
  if (normalized === 'observed') return 'Recorded outcome available'
  return normalized ? humanize(normalized) : 'Follow-up outcome unavailable'
}

function roleMixLabel(value: string) {
  const labels: Record<string, string> = { Dept: 'department', Department: 'department', JobLevel: 'job level', JobTitle: 'job title' }
  return labels[value] ?? humanize(value).toLowerCase()
}

function outcomeLabel(value?: string | null) {
  if (value === 'LastRating') return 'later performance rating'
  if (value === 'Attrition') return 'recorded departures'
  return value ? humanize(value).toLowerCase() : 'the post-hire outcome'
}

function askHref(question: string) {
  return `/advisor?q=${encodeURIComponent(question)}`
}

function recommendationQuestion(recommendation: string, index: number) {
  const value = recommendation.toLowerCase()
  if (/department|team|function/.test(value)) return 'Headcount by department'
  if (/location|office|country|city/.test(value)) return 'Headcount by location'
  if (/role|job|level/.test(value)) return 'Headcount by role'
  if (/attrition|departure|retention|turnover/.test(value)) return 'Recorded attrition share by department'
  if (/pay|salary|compensation/.test(value)) return 'Average salary by department'
  return index % 2 === 0 ? 'Headcount by department' : 'Headcount by role'
}

export default function QualityOfHirePage() {
  const [tab, setTab] = useState<Tab>('overview')
  const [showAllAssociations, setShowAllAssociations] = useState(false)
  const { data, isLoading, isError, error, refetch } = useQuery<QualityOfHireAnalysisResult>({
    queryKey: ['quality-of-hire', 'analysis'],
    queryFn: () => api.qualityOfHire.getAnalysis() as Promise<QualityOfHireAnalysisResult>,
  })

  const header = <PageHeader eyebrow="Insights · Hiring" title="What can we learn from our hiring data?" description="Compare where people were hired from and how they performed after joining, when enough follow-up data is available. These are observations to investigate, not a score of hiring effectiveness." actions={<Link href="/advisor" className="inline-flex shrink-0 items-center justify-center gap-2 rounded-xl bg-violet-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-violet-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500">Ask PeopleOS <ArrowRight className="h-4 w-4" /></Link>} />
  if (isLoading) return <Page>{header}<StateSummary title="Preparing hiring insights" description="Reading hiring sources and recorded outcomes after people join." tone="info" /></Page>
  if (isError) return <Page>{header}<EmptyState title="Hiring insights are temporarily unavailable" description={error instanceof Error ? error.message : 'PeopleOS could not complete the hiring analysis.'} action={<Button onClick={() => refetch()}>Retry analysis</Button>} tone="warning" /></Page>

  const summary = data?.summary
  const sources: SourceEffectiveness[] = data?.source_effectiveness ?? []
  const associations = data?.correlations?.correlations ?? []
  const warnings = data?.warnings ?? []
  const recommendations = data?.recommendations ?? []
  const visibleAssociations = showAllAssociations ? associations : associations.slice(0, 4)
  const count = (value?: number) => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value.toLocaleString() : 'Unavailable'

  return (
    <Page>
      {header}

      <div className="flex flex-wrap gap-2" role="tablist" aria-label="Quality of hire views">
        <Button role="tab" aria-selected={tab === 'overview'} aria-controls="quality-overview" size="sm" variant={tab === 'overview' ? 'primary' : 'secondary'} onClick={() => setTab('overview')}><Award className="h-4 w-4" />Overview</Button>
        <Button role="tab" aria-selected={tab === 'sources'} aria-controls="quality-sources" size="sm" variant={tab === 'sources' ? 'primary' : 'secondary'} onClick={() => setTab('sources')}><Target className="h-4 w-4" />Hiring sources</Button>
        <Button role="tab" aria-selected={tab === 'associations'} aria-controls="quality-associations" size="sm" variant={tab === 'associations' ? 'primary' : 'secondary'} onClick={() => setTab('associations')}><BarChart3 className="h-4 w-4" />Pre-hire signals</Button>
      </div>

      {tab === 'overview' && <div id="quality-overview" role="tabpanel" className="contents">
        <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          <MetricCard label="People represented" value={count(summary?.total_hires ?? summary?.total_employees)} detail={`${count(summary?.performance_observations)} follow-up ratings recorded`} icon={UserPlus} />
          <MetricCard label="Hiring sources" value={count(summary?.sources_analyzed)} detail="Recruiting channels with enough people to compare" icon={Target} />
          <MetricCard label="Before-hire measures" value={count(summary?.prehire_signals_count)} detail="Candidate-stage information available to compare" icon={BarChart3} />
        </section>

        <div className="grid gap-6 lg:grid-cols-2">
          <Surface padding="lg">
            <SectionHeader title="Highest observed source result" description="The source group with the highest available combined measure. This is not a hiring-effectiveness score." />
            <div className="mt-5"><div className="text-3xl font-semibold">{summary?.best_source ?? 'Not available'}</div><p className="mt-2 text-sm leading-6 text-text-secondary">Treat this as a place to investigate—not a ranking of recruiting sources. Role mix, sample size and tenure exposure can all affect the comparison.</p></div>
          </Surface>
          <Surface padding="lg">
            <SectionHeader title="Strongest observed connection" description="The before-hire measure most closely associated with the selected later outcome." />
            <div className="mt-5"><div className="text-3xl font-semibold">{humanize(summary?.top_predictor)}</div><p className="mt-2 text-sm leading-6 text-text-secondary">Use this to form a hypothesis for validation. It does not show that the pre-hire measure caused the later outcome.</p></div>
          </Surface>
        </div>

        {recommendations.length ? <Surface padding="lg"><SectionHeader title="Useful next checks" description="These suggestions now open a concrete, verified starting check in Ask PeopleOS." /><div className="mt-5 grid gap-3 md:grid-cols-2">{recommendations.slice(0, 4).map((item, index) => <div key={`${item}-${index}`} className="rounded-xl border border-border p-4 text-sm leading-6 text-text-secondary"><div><span className="mr-2 font-semibold text-accent">0{index + 1}</span>{item}</div><Link href={askHref(recommendationQuestion(item, index))} className="mt-3 inline-flex items-center gap-1.5 text-xs font-semibold text-accent">Open a starting check <ArrowRight className="h-3.5 w-3.5" aria-hidden="true" /></Link></div>)}</div></Surface> : null}
      </div>}

      {tab === 'sources' && <Surface id="quality-sources" role="tabpanel" padding="lg">
        <SectionHeader title="Hiring source groups" description="See how many people came from each source, what follow-up is available and how the groups compare. The combined score is descriptive only." />
        <div className="mt-5 overflow-x-auto">
          <table className="w-full min-w-[1040px] text-sm"><thead><tr className="border-b border-border text-[11px] uppercase tracking-wider text-text-muted"><th className="py-3 pr-4 text-left">Source</th><th className="px-4 py-3 text-right">People hired</th><th className="px-4 py-3 text-right">Performance ratings</th><th className="px-4 py-3 text-right">12-month ratings</th><th className="px-4 py-3 text-right">Still here after 12 months</th><th className="py-3 pl-4 text-right">Observed combined score</th></tr></thead><tbody>{sources.map((source) => <tr key={source.HireSource} className="border-b border-border last:border-0"><td className="py-3 pr-4"><div className="font-medium">{source.HireSource}</div><div className="text-xs text-text-muted">{source.performance_coverage == null ? 'Rating coverage unavailable' : `${Math.round(source.performance_coverage * 100)}% rating coverage`} · {followUpLabel(source.retention_maturity)}</div>{source.role_mix_columns?.length ? <div className="text-xs text-text-muted">Compared across: {source.role_mix_columns.map(roleMixLabel).join(', ')}</div> : null}{source.quality_unavailable_reason ? <div className="text-xs text-amber-700">{source.quality_unavailable_reason}</div> : null}</td><td className="px-4 py-3 text-right">{source.total_hires ?? source.hire_count}</td><td className="px-4 py-3 text-right">{source.performance_recorded_observations ?? '—'}</td><td className="px-4 py-3 text-right">{source.performance_window_observations ?? source.performance_observations ?? '—'}</td><td className="px-4 py-3 text-right">{source.retention_rate_pct == null ? '—' : `${source.retention_rate_pct.toFixed(0)}%`}</td><td className="py-3 pl-4 text-right"><StatusBadge tone="neutral">{source.quality_score?.toFixed(1) ?? 'Unavailable'}</StatusBadge></td></tr>)}</tbody></table>
          {!sources.length && <EmptyState title="No source cohort data available" description="No source groups met the minimum analysis requirements." />}
        </div>
      </Surface>}

      {tab === 'associations' && <Surface id="quality-associations" role="tabpanel" padding="lg">
        <SectionHeader title="Measured pre-hire relationships" description="Relationships between recorded pre-hire measures and post-hire outcomes." />
        <div className="mt-5 space-y-4">{visibleAssociations.length ? visibleAssociations.map((item: PreHireCorrelation, index: number) => { const name = humanize(item.display_name ?? item.predictor ?? `Signal ${index + 1}`); const value = Number(item.correlation ?? 0); return <RelationshipInsight key={`${name}-${index}`} signal={name} outcome={outcomeLabel(data?.correlations?.outcome_column)} correlation={value} observations={item.sample_size} pValue={item.p_value} context="hiring" nextStep="Compare this pattern across roles and future hiring cohorts before changing the interview process." /> }) : <EmptyState title="No relationship analysis available" description="The current dataset does not contain enough paired pre-hire and post-hire measures." />}</div>
        {associations.length > 4 && <div className="mt-5 flex justify-center"><Button type="button" variant="secondary" size="sm" aria-expanded={showAllAssociations} onClick={() => setShowAllAssociations(value => !value)}>{showAllAssociations ? 'Show fewer signals' : `Show all ${associations.length} signals`}<ArrowRight className={`h-3.5 w-3.5 transition-transform ${showAllAssociations ? 'rotate-[-90deg]' : 'rotate-90'}`} aria-hidden="true" /></Button></div>}
        {data?.correlations?.measurement_gaps?.length ? <div className="mt-5 rounded-xl border border-border bg-surface-muted p-4 text-sm leading-6 text-text-secondary"><div className="font-medium text-text-primary">Signals without enough measured support</div><div className="mt-1">{data.correlations.measurement_gaps.map((gap) => `${humanize(gap.display_name)}: ${gap.paired_observations} of ${gap.minimum_paired_observations} paired observations (${humanize(gap.reason)})`).join(' · ')}</div></div> : null}
      </Surface>}

      <TrustDisclosure title="How to interpret hiring insights" summary="Observed patterns, not causal hiring effects">
        <p>Source scores are configurable descriptive composites. They are not validated measures of hire quality, and source differences may reflect role mix, tenure exposure, sample composition or other factors.</p>
        <p className="mt-2">Correlations show association, not causation or expected improvement from changing a hiring criterion. Retained share is an observed cohort share unless a duration-qualified retention window is explicitly available.</p>
        {warnings.length ? <ul className="mt-3 space-y-1.5">{warnings.slice(0, 4).map((item, index) => <li key={index}>• {item}</li>)}</ul> : null}
      </TrustDisclosure>
    </Page>
  )
}
