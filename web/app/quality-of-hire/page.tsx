'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api-client'
import { Button, EmptyState, MetricCard, Page, PageHeader, SectionHeader, StatusBadge, Surface, TrustDisclosure } from '@/components/ui'
import { Award, BarChart3, Target, UserPlus } from 'lucide-react'
import type { QualityOfHireAnalysisResult, SourceEffectiveness } from '@/types/api'

type Tab = 'overview' | 'sources' | 'associations'

function humanize(value?: string | null) {
  if (!value) return 'Not available'
  return value.replaceAll('_', ' ').replace(/([a-z0-9])([A-Z])/g, '$1 $2').replace(/\b\w/g, char => char.toUpperCase())
}

export default function QualityOfHirePage() {
  const [tab, setTab] = useState<Tab>('overview')
  const { data, isLoading, isError, error, refetch } = useQuery<QualityOfHireAnalysisResult>({
    queryKey: ['quality-of-hire', 'analysis'],
    queryFn: () => api.qualityOfHire.getAnalysis() as Promise<QualityOfHireAnalysisResult>,
  })

  const header = <PageHeader eyebrow="Insights · Hiring" title="What can we learn from our hiring data?" description="Compare hiring-source cohorts and measured pre-hire signals with recorded post-hire outcomes." />
  if (isLoading) return <Page>{header}<div className="rounded-2xl border border-border p-5 text-sm text-text-secondary">Preparing hiring insights…</div></Page>
  if (isError) return <Page>{header}<EmptyState title="Hiring insights are temporarily unavailable" description={error instanceof Error ? error.message : 'PeopleOS could not complete the hiring analysis.'} action={<Button onClick={() => refetch()}>Retry analysis</Button>} tone="warning" /></Page>

  const summary = data?.summary
  const sources: (SourceEffectiveness & {performance_observations?: number; outcome_observations?: number; quality_unavailable_reason?: string; quality_weights?: Record<string, number>})[] = data?.source_effectiveness ?? []
  const associations = data?.correlations?.correlations ?? []
  const warnings = data?.warnings ?? []
  const recommendations = data?.recommendations ?? []
  const count = (value?: number) => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value.toLocaleString() : 'Unavailable'

  return (
    <Page>
      {header}

      <div className="flex flex-wrap gap-2" role="tablist" aria-label="Quality of hire views">
        <Button role="tab" aria-selected={tab === 'overview'} aria-controls="quality-overview" size="sm" variant={tab === 'overview' ? 'primary' : 'secondary'} onClick={() => setTab('overview')}><Award className="h-4 w-4" />Overview</Button>
        <Button role="tab" aria-selected={tab === 'sources'} aria-controls="quality-sources" size="sm" variant={tab === 'sources' ? 'primary' : 'secondary'} onClick={() => setTab('sources')}><Target className="h-4 w-4" />Source cohorts</Button>
        <Button role="tab" aria-selected={tab === 'associations'} aria-controls="quality-associations" size="sm" variant={tab === 'associations' ? 'primary' : 'secondary'} onClick={() => setTab('associations')}><BarChart3 className="h-4 w-4" />Related signals</Button>
      </div>

      {tab === 'overview' && <div id="quality-overview" role="tabpanel" className="contents">
        <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          <MetricCard label="People represented" value={count(summary?.total_employees)} detail="Historical post-hire outcome sample" icon={UserPlus} />
          <MetricCard label="Source cohorts" value={count(summary?.sources_analyzed)} detail="Recruiting channels with enough data" icon={Target} />
          <MetricCard label="Pre-hire measures" value={count(summary?.prehire_signals_count)} detail="Candidate-stage measures available to compare" icon={BarChart3} />
        </section>

        <div className="grid gap-6 lg:grid-cols-2">
          <Surface padding="lg">
            <SectionHeader title="Highest configured source score" description="The source cohort with the highest descriptive composite in this dataset." />
            <div className="mt-5"><div className="text-3xl font-semibold">{summary?.best_source ?? 'Not available'}</div><p className="mt-2 text-sm leading-6 text-text-secondary">Treat this as a place to investigate—not a ranking of recruiting sources. Role mix, sample size and tenure exposure can all affect the comparison.</p></div>
          </Surface>
          <Surface padding="lg">
            <SectionHeader title="Strongest measured pre-hire relationship" description="The largest observed association with the selected post-hire outcome." />
            <div className="mt-5"><div className="text-3xl font-semibold">{humanize(summary?.top_predictor)}</div><p className="mt-2 text-sm leading-6 text-text-secondary">Use this to form a hypothesis for validation. It does not show that the pre-hire measure caused the later outcome.</p></div>
          </Surface>
        </div>

        {recommendations.length ? <Surface padding="lg"><SectionHeader title="Useful next checks" description="Validation tasks you can explore before changing a hiring process." /><div className="mt-5 grid gap-3 md:grid-cols-2">{recommendations.slice(0, 4).map((item, index) => <div key={`${item}-${index}`} className="rounded-xl border border-border p-4 text-sm leading-6 text-text-secondary"><span className="mr-2 font-semibold text-accent">0{index + 1}</span>{item}</div>)}</div></Surface> : null}
      </div>}

      {tab === 'sources' && <Surface id="quality-sources" role="tabpanel" padding="lg">
        <SectionHeader title="Hiring source cohorts" description="Compare what is recorded for each source cohort. Open the methodology note below for measurement limits." />
        <div className="mt-5 overflow-x-auto">
          <table className="w-full min-w-[720px] text-sm"><thead><tr className="border-b border-border text-[11px] uppercase tracking-wider text-text-muted"><th className="py-3 pr-4 text-left">Source</th><th className="px-4 py-3 text-right">Hires</th><th className="px-4 py-3 text-right">Avg rating</th><th className="px-4 py-3 text-right">Observed retained share</th><th className="py-3 pl-4 text-right">Configured score</th></tr></thead><tbody>{sources.map((source) => <tr key={source.HireSource} className="border-b border-border last:border-0"><td className="py-3 pr-4"><div className="font-medium">{source.HireSource}</div><div className="text-xs text-text-muted">Measured ratings: {source.performance_observations ?? 'Unavailable'} · Recorded outcomes: {source.outcome_observations ?? 'Unavailable'}</div></td><td className="px-4 py-3 text-right">{source.hire_count}</td><td className="px-4 py-3 text-right">{source.avg_performance?.toFixed(2) ?? '—'}</td><td className="px-4 py-3 text-right">{source.retention_rate_pct == null ? '—' : `${source.retention_rate_pct.toFixed(0)}%`}</td><td className="py-3 pl-4 text-right"><StatusBadge tone="neutral">{source.quality_score?.toFixed(1) ?? '—'}</StatusBadge></td></tr>)}</tbody></table>
          {!sources.length && <EmptyState title="No source cohort data available" description="No source groups met the minimum analysis requirements." />}
        </div>
      </Surface>}

      {tab === 'associations' && <Surface id="quality-associations" role="tabpanel" padding="lg">
        <SectionHeader title="Measured pre-hire relationships" description="Relationships between recorded pre-hire measures and post-hire outcomes." />
        <div className="mt-5 space-y-3">{associations.length ? associations.map((item: any, index: number) => { const name = humanize(item.display_name ?? item.predictor ?? `Signal ${index + 1}`); const value = Number(item.correlation ?? 0); return <div key={`${name}-${index}`} className="flex items-center justify-between gap-4 border-b border-border py-3 last:border-0"><div><div className="font-medium">{name}</div><div className="text-xs text-text-muted">{item.sample_size == null ? 'Sample size unavailable' : `${item.sample_size.toLocaleString()} paired observations`} · {item.is_significant ? 'distinguishable from zero in this sample' : 'not distinguishable from zero in this sample'}</div></div><StatusBadge tone={Math.abs(value) >= .4 ? 'info' : 'neutral'}>{Number.isFinite(value) ? `r=${value.toFixed(2)}` : '—'}</StatusBadge></div> }) : <EmptyState title="No relationship analysis available" description="The current dataset does not contain enough paired pre-hire and post-hire measures." />}</div>
      </Surface>}

      <TrustDisclosure title="How to interpret hiring insights" summary="Observed relationships, not causal hiring effects">
        <p>Source scores are configurable descriptive composites. They are not validated measures of hire quality, and source differences may reflect role mix, tenure exposure, sample composition or other factors.</p>
        <p className="mt-2">Correlations show association, not causation or expected improvement from changing a hiring criterion. Retained share is an observed cohort share unless a duration-qualified retention window is explicitly available.</p>
        {warnings.length ? <ul className="mt-3 space-y-1.5">{warnings.slice(0, 4).map((item, index) => <li key={index}>• {item}</li>)}</ul> : null}
      </TrustDisclosure>
    </Page>
  )
}
