'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api-client'
import { Button, EmptyState, MetricCard, Page, PageHeader, SectionHeader, StateSummary, StatusBadge, Surface } from '@/components/ui'
import { Award, BarChart3, Target, UserPlus } from 'lucide-react'
import type { QualityOfHireAnalysisResult, SourceEffectiveness } from '@/types/api'

type Tab = 'overview' | 'sources' | 'associations'

export default function QualityOfHirePage() {
  const [tab, setTab] = useState<Tab>('overview')
  const { data, isLoading, isError, error, refetch } = useQuery<QualityOfHireAnalysisResult>({
    queryKey: ['quality-of-hire', 'analysis'],
    queryFn: () => api.qualityOfHire.getAnalysis() as Promise<QualityOfHireAnalysisResult>,
  })

  const header = <PageHeader eyebrow="Understand · Quality of Hire" title="Which hiring inputs are associated with post-hire outcomes?" description="Observational source and pre-hire evidence for hypothesis generation. PeopleOS does not infer causation or automatically change selection criteria." />
  if (isLoading) return <Page>{header}<StateSummary title="Evaluating hiring evidence" description="Comparing source cohorts and paired pre-hire/outcome measurements." tone="info" /></Page>
  if (isError) return <Page>{header}<EmptyState title="Quality of Hire analysis is temporarily unavailable" description={error instanceof Error ? error.message : 'PeopleOS could not complete the hiring analysis.'} action={<Button onClick={() => refetch()}>Retry analysis</Button>} tone="warning" /></Page>

  const summary = data?.summary
  const sources: (SourceEffectiveness & {performance_observations?: number; outcome_observations?: number; quality_unavailable_reason?: string; quality_weights?: Record<string, number>})[] = data?.source_effectiveness ?? []
  const associations = data?.correlations?.correlations ?? []
  const warnings = data?.warnings ?? []
  const recommendations = data?.recommendations ?? []
  const count = (value?: number) => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value.toLocaleString() : 'Unavailable'

  return (
    <Page>
      {header}
      <StateSummary title="Association is not causation" description="Quality composites are configurable heuristics. Correlations and source differences should be validated prospectively before changing hiring rubrics, sourcing allocation or candidate decisions." tone="info" />

      <div className="flex flex-wrap gap-2" role="tablist" aria-label="Quality of hire views">
        <Button role="tab" aria-selected={tab === 'overview'} aria-controls="quality-overview" size="sm" variant={tab === 'overview' ? 'primary' : 'secondary'} onClick={() => setTab('overview')}><Award className="h-4 w-4" />Overview</Button>
        <Button role="tab" aria-selected={tab === 'sources'} aria-controls="quality-sources" size="sm" variant={tab === 'sources' ? 'primary' : 'secondary'} onClick={() => setTab('sources')}><Target className="h-4 w-4" />Source cohorts</Button>
        <Button role="tab" aria-selected={tab === 'associations'} aria-controls="quality-associations" size="sm" variant={tab === 'associations' ? 'primary' : 'secondary'} onClick={() => setTab('associations')}><BarChart3 className="h-4 w-4" />Observed associations</Button>
      </div>

      {warnings.length ? <StateSummary title="Interpretation limits" description={warnings.slice(0, 3).join(' · ')} tone="info" /> : null}

      {tab === 'overview' && <div id="quality-overview" role="tabpanel" className="contents">
        <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          <MetricCard label="People represented" value={count(summary?.total_employees)} detail="Current historical outcome sample" icon={UserPlus} />
          <MetricCard label="Source cohorts" value={count(summary?.sources_analyzed)} detail="Recruiting channels meeting analysis thresholds" icon={Target} />
          <MetricCard label="Pre-hire measures" value={count(summary?.prehire_signals_count)} detail="Candidate-stage variables available for association screening" icon={BarChart3} />
        </section>
        <div className="grid gap-6 lg:grid-cols-2">
          <Surface padding="lg"><SectionHeader title="Highest composite source cohort" description="Highest configured descriptive composite in this sample — not a causal source ranking." /><div className="mt-5"><div className="text-2xl font-semibold">{summary?.best_source ?? 'Not available'}</div><p className="mt-2 text-sm text-text-secondary">Compare role mix, tenure exposure, sample size and source cost before interpreting differences.</p></div></Surface>
          <Surface padding="lg"><SectionHeader title="Strongest observed pre-hire association" description="Largest observed relationship with the selected post-hire outcome." /><div className="mt-5"><div className="text-2xl font-semibold">{summary?.top_predictor ?? 'Not available'}</div><p className="mt-2 text-sm text-text-secondary">This is screening evidence only. Validate stability, fairness and prospective performance before operational use.</p></div></Surface>
        </div>
        {recommendations.length ? <Surface padding="lg"><SectionHeader title="What to validate next" description="Safe next steps are validation tasks, not automated hiring actions." /><div className="mt-5 space-y-3">{recommendations.map((item, index) => <div key={`${item}-${index}`} className="rounded-xl border border-border p-4 text-sm leading-6 text-text-secondary"><span className="mr-2 font-semibold text-accent">0{index + 1}</span>{item}</div>)}</div></Surface> : null}
      </div>}

      {tab === 'sources' && <Surface id="quality-sources" role="tabpanel" padding="lg">
        <SectionHeader title="Hiring source cohorts" description="Descriptive comparisons across source cohorts; retained share is not a duration-qualified retention rate unless a window is explicitly modeled." />
        <div className="mt-5 overflow-x-auto">
          <table className="w-full min-w-[760px] text-sm"><thead><tr className="border-b border-border text-[11px] uppercase tracking-wider text-text-muted"><th className="py-3 pr-4 text-left">Source</th><th className="px-4 py-3 text-right">Hires</th><th className="px-4 py-3 text-right">Avg rating</th><th className="px-4 py-3 text-right">Observed retained share</th><th className="py-3 pl-4 text-right">Comparable heuristic composite</th></tr></thead><tbody>{sources.map((source) => <tr key={source.HireSource} className="border-b border-border last:border-0"><td className="py-3 pr-4"><div className="font-medium">{source.HireSource}</div><div className="text-xs text-text-muted">{source.quality_unavailable_reason || source.recommendation}</div><div className="text-xs text-text-muted">Measured ratings: {source.performance_observations ?? 'Unavailable'} · Observed outcomes: {source.outcome_observations ?? 'Unavailable'}</div><div className="text-xs text-text-muted">{Object.entries(source.quality_weights ?? {}).map(([component, weight]) => `${component}: ${(weight * 100).toFixed(0)}%`).join(' · ')}</div></td><td className="px-4 py-3 text-right">{source.hire_count}</td><td className="px-4 py-3 text-right">{source.avg_performance?.toFixed(2) ?? '—'}</td><td className="px-4 py-3 text-right">{source.retention_rate_pct == null ? '—' : `${source.retention_rate_pct.toFixed(0)}%`}</td><td className="py-3 pl-4 text-right"><StatusBadge tone="neutral">{source.quality_score?.toFixed(1) ?? '—'}</StatusBadge></td></tr>)}</tbody></table>
          {!sources.length && <EmptyState title="No source cohort data available" description="No source groups met the minimum analysis requirements." />}
        </div>
      </Surface>}

      {tab === 'associations' && <Surface id="quality-associations" role="tabpanel" padding="lg">
        <SectionHeader title="Observed pre-hire associations" description="Pearson correlations with sample size and significance. Correlation magnitude is not a percentage improvement or treatment effect." />
        <div className="mt-5 space-y-3">{associations.length ? associations.map((item: any, index: number) => { const name = item.display_name ?? item.predictor ?? `Signal ${index + 1}`; const value = Number(item.correlation ?? 0); return <div key={`${name}-${index}`} className="flex items-center justify-between gap-4 border-b border-border py-3 last:border-0"><div><div className="font-medium">{name}</div><div className="text-xs text-text-muted">n={item.sample_size ?? '—'} · {item.is_significant ? 'statistically distinguishable in this sample' : 'not statistically distinguishable in this sample'}</div></div><StatusBadge tone={Math.abs(value) >= .4 ? 'info' : 'neutral'}>{Number.isFinite(value) ? `r=${value.toFixed(2)}` : '—'}</StatusBadge></div> }) : <EmptyState title="No association analysis available" description="The current dataset does not contain enough paired pre-hire and post-hire measures." />}</div>
      </Surface>}
    </Page>
  )
}
