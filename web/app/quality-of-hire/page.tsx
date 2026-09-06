'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api-client'
import { Button, EmptyState, MetricCard, Page, PageHeader, SectionHeader, StateSummary, StatusBadge, Surface } from '@/components/ui'
import { AlertTriangle, Award, BarChart3, Target, UserPlus } from 'lucide-react'
import type { QualityOfHireAnalysisResult, SourceEffectiveness } from '@/types/api'

type Tab = 'overview' | 'sources' | 'predictors'

export default function QualityOfHirePage() {
  const [tab, setTab] = useState<Tab>('overview')
  const { data, isLoading, isError, error, refetch } = useQuery<QualityOfHireAnalysisResult>({ queryKey: ['quality-of-hire', 'analysis'], queryFn: () => api.qualityOfHire.getAnalysis() as Promise<QualityOfHireAnalysisResult> })

  if (isLoading) return <Page><PageHeader eyebrow="Understand · Quality of Hire" title="Which hiring inputs are associated with better outcomes?" description="A source and signal view that keeps hiring evidence separate from individual employment decisions." /><StateSummary title="Evaluating hiring quality" description="Comparing source quality, performance signals and retention outcomes." tone="info" /></Page>
  if (isError) return <Page><PageHeader eyebrow="Understand · Quality of Hire" title="Which hiring inputs are associated with better outcomes?" description="A source and signal view that keeps hiring evidence separate from individual employment decisions." /><EmptyState title="Quality of Hire analysis is temporarily unavailable" description={error instanceof Error ? error.message : 'PeopleOS could not complete the hiring analysis. Other workforce evidence remains available.'} action={<Button onClick={() => refetch()}>Retry analysis</Button>} tone="warning" /></Page>

  const summary = data?.summary
  const sources: SourceEffectiveness[] = data?.source_effectiveness ?? []
  const correlations = data?.correlations
  const predictorRows = correlations?.correlations ?? []
  const warnings = data?.warnings ?? []
  const recommendations = data?.recommendations ?? []

  return (
    <Page>
      <PageHeader eyebrow="Understand · Quality of Hire" title="Which hiring inputs are associated with better outcomes?" description="A source and signal view that keeps hiring evidence separate from individual employment decisions." />
      <div className="flex flex-wrap gap-2">
        <Button size="sm" variant={tab === 'overview' ? 'primary' : 'secondary'} onClick={() => setTab('overview')}><Award className="h-4 w-4" />Overview</Button>
        <Button size="sm" variant={tab === 'sources' ? 'primary' : 'secondary'} onClick={() => setTab('sources')}><Target className="h-4 w-4" />Hiring sources</Button>
        <Button size="sm" variant={tab === 'predictors' ? 'primary' : 'secondary'} onClick={() => setTab('predictors')}><BarChart3 className="h-4 w-4" />Success signals</Button>
      </div>

      {warnings.length ? <StateSummary title="Data quality affects interpretation" description={warnings.slice(0, 2).join(' · ')} tone="warning" /> : null}

      {tab === 'overview' && <>
        <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <MetricCard label="People analysed" value={(summary?.total_employees ?? 0).toLocaleString()} detail="Historical outcome coverage" icon={UserPlus} />
          <MetricCard label="Sources analysed" value={(summary?.sources_analyzed ?? 0).toLocaleString()} detail="Recruiting channels represented" icon={Target} />
          <MetricCard label="Pre-hire signals" value={(summary?.prehire_signals_count ?? 0).toLocaleString()} detail="Potential success indicators" icon={BarChart3} />
          <MetricCard label="New-hire attention" value={(summary?.new_hires_at_risk ?? 0).toLocaleString()} detail="Aggregate early-tenure signal" icon={AlertTriangle} tone={(summary?.new_hires_at_risk ?? 0) ? 'warning' : 'success'} />
        </section>
        <div className="grid gap-6 lg:grid-cols-2">
          <Surface padding="lg"><SectionHeader title="Strongest source signal" description="Best observed source based on the current quality definition." /><div className="mt-5"><div className="text-2xl font-semibold">{summary?.best_source ?? 'Not available'}</div><p className="mt-2 text-sm text-text-secondary">Treat source effectiveness as portfolio evidence. It should not become an automatic screening rule.</p></div></Surface>
          <Surface padding="lg"><SectionHeader title="Strongest pre-hire signal" description="Highest observed association with the defined success outcome." /><div className="mt-5"><div className="text-2xl font-semibold">{summary?.top_predictor ?? 'Not available'}</div><p className="mt-2 text-sm text-text-secondary">Association is not causation. Validate fairness and stability before operational use.</p></div></Surface>
        </div>
        {recommendations.length ? <Surface padding="lg"><SectionHeader title="What to investigate next" description="Systemic improvements suggested by the observed evidence." /><div className="mt-5 grid gap-3 md:grid-cols-2">{recommendations.slice(0, 6).map((item, index) => <div key={`${item}-${index}`} className="rounded-xl border border-border p-4 text-sm leading-6 text-text-secondary"><span className="mr-2 font-semibold text-accent">0{index + 1}</span>{item}</div>)}</div></Surface> : null}
      </>}

      {tab === 'sources' && <Surface padding="lg">
        <SectionHeader title="Hiring source effectiveness" description="Compare volume, quality, performance and retention without reducing the result to a single grade." />
        <div className="mt-5 overflow-x-auto">
          <table className="w-full min-w-[720px] text-sm"><thead><tr className="border-b border-border text-[11px] uppercase tracking-wider text-text-muted"><th className="py-3 pr-4 text-left">Source</th><th className="px-4 py-3 text-right">Hires</th><th className="px-4 py-3 text-right">Rating</th><th className="px-4 py-3 text-right">Retention</th><th className="py-3 pl-4 text-right">Quality</th></tr></thead><tbody>{sources.map((source) => <tr key={source.HireSource} className="border-b border-border last:border-0"><td className="py-3 pr-4"><div className="font-medium">{source.HireSource}</div><div className="text-xs text-text-muted">{source.recommendation}</div></td><td className="px-4 py-3 text-right">{source.hire_count}</td><td className="px-4 py-3 text-right">{source.avg_performance?.toFixed(2) ?? '—'}</td><td className="px-4 py-3 text-right"><StatusBadge tone={(source.retention_rate_pct ?? 0) >= 85 ? 'success' : (source.retention_rate_pct ?? 0) >= 70 ? 'warning' : 'danger'}>{source.retention_rate_pct == null ? '—' : `${source.retention_rate_pct.toFixed(0)}%`}</StatusBadge></td><td className="py-3 pl-4 text-right font-semibold">{source.quality_score?.toFixed(1) ?? '—'}</td></tr>)}</tbody></table>
          {!sources.length && <EmptyState title="No source effectiveness data available" />}
        </div>
      </Surface>}

      {tab === 'predictors' && <Surface padding="lg">
        <SectionHeader title="Observed success signals" description="Ranked associations from the available pre-hire and outcome data." />
        <div className="mt-5 space-y-3">{predictorRows.length ? predictorRows.map((item: any, index: number) => { const name = item.factor ?? item.feature ?? item.signal ?? `Signal ${index + 1}`; const value = Number(item.correlation ?? item.value ?? 0); return <div key={`${name}-${index}`} className="flex items-center justify-between gap-4 border-b border-border py-3 last:border-0"><div><div className="font-medium">{name}</div><div className="text-xs text-text-muted">Observed relationship with hiring outcome</div></div><StatusBadge tone={Math.abs(value) >= .4 ? 'info' : 'neutral'}>{Number.isFinite(value) ? value.toFixed(2) : '—'}</StatusBadge></div> }) : <EmptyState title="No predictor analysis available" description="The current dataset does not contain enough paired pre-hire and outcome signals." />}</div>
      </Surface>}
    </Page>
  )
}
