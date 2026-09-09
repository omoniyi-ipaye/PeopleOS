'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api-client'
import { Button, EmptyState, MetricCard, Page, PageHeader, SectionHeader, StateSummary, StatusBadge, Surface, TrustDisclosure } from '@/components/ui'
import { ArrowUpRight, Heart, Layers, RefreshCw, Signal, Target } from 'lucide-react'

type ExperienceTab = 'overview' | 'patterns'
interface Segment { segment: string; count: number; percentage: number; avg_exi: number | null }
interface Driver { factor: string; correlation: number; impact: string; direction: string }
interface Stage { stage: string; count: number; avg_exi: number | null; at_risk_count: number }
interface ExperienceAnalysis {
  experience_index: { available: boolean; reason?: string; overall_exi?: number; respondent_count?: number; response_coverage?: number; interpretation?: string }
  segments: { available: boolean; reason?: string; segments?: Segment[]; thriving_percentage?: number; at_risk_percentage?: number }
  drivers: { available: boolean; reason?: string; drivers?: Driver[] }
  lifecycle: { available: boolean; reason?: string; stages?: Stage[] }
  signals: { has_enps: boolean; has_pulse: boolean; total_signals: number; coverage_percentage?: number }
  summary: { overall_exi?: number; health_indicator: string; total_employees: number; at_risk_count: number }
  warnings: string[]
  recommendations: string[]
}

export default function EmployeeExperiencePage() {
  const [tab, setTab] = useState<ExperienceTab>('overview')
  const { data, isLoading, isError, error, refetch } = useQuery<ExperienceAnalysis>({ queryKey: ['experience', 'analysis'], queryFn: () => api.experience.getAnalysis() as Promise<ExperienceAnalysis> })
  const header = <PageHeader eyebrow="Insights · Experience" title="How are people experiencing work?" description="See what your measured experience signals say, with response coverage kept visible and assumptions available when you want them." />
  if (isLoading) return <Page>{header}<StateSummary title="Preparing experience insights" description="Reading the measured survey and experience signals available in your data." tone="info" /></Page>
  if (isError) return <Page>{header}<EmptyState title="Experience insights are unavailable" description={error instanceof Error ? error.message : 'PeopleOS could not read the experience data.'} action={<Button onClick={() => refetch()}><RefreshCw className="h-4 w-4" />Retry</Button>} /></Page>

  const measured = Boolean(data?.experience_index.available)
  const rawScore = measured ? (data?.summary.overall_exi ?? data?.experience_index.overall_exi) : undefined
  const score = typeof rawScore === 'number' && Number.isFinite(rawScore) ? rawScore : undefined
  const respondents = measured && Number.isFinite(data?.experience_index.respondent_count) ? data?.experience_index.respondent_count : undefined
  const responseCoverage = measured && Number.isFinite(data?.experience_index.response_coverage) ? data?.experience_index.response_coverage : undefined
  const segments = measured ? (data?.segments.segments ?? []) : []
  const drivers = measured ? (data?.drivers.drivers ?? []) : []
  const stages = measured ? (data?.lifecycle.stages ?? []) : []

  return <Page>
    {header}
    {!measured && <EmptyState title="No measured experience data yet" description="Add explicit survey or experience fields to analyse employee experience. PeopleOS will not guess engagement from salary, tenure or performance." />}

    {measured && <>
      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="Experience composite" value={score === undefined ? 'Unavailable' : Math.round(score)} detail="Configured composite of measured signals" icon={Heart} />
        <MetricCard label="Respondents" value={respondents == null ? 'Unavailable' : respondents.toLocaleString()} detail={responseCoverage == null ? 'Coverage unavailable' : `${(responseCoverage * 100).toFixed(1)}% response coverage`} icon={Target} />
        <MetricCard label="Signals available" value={(data?.signals.total_signals ?? 0).toLocaleString()} detail={`${data?.signals.has_enps ? 'eNPS · ' : ''}${data?.signals.has_pulse ? 'Pulse · ' : ''}measured inputs`} icon={Signal} />
        <MetricCard label="Lower-score band" value={(data?.summary.at_risk_count ?? 0).toLocaleString()} detail="Aggregate count; no employee list exposed" icon={Heart} tone={(data?.summary.at_risk_count ?? 0) > 0 ? 'warning' : 'neutral'} />
      </section>

      <div className="flex flex-wrap gap-2" role="tablist" aria-label="Employee experience views">
        <Button role="tab" aria-selected={tab === 'overview'} variant={tab === 'overview' ? 'primary' : 'secondary'} size="sm" onClick={() => setTab('overview')}><Heart className="h-4 w-4" />Overview</Button>
        <Button role="tab" aria-selected={tab === 'patterns'} variant={tab === 'patterns' ? 'primary' : 'secondary'} size="sm" onClick={() => setTab('patterns')}><Layers className="h-4 w-4" />Patterns</Button>
      </div>

      {tab === 'overview' ? <div className="grid gap-6 xl:grid-cols-[minmax(0,1.2fr)_minmax(320px,0.8fr)]">
        <Surface padding="lg"><SectionHeader title="Experience distribution" description="How measured responses fall across the configured score bands." /><div className="mt-5 space-y-4">{segments.length ? segments.map(segment => <div key={segment.segment} className="grid gap-3 border-b border-border py-3 last:border-0 sm:grid-cols-[minmax(150px,0.6fr)_minmax(220px,1fr)_auto] sm:items-center"><div><div className="font-semibold">{segment.segment}</div><div className="text-xs text-text-muted">{segment.count.toLocaleString()} responses</div></div><div className="h-2 overflow-hidden rounded-full bg-background-secondary"><div className="h-full rounded-full bg-accent" style={{ width: `${Math.min(100, Math.max(0, segment.percentage))}%` }} /></div><div className="text-sm font-semibold">{segment.percentage.toFixed(1)}%</div></div>) : <EmptyState title="No distribution available" />}</div></Surface>
        <Surface padding="lg"><SectionHeader title="What this tells you" description="The clearest interpretation supported by the measured responses." /><div className="mt-5 space-y-3"><StateSummary title="Measured experience is available" description={data?.experience_index.interpretation ?? 'Interpret the composite together with response coverage and the underlying signal definitions.'} tone="info" />{(data?.recommendations ?? []).slice(0, 3).map((item, index) => <div key={`${item}-${index}`} className="rounded-xl border border-border p-4 text-sm leading-6 text-text-secondary">{item}</div>)}</div></Surface>
      </div> : <div className="grid gap-6 lg:grid-cols-2">
        <Surface padding="lg"><SectionHeader title="Related patterns" description="Signals that move with the experience composite and may deserve investigation." /><div className="mt-5 space-y-3">{drivers.length ? drivers.map(driver => <div key={driver.factor} className="flex items-center justify-between gap-4 border-b border-border py-3 last:border-0"><div><div className="font-medium">{driver.factor}</div><div className="text-xs text-text-muted">{driver.direction} relationship</div></div><StatusBadge tone={Math.abs(driver.correlation) >= .4 ? 'info' : 'neutral'}>r={driver.correlation.toFixed(2)}</StatusBadge></div>) : <EmptyState title="No reliable related patterns yet" description={data?.drivers.reason ?? 'More paired measurements are needed.'} />}</div></Surface>
        <Surface padding="lg"><SectionHeader title="Lifecycle view" description="How the measured composite differs across workforce stages." /><div className="mt-5 space-y-3">{stages.length ? stages.map(stage => <div key={stage.stage} className="grid grid-cols-[1fr_auto] gap-4 border-b border-border py-3 last:border-0"><div><div className="font-medium">{stage.stage}</div><div className="text-xs text-text-muted">{stage.count.toLocaleString()} measured people</div></div><div className="text-right"><div className="font-semibold">{stage.avg_exi == null ? '—' : stage.avg_exi.toFixed(1)}</div><div className="text-[11px] text-text-muted">composite</div></div></div>) : <EmptyState title="No lifecycle comparison available" description={data?.lifecycle.reason ?? 'More measured experience data is needed.'} />}</div></Surface>
      </div>}

      <TrustDisclosure title="How this experience score works" summary={responseCoverage == null ? undefined : `${(responseCoverage * 100).toFixed(1)}% response coverage`}>
        <p>The Experience composite is a configured weighted combination of explicit measured survey or experience signals. It is not an external benchmark or a diagnosis of individual engagement.</p><p className="mt-2">Related patterns are correlations with the composite and do not prove cause and effect. Missing or out-of-range responses are excluded rather than filled with HRIS proxies.</p>
      </TrustDisclosure>

      <Surface padding="md" className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between"><div><div className="font-semibold">Want to understand a pattern?</div><div className="text-sm text-text-secondary">Take it into People Intelligence and ask a follow-up in normal People language.</div></div><a href="/advisor" className="inline-flex items-center gap-2 text-sm font-semibold text-accent">Ask PeopleOS <ArrowUpRight className="h-4 w-4" /></a></Surface>
    </>}
  </Page>
}
