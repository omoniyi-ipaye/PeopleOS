'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api-client'
import { Button, EmptyState, MetricCard, Page, PageHeader, SectionHeader, StateSummary, StatusBadge, Surface } from '@/components/ui'
import { Activity, ArrowUpRight, Heart, Layers, RefreshCw, Signal, Target } from 'lucide-react'

type ExperienceTab = 'overview' | 'associations'

interface Segment { segment: string; count: number; percentage: number; avg_exi: number }
interface Driver { factor: string; correlation: number; impact: string; direction: string }
interface Stage { stage: string; count: number; avg_exi: number; at_risk_count: number }
interface ExperienceAnalysis {
  experience_index: { available: boolean; reason?: string; overall_exi?: number; interpretation?: string }
  segments: { available: boolean; reason?: string; segments?: Segment[]; thriving_percentage?: number; at_risk_percentage?: number }
  drivers: { available: boolean; reason?: string; drivers?: Driver[] }
  lifecycle: { available: boolean; reason?: string; stages?: Stage[] }
  signals: { has_enps: boolean; has_pulse: boolean; total_signals: number; coverage_percentage?: number }
  summary: { overall_exi?: number; health_indicator: string; total_employees: number; at_risk_count: number }
  warnings: string[]
  recommendations: string[]
}

function toneForScore(value?: number) {
  if (value === undefined) return 'neutral' as const
  if (value >= 80) return 'success' as const
  if (value >= 60) return 'info' as const
  if (value >= 40) return 'warning' as const
  return 'danger' as const
}

export default function EmployeeExperiencePage() {
  const [tab, setTab] = useState<ExperienceTab>('overview')
  const { data, isLoading, isError, error, refetch } = useQuery<ExperienceAnalysis>({
    queryKey: ['experience', 'analysis'],
    queryFn: () => api.experience.getAnalysis() as Promise<ExperienceAnalysis>,
  })

  const header = <PageHeader eyebrow="Understand · Employee Experience" title="What do measured experience signals tell us?" description="PeopleOS reports experience only from explicit survey or experience measurements. It will not infer engagement from tenure, performance, salary or promotion proxies." />
  if (isLoading) return <Page>{header}<StateSummary title="Reading measured experience signals" description="Checking survey coverage and aggregate experience evidence." tone="info" /></Page>
  if (isError) return <Page>{header}<EmptyState title="Employee Experience is unavailable" description={error instanceof Error ? error.message : 'The experience analysis could not be loaded.'} action={<Button onClick={() => refetch()}><RefreshCw className="h-4 w-4" />Retry</Button>} /></Page>

  const measured = Boolean(data?.experience_index.available)
  const score = measured ? (data?.summary.overall_exi ?? data?.experience_index.overall_exi) : undefined
  const segments = measured ? (data?.segments.segments ?? []) : []
  const drivers = measured ? (data?.drivers.drivers ?? []) : []
  const stages = measured ? (data?.lifecycle.stages ?? []) : []

  return (
    <Page>
      {header}
      {!measured && <StateSummary title="Measured experience data is not available" description={data?.experience_index.reason ?? 'Add explicit experience survey signals before interpreting workforce experience.'} tone="warning" />}
      {data?.warnings?.length ? <StateSummary title="Interpretation limits" description={data.warnings.slice(0, 3).join(' · ')} tone="warning" /> : null}

      <div className="flex flex-wrap gap-2">
        <Button variant={tab === 'overview' ? 'primary' : 'secondary'} size="sm" onClick={() => setTab('overview')}><Heart className="h-4 w-4" />Measured overview</Button>
        <Button variant={tab === 'associations' ? 'primary' : 'secondary'} size="sm" onClick={() => setTab('associations')}><Layers className="h-4 w-4" />Associations & lifecycle</Button>
      </div>

      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="Experience composite" value={score === undefined ? 'Not available' : Math.round(score)} detail={measured ? 'Configured weighted composite of measured signals' : 'No proxy-derived score is created'} icon={Activity} tone={toneForScore(score)} />
        <MetricCard label="People represented" value={(data?.summary.total_employees ?? 0).toLocaleString()} detail="Current workforce sample" icon={Target} />
        <MetricCard label="Measured signals" value={(data?.signals.total_signals ?? 0).toLocaleString()} detail={`${data?.signals.has_enps ? 'eNPS · ' : ''}${data?.signals.has_pulse ? 'Pulse · ' : ''}${data?.signals.coverage_percentage == null ? 'coverage not reported' : `${data.signals.coverage_percentage.toFixed(0)}% employee coverage`}`} icon={Signal} />
        <MetricCard label="Low-score aggregate" value={measured ? (data?.summary.at_risk_count ?? 0).toLocaleString() : '—'} detail={measured ? 'Aggregate score-band count; no employee list exposed' : 'Unavailable without measured signals'} icon={Heart} tone={measured && (data?.summary.at_risk_count ?? 0) > 0 ? 'warning' : 'neutral'} />
      </section>

      {tab === 'overview' ? (
        <div className="grid gap-6 xl:grid-cols-[minmax(0,1.15fr)_minmax(340px,0.85fr)]">
          <Surface padding="lg">
            <SectionHeader title="Measured score distribution" description="Configured score bands for aggregate monitoring, not diagnoses of individual engagement." />
            <div className="mt-5 space-y-4">
              {segments.length ? segments.map((segment) => (
                <div key={segment.segment} className="grid gap-3 border-b border-border py-3 last:border-0 sm:grid-cols-[minmax(160px,0.6fr)_minmax(220px,1fr)_auto] sm:items-center">
                  <div><div className="font-semibold">{segment.segment}</div><div className="text-xs text-text-muted">{segment.count.toLocaleString()} people · composite {segment.avg_exi.toFixed(1)}</div></div>
                  <div className="h-2 overflow-hidden rounded-full bg-background-secondary"><div className="h-full rounded-full bg-accent" style={{ width: `${Math.min(100, Math.max(0, segment.percentage))}%` }} /></div>
                  <div className="text-sm font-semibold">{segment.percentage.toFixed(1)}%</div>
                </div>
              )) : <EmptyState title="No measured segment view available" description="PeopleOS will not create engagement segments from HRIS proxy fields." />}
            </div>
          </Surface>

          <Surface padding="lg">
            <SectionHeader title="What can be concluded" description="Keep the measurement boundary visible." />
            <div className="mt-5 space-y-3">
              <StateSummary title={measured ? 'Measured composite available' : 'Measurement unavailable'} description={measured ? (data?.experience_index.interpretation ?? 'Interpret together with signal coverage and component definitions.') : 'Collect explicit experience measurements before drawing experience conclusions.'} tone={measured ? 'info' : 'warning'} />
              {(data?.recommendations ?? []).slice(0, 4).map((item, index) => <div key={`${item}-${index}`} className="flex gap-3 rounded-xl border border-border p-4"><span className="mt-0.5 text-xs font-bold text-accent">0{index + 1}</span><p className="text-sm leading-6 text-text-secondary">{item}</p></div>)}
            </div>
          </Surface>
        </div>
      ) : (
        <div className="grid gap-6 lg:grid-cols-2">
          <Surface padding="lg">
            <SectionHeader title="Observed associations" description="Correlations with the measured composite are not causal drivers." />
            <div className="mt-5 space-y-3">
              {drivers.length ? drivers.map((driver) => <div key={driver.factor} className="flex items-center justify-between gap-4 border-b border-border py-3 last:border-0"><div><div className="font-medium">{driver.factor}</div><div className="text-xs text-text-muted">{driver.direction} observed association · legacy magnitude band {driver.impact}</div></div><StatusBadge tone={Math.abs(driver.correlation) >= .4 ? 'info' : 'neutral'}>r={driver.correlation.toFixed(2)}</StatusBadge></div>) : <EmptyState title="No association analysis available" description={data?.drivers.reason ?? 'Measured paired data is required.'} />}
            </div>
          </Surface>
          <Surface padding="lg">
            <SectionHeader title="Lifecycle comparison" description="Descriptive differences by stage; not estimated stage effects." />
            <div className="mt-5 space-y-3">
              {stages.length ? stages.map((stage) => <div key={stage.stage} className="grid grid-cols-[1fr_auto] gap-4 border-b border-border py-3 last:border-0"><div><div className="font-medium">{stage.stage}</div><div className="text-xs text-text-muted">{stage.count.toLocaleString()} people · composite {stage.avg_exi.toFixed(1)}</div></div><div className="text-right"><div className="font-semibold">{stage.at_risk_count}</div><div className="text-[11px] text-text-muted">low-score band</div></div></div>) : <EmptyState title="No lifecycle comparison available" description={data?.lifecycle.reason ?? 'Measured experience data is required.'} />}
            </div>
          </Surface>
        </div>
      )}

      <Surface padding="md" className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div><div className="font-semibold">Need to understand a measured signal?</div><div className="text-sm text-text-secondary">Take the aggregate signal into People Intelligence and inspect supporting evidence and gaps before acting.</div></div>
        <a href="/advisor" className="inline-flex items-center gap-2 text-sm font-semibold text-accent">Investigate evidence <ArrowUpRight className="h-4 w-4" /></a>
      </Surface>
    </Page>
  )
}
