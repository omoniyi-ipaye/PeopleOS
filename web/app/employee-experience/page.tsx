'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api-client'
import { Button, EmptyState, MetricCard, Page, PageHeader, SectionHeader, StateSummary, StatusBadge, Surface } from '@/components/ui'
import { Activity, ArrowUpRight, Heart, Layers, RefreshCw, Signal, Sparkles, Target } from 'lucide-react'

type ExperienceTab = 'overview' | 'drivers'

interface Segment { segment: string; count: number; percentage: number; avg_exi: number }
interface Driver { factor: string; correlation: number; impact: string; direction: string }
interface Stage { stage: string; count: number; avg_exi: number; at_risk_count: number }
interface ExperienceAnalysis {
  experience_index: { available: boolean; overall_exi?: number; interpretation?: string; benchmark?: string }
  segments: { available: boolean; segments?: Segment[]; thriving_percentage?: number; at_risk_percentage?: number }
  drivers: { available: boolean; drivers?: Driver[] }
  lifecycle: { available: boolean; stages?: Stage[] }
  signals: { has_enps: boolean; has_pulse: boolean; total_signals: number }
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

  if (isLoading) return <Page><StateSummary title="Building the experience picture" description="Calculating engagement, lifecycle and experience signals from the active dataset." tone="info" /></Page>
  if (isError) return <Page><EmptyState title="Employee Experience is unavailable" description={error instanceof Error ? error.message : 'The experience analysis could not be loaded.'} action={<Button onClick={() => refetch()}><RefreshCw className="h-4 w-4" />Retry</Button>} /></Page>

  const score = data?.summary.overall_exi ?? data?.experience_index.overall_exi
  const segments = data?.segments.segments ?? []
  const drivers = data?.drivers.drivers ?? []
  const stages = data?.lifecycle.stages ?? []

  return (
    <Page>
      <PageHeader eyebrow="Understand · Employee Experience" title="How are people experiencing the organisation?" description="A decision-ready view of experience quality, engagement segments, lifecycle pressure and the strongest observed drivers." />

      <div className="flex flex-wrap gap-2">
        <Button variant={tab === 'overview' ? 'primary' : 'secondary'} size="sm" onClick={() => setTab('overview')}><Heart className="h-4 w-4" />Experience overview</Button>
        <Button variant={tab === 'drivers' ? 'primary' : 'secondary'} size="sm" onClick={() => setTab('drivers')}><Layers className="h-4 w-4" />Drivers & lifecycle</Button>
      </div>

      {data?.warnings?.length ? <StateSummary title="Use with context" description={data.warnings.slice(0, 2).join(' · ')} tone="warning" /> : null}

      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="Experience index" value={score === undefined ? '—' : Math.round(score)} detail={data?.experience_index.interpretation ?? 'Composite experience signal'} icon={Activity} tone={toneForScore(score)} />
        <MetricCard label="People analysed" value={(data?.summary.total_employees ?? 0).toLocaleString()} detail="Current dataset coverage" icon={Target} />
        <MetricCard label="Signals available" value={(data?.signals.total_signals ?? 0).toLocaleString()} detail={`${data?.signals.has_enps ? 'eNPS · ' : ''}${data?.signals.has_pulse ? 'Pulse' : 'Structured workforce data'}`} icon={Signal} />
        <MetricCard label="Needs attention" value={(data?.summary.at_risk_count ?? 0).toLocaleString()} detail="Low-experience population signal" icon={Sparkles} tone={(data?.summary.at_risk_count ?? 0) > 0 ? 'warning' : 'success'} />
      </section>

      {tab === 'overview' ? (
        <div className="grid gap-6 xl:grid-cols-[minmax(0,1.15fr)_minmax(340px,0.85fr)]">
          <Surface padding="lg">
            <SectionHeader title="Experience distribution" description="See the workforce mix before jumping to interventions." />
            <div className="mt-5 space-y-4">
              {segments.length ? segments.map((segment) => (
                <div key={segment.segment} className="grid gap-3 border-b border-border py-3 last:border-0 sm:grid-cols-[minmax(160px,0.6fr)_minmax(220px,1fr)_auto] sm:items-center">
                  <div><div className="font-semibold">{segment.segment}</div><div className="text-xs text-text-muted">{segment.count.toLocaleString()} people · EXI {segment.avg_exi.toFixed(1)}</div></div>
                  <div className="h-2 overflow-hidden rounded-full bg-background-secondary"><div className="h-full rounded-full bg-accent" style={{ width: `${Math.min(100, Math.max(0, segment.percentage))}%` }} /></div>
                  <div className="text-sm font-semibold">{segment.percentage.toFixed(1)}%</div>
                </div>
              )) : <EmptyState title="No segment view available" description="The active dataset does not contain enough experience signals to segment the workforce." />}
            </div>
          </Surface>

          <Surface padding="lg">
            <SectionHeader title="What deserves attention" description="Signals are shown as investigation prompts, not automatic people decisions." />
            <div className="mt-5 space-y-3">
              <StateSummary title={data?.summary.health_indicator || 'Experience state'} description={data?.experience_index.benchmark ?? 'Compare this signal with your own organisational baseline.'} tone={toneForScore(score)} />
              {(data?.recommendations ?? []).slice(0, 4).map((item, index) => <div key={`${item}-${index}`} className="flex gap-3 rounded-xl border border-border p-4"><span className="mt-0.5 text-xs font-bold text-accent">0{index + 1}</span><p className="text-sm leading-6 text-text-secondary">{item}</p></div>)}
            </div>
          </Surface>
        </div>
      ) : (
        <div className="grid gap-6 lg:grid-cols-2">
          <Surface padding="lg">
            <SectionHeader title="Observed experience drivers" description="Correlation is directional evidence, not proof of causation." />
            <div className="mt-5 space-y-3">
              {drivers.length ? drivers.map((driver) => <div key={driver.factor} className="flex items-center justify-between gap-4 border-b border-border py-3 last:border-0"><div><div className="font-medium">{driver.factor}</div><div className="text-xs text-text-muted">{driver.impact} · {driver.direction}</div></div><StatusBadge tone={Math.abs(driver.correlation) >= .4 ? 'info' : 'neutral'}>{driver.correlation.toFixed(2)}</StatusBadge></div>) : <EmptyState title="No driver analysis available" />}
            </div>
          </Surface>
          <Surface padding="lg">
            <SectionHeader title="Lifecycle experience" description="Where experience pressure appears across tenure or employee stages." />
            <div className="mt-5 space-y-3">
              {stages.length ? stages.map((stage) => <div key={stage.stage} className="grid grid-cols-[1fr_auto] gap-4 border-b border-border py-3 last:border-0"><div><div className="font-medium">{stage.stage}</div><div className="text-xs text-text-muted">{stage.count.toLocaleString()} people · EXI {stage.avg_exi.toFixed(1)}</div></div><div className="text-right"><div className="font-semibold">{stage.at_risk_count}</div><div className="text-[11px] text-text-muted">attention</div></div></div>) : <EmptyState title="No lifecycle analysis available" />}
            </div>
          </Surface>
        </div>
      )}

      <Surface padding="md" className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div><div className="font-semibold">Need to understand why this is happening?</div><div className="text-sm text-text-secondary">Take the strongest experience signal into People Intelligence and inspect the evidence before acting.</div></div>
        <a href="/advisor" className="inline-flex items-center gap-2 text-sm font-semibold text-accent">Investigate evidence <ArrowUpRight className="h-4 w-4" /></a>
      </Surface>
    </Page>
  )
}
