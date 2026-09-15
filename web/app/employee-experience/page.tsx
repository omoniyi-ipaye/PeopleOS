'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import Link from 'next/link'
import { api } from '@/lib/api-client'
import { Button, EmptyState, MetricCard, Page, PageHeader, SectionHeader, StateSummary, Surface, TrustDisclosure } from '@/components/ui'
import { RelationshipInsight } from '@/components/relationship-insight'
import { ArrowUpRight, Heart, Layers, RefreshCw, Signal, Target } from 'lucide-react'

type ExperienceTab = 'overview' | 'patterns'
interface Segment { segment: string; count: number | null; percentage: number | null; avg_exi: number | null; suppressed?: boolean }
interface Driver { factor: string; correlation: number; impact: string; direction: string; sample_size?: number }
interface Stage { stage: string; count: number; avg_exi: number | null; respondent_count?: number; at_risk_count: number | null; at_risk_suppressed?: boolean }
interface ExperienceAnalysis {
  experience_index: { available: boolean; reason?: string; overall_exi?: number; respondent_count?: number; response_coverage?: number; interpretation?: string }
  segments: { available: boolean; reason?: string; segments?: Segment[]; thriving_percentage?: number | null; at_risk_percentage?: number | null; suppression_applied?: boolean }
  drivers: { available: boolean; reason?: string; drivers?: Driver[] }
  lifecycle: { available: boolean; reason?: string; stages?: Stage[] }
  signals: { has_enps: boolean; has_pulse: boolean; total_signals: number; coverage_percentage?: number }
  summary: { overall_exi?: number; health_indicator: string; total_employees: number; at_risk_count: number | null }
  warnings: string[]
  recommendations: string[]
}

const segmentLabels: Record<string, string> = {
  'Very high score band': 'Very high score band',
  'High score band': 'High score band',
  'Mid score band': 'Middle score band',
  'Low score band': 'Low score band',
  'Very low score band': 'Very low score band',
}

function segmentLabel(value: string) { return segmentLabels[value] ?? value.replaceAll('_', ' ') }
function humanize(value: string) { return value.replaceAll('_', ' ').replace(/([a-z0-9])([A-Z])/g, '$1 $2').replace(/\b\w/g, char => char.toUpperCase()) }

export default function EmployeeExperiencePage() {
  const [tab, setTab] = useState<ExperienceTab>('overview')
  const [showAllDrivers, setShowAllDrivers] = useState(false)
  const { data, isLoading, isError, error, refetch } = useQuery<ExperienceAnalysis>({ queryKey: ['experience', 'analysis'], queryFn: () => api.experience.getAnalysis() as Promise<ExperienceAnalysis> })
  const header = <PageHeader eyebrow="Insights · Experience" title="How are people experiencing work?" description="See what your measured experience signals say, with response coverage kept visible and assumptions available when you want them." actions={<Link href="/advisor" className="inline-flex shrink-0 items-center justify-center gap-2 rounded-xl bg-violet-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-violet-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500">Ask PeopleOS <ArrowUpRight className="h-4 w-4" /></Link>} />
  if (isLoading) return <Page>{header}<StateSummary title="Preparing experience insights" description="Reading the measured survey and experience signals available in your data." tone="info" /></Page>
  if (isError) return <Page>{header}<EmptyState title="Experience insights are unavailable" description={error instanceof Error ? error.message : 'PeopleOS could not read the experience data.'} action={<Button onClick={() => refetch()}><RefreshCw className="h-4 w-4" />Retry</Button>} /></Page>

  const measured = Boolean(data?.experience_index.available)
  const rawScore = measured ? (data?.summary.overall_exi ?? data?.experience_index.overall_exi) : undefined
  const score = typeof rawScore === 'number' && Number.isFinite(rawScore) ? rawScore : undefined
  const respondents = measured && Number.isFinite(data?.experience_index.respondent_count) ? data?.experience_index.respondent_count : undefined
  const responseCoverage = measured && Number.isFinite(data?.experience_index.response_coverage) ? data?.experience_index.response_coverage : undefined
  const segments = measured ? (data?.segments.segments ?? []) : []
  const drivers = measured ? (data?.drivers.drivers ?? []) : []
  const visibleDrivers = showAllDrivers ? drivers : drivers.slice(0, 4)
  const stages = measured ? (data?.lifecycle.stages ?? []) : []
  const visibleSegments = segments.filter(segment => !segment.suppressed && typeof segment.percentage === 'number' && Number.isFinite(segment.percentage))
  const largestSegment = [...visibleSegments].sort((a, b) => (b.percentage ?? 0) - (a.percentage ?? 0))[0]
  const lowerShare = typeof data?.segments.at_risk_percentage === 'number' && Number.isFinite(data.segments.at_risk_percentage) ? data.segments.at_risk_percentage : undefined
  const lowScoreCount = typeof data?.summary.at_risk_count === 'number' && Number.isFinite(data.summary.at_risk_count) ? data.summary.at_risk_count : undefined

  return <Page>
    {header}
    {!measured && <EmptyState title="No measured experience data yet" description="Add explicit survey or experience fields to analyse employee experience. PeopleOS will not guess engagement from salary, tenure or performance." />}

    {measured && <>
      <div className="flex flex-wrap gap-2" role="tablist" aria-label="Employee experience views">
        <Button role="tab" aria-selected={tab === 'overview'} aria-controls="experience-overview" variant={tab === 'overview' ? 'primary' : 'secondary'} size="sm" onClick={() => setTab('overview')}><Heart className="h-4 w-4" />Overview</Button>
        <Button role="tab" aria-selected={tab === 'patterns'} aria-controls="experience-patterns" variant={tab === 'patterns' ? 'primary' : 'secondary'} size="sm" onClick={() => setTab('patterns')}><Layers className="h-4 w-4" />Patterns</Button>
      </div>

      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="Experience score" value={score === undefined ? 'Unavailable' : Math.round(score)} detail="Combined score from the survey signals in this dataset" icon={Heart} />
        <MetricCard label="Respondents" value={respondents == null ? 'Unavailable' : respondents.toLocaleString()} detail={responseCoverage == null ? 'Coverage unavailable' : `${(responseCoverage * 100).toFixed(1)}% response coverage`} icon={Target} />
        <MetricCard label="Survey signals" value={(data?.signals.total_signals ?? 0).toLocaleString()} detail={`${data?.signals.has_enps ? 'eNPS · ' : ''}${data?.signals.has_pulse ? 'Pulse · ' : ''}other measured inputs`} icon={Signal} />
        <MetricCard label="Lower-score responses" value={lowScoreCount == null ? 'Suppressed' : lowScoreCount.toLocaleString()} detail={lowScoreCount == null ? 'Small-cell privacy protection applied' : 'People in the two lower score bands; aggregate only'} icon={Heart} tone={lowScoreCount != null && lowScoreCount > 0 ? 'warning' : 'neutral'} />
      </section>

      {tab === 'overview' ? <div id="experience-overview" role="tabpanel" className="space-y-6">
        <div className="grid gap-6 xl:grid-cols-[minmax(0,1.2fr)_minmax(320px,0.8fr)]">
        <Surface padding="lg"><SectionHeader title="Experience distribution" description="How measured responses fall across the configured score bands." /><div className="mt-5 space-y-4">{segments.length ? segments.map(segment => {
          const suppressed = Boolean(segment.suppressed) || segment.count == null || segment.percentage == null
          const percentage = suppressed ? 0 : segment.percentage as number
          return <div key={segment.segment} className="grid gap-3 border-b border-border py-3 last:border-0 sm:grid-cols-[minmax(170px,0.7fr)_minmax(220px,1fr)_auto] sm:items-center"><div><div className="font-semibold">{segmentLabel(segment.segment)}</div><div className="text-xs text-text-muted">{suppressed ? 'Small cell suppressed' : `${segment.count!.toLocaleString()} responses`}</div></div><div className="h-2 overflow-hidden rounded-full bg-background-secondary"><div className="h-full rounded-full bg-accent" style={{ width: `${Math.min(100, Math.max(0, percentage))}%` }} /></div><div className="text-sm font-semibold">{suppressed ? '—' : `${percentage.toFixed(1)}%`}</div></div>
        }) : <EmptyState title="No distribution available" />}</div></Surface>
        <Surface padding="lg">
          <SectionHeader title="What the responses show" description="A descriptive read of the measured distribution, not a diagnosis of individual engagement." />
          <div className="mt-5 space-y-4">
            {largestSegment ? <div className="rounded-2xl bg-background-secondary p-5"><div className="text-3xl font-semibold">{largestSegment.percentage!.toFixed(1)}%</div><div className="mt-1 font-semibold">fall in the {segmentLabel(largestSegment.segment).toLowerCase()}</div><div className="mt-2 text-sm leading-6 text-text-secondary">{largestSegment.count!.toLocaleString()} measured responses are represented in this configured band.</div></div> : <div className="rounded-2xl bg-background-secondary p-5 text-sm leading-6 text-text-secondary">The distribution contains privacy-suppressed cells, so PeopleOS is not highlighting a largest band that could help reconstruct them.</div>}
            <div className="rounded-xl border border-border p-4 text-sm leading-6 text-text-secondary">{lowerShare == null ? 'The combined lower-band share is suppressed because revealing it could expose a small survey-derived cell.' : `${lowerShare.toFixed(1)}% of measured responses fall in the two lower configured score bands. Use the underlying survey questions and local context before deciding what this means.`}</div>
          </div>
        </Surface>
        </div>
      </div> : <div id="experience-patterns" role="tabpanel" className="space-y-6">
        <Surface padding="lg"><SectionHeader title="Related patterns" description="Signals that move with the experience score. Open a next check when you want to put a signal into workforce context." /><div className="mt-5 grid gap-4 md:grid-cols-2">{visibleDrivers.length ? visibleDrivers.map(driver => <RelationshipInsight key={driver.factor} signal={humanize(driver.factor)} outcome="overall experience score" correlation={driver.correlation} observations={driver.sample_size} context="experience" nextStep="Check the underlying survey questions and response coverage before treating this as the main lever for experience." />) : <EmptyState title="No reliable related patterns yet" description={data?.drivers.reason ?? 'More paired measurements are needed.'} />}</div>{drivers.length > 4 && <div className="mt-5 flex justify-center"><Button type="button" variant="secondary" size="sm" aria-expanded={showAllDrivers} onClick={() => setShowAllDrivers(value => !value)}>{showAllDrivers ? 'Show fewer patterns' : `Show all ${drivers.length} patterns`}<ArrowUpRight className={`h-3.5 w-3.5 transition-transform ${showAllDrivers ? 'rotate-[-90deg]' : 'rotate-90'}`} aria-hidden="true" /></Button></div>}</Surface>
        <Surface padding="lg"><SectionHeader title="Lifecycle view" description="How the experience score differs across workforce stages." /><div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">{stages.length ? stages.map(stage => <div key={stage.stage} className="rounded-xl border border-border p-4"><div className="font-medium">{humanize(stage.stage)}</div><div className="mt-1 text-xs text-text-muted">{stage.count.toLocaleString()} people · {(stage.respondent_count ?? stage.count).toLocaleString()} respondents</div><div className="mt-4 text-xl font-semibold">{stage.avg_exi == null ? '—' : stage.avg_exi.toFixed(1)}</div><div className="text-[11px] text-text-muted">experience score</div></div>) : <EmptyState title="No lifecycle comparison available" description={data?.lifecycle.reason ?? 'More measured experience data is needed.'} />}</div></Surface>
      </div>}

      <TrustDisclosure title="How this experience score works" summary={responseCoverage == null ? undefined : `${(responseCoverage * 100).toFixed(1)}% response coverage`}>
        <p>The Experience composite is a configured weighted combination of explicit measured survey or experience signals. Band names in the product are intentionally neutral because the composite is not an external benchmark or a diagnosis of individual engagement.</p><p className="mt-2">Related patterns are correlations with the composite and do not prove cause and effect. Missing or out-of-range responses are excluded rather than filled with HRIS proxies. Small survey-derived cells are suppressed rather than displayed as zero.</p>
      </TrustDisclosure>

      <Surface padding="md" className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between"><div><div className="font-semibold">Want to understand a pattern?</div><div className="text-sm text-text-secondary">Take it into People Intelligence and ask a follow-up in normal People language.</div></div><a href="/advisor" className="inline-flex items-center gap-2 text-sm font-semibold text-accent">Ask PeopleOS <ArrowUpRight className="h-4 w-4" /></a></Surface>
    </>}
  </Page>
}
