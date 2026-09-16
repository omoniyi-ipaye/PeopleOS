'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import Link from 'next/link'
import { api } from '@/lib/api-client'
import { Button, EmptyState, MetricCard, Page, PageHeader, SectionHeader, StateSummary, Surface, TrustDisclosure } from '@/components/ui'
import { Activity, ArrowRight, Clock3, Layers, ShieldAlert } from 'lucide-react'
import { RetentionCurveChart } from '@/components/charts/retention-curve-chart'
import type { SurvivalAnalysisResult, CohortInsight } from '@/types/api'

type Tab = 'overview' | 'cohorts'

function humanize(value?: string | null) {
  if (!value) return null
  return value.replaceAll('_', ' ').replace(/([a-z0-9])([A-Z])/g, '$1 $2').replace(/\b\w/g, char => char.toUpperCase())
}

function cohortLabel(value?: string | null) {
  if (!value) return 'Recorded cohort'
  const department = value.match(/(?:Dept|Department)\s*=\s*([^,)]+)/i)?.[1]?.trim()
  if (department) return `${department} team`
  const tenure = value.match(/Tenure\s*[≥>=]+\s*([\d.]+)\s*y?/i)?.[1]
  const promotion = value.match(/Years\s*Since\s*Promotion\s*[≥>=]+\s*([\d.]+)\s*y?/i)?.[1]
  if (tenure && promotion) return `People with at least ${tenure} years' tenure and ${promotion}+ years since promotion`
  if (tenure) return `People with at least ${tenure} years' tenure`
  return humanize(value) ?? 'Recorded cohort'
}

function cohortFollowUp(name: string) {
  return /team|department/i.test(name) ? 'Recorded attrition share by department' : 'Headcount by department'
}

export default function RetentionForecastPage() {
  const [tab, setTab] = useState<Tab>('overview')
  const { data, isLoading, isError, error, refetch } = useQuery<SurvivalAnalysisResult>({
    queryKey: ['survival', 'analysis'],
    queryFn: () => api.survival.getAnalysis() as Promise<SurvivalAnalysisResult>,
  })

  const header = <PageHeader eyebrow="Insights · Retention" title="What does recorded retention history show?" description="Explore aggregate cohort patterns over time. PeopleOS does not predict whether an individual employee will leave." actions={<Link href="/advisor" className="inline-flex shrink-0 items-center justify-center gap-2 rounded-xl bg-violet-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-violet-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500">Ask PeopleOS <ArrowRight className="h-4 w-4" /></Link>} />
  if (isLoading) return <Page>{header}<StateSummary title="Preparing retention insights" description="Reading recorded tenure and departure outcomes across your workforce." tone="info" /></Page>
  if (isError) return <Page>{header}<EmptyState title="Retention cohort analysis is unavailable" description={error instanceof Error ? error.message : 'Unable to generate cohort retention analysis.'} action={<Button onClick={() => refetch()}>Retry</Button>} /></Page>

  const summary = data?.summary
  const km = data?.kaplan_meier
  const cohorts: CohortInsight[] = data?.cohort_insights ?? []
  const warnings = data?.warnings ?? []

  return (
    <Page>
      {header}

      <div className="flex flex-wrap gap-2" role="tablist" aria-label="Retention views">
        <Button role="tab" aria-selected={tab === 'overview'} aria-controls="retention-overview" size="sm" variant={tab === 'overview' ? 'primary' : 'secondary'} onClick={() => setTab('overview')}><Activity className="h-4 w-4" />Retention curve</Button>
        <Button role="tab" aria-selected={tab === 'cohorts'} aria-controls="retention-cohorts" size="sm" variant={tab === 'cohorts' ? 'primary' : 'secondary'} onClick={() => setTab('cohorts')}><Layers className="h-4 w-4" />Cohorts</Button>
      </div>

      {tab === 'overview' && <div id="retention-overview" role="tabpanel" className="space-y-6">
        <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <MetricCard label="Observed midpoint tenure" value={summary?.median_tenure == null ? '—' : `${summary.median_tenure.toFixed(1)}y`} detail="Recorded tenure point where half the observed cohort history remains, if available" icon={Clock3} />
          <MetricCard label="Recorded attrition" value={summary?.overall_attrition_rate == null ? '—' : `${(summary.overall_attrition_rate * 100).toFixed(1)}%`} detail="Known outcomes marked as departed" icon={Activity} />
          <MetricCard label="Aggregate retention model" value={summary?.cox_model_fitted ? 'Available' : 'Not available'} detail="Optional group-level comparison; not an individual forecast" icon={ShieldAlert} tone={summary?.cox_model_fitted ? 'info' : 'neutral'} />
          <MetricCard label="Cohorts compared" value={cohorts.length.toLocaleString()} detail="Groups meeting analysis requirements" icon={Layers} />
        </section>

        <Surface padding="lg">
          <SectionHeader title="People still here over tenure" description="The share of each recorded group still here at each tenure point." />
          <div className="mt-6">{km?.overall?.survival_function?.length ? <RetentionCurveChart data={km.overall.survival_function} /> : <EmptyState title="No retention curve available" description="The current dataset does not provide enough valid time-to-event information." />}</div>
        </Surface>
      </div>}

      {tab === 'cohorts' && <Surface id="retention-cohorts" role="tabpanel" padding="lg">
        <SectionHeader title="Cohort comparison" description="Compare recorded time-to-exit patterns across the groups available in your data." />
        <div className="mt-5 grid gap-4 lg:grid-cols-2">{cohorts.length ? cohorts.map((cohort, index) => {
          const name = cohortLabel(cohort.cohort_name ?? cohort.cohort_description) || `Cohort ${index + 1}`
          return <div key={`${cohort.cohort_name ?? index}-${index}`} className="rounded-2xl border border-border p-5">
            <div className="font-semibold">{name}</div>
            {cohort.narrative ? <p className="mt-3 text-sm leading-6 text-text-secondary">{cohort.narrative}</p> : null}
            <div className="mt-5 grid grid-cols-3 gap-3 border-t border-border pt-4 text-center">
              <div><div className="text-lg font-semibold">{cohort.cohort_size ?? '—'}</div><div className="text-[11px] text-text-muted">people</div></div>
              <div><div className="text-lg font-semibold">{cohort.avg_tenure_years == null ? '—' : `${cohort.avg_tenure_years.toFixed(1)}y`}</div><div className="text-[11px] text-text-muted">average recorded tenure</div></div>
              <div><div className="text-lg font-semibold">{cohort.survival_probability_12mo == null ? '—' : `${(cohort.survival_probability_12mo * 100).toFixed(0)}%`}</div><div className="text-[11px] text-text-muted">still here after 12 months</div></div>
            </div>
            <p className="mt-4 rounded-xl bg-background-secondary p-3 text-xs leading-5 text-text-secondary">{cohort.survival_probability_12mo == null ? 'There is not enough recorded history to describe the 12-month picture.' : `In this recorded cohort history, ${(cohort.survival_probability_12mo * 100).toFixed(0)}% of the group remained beyond month 12. This describes the group history; it is not a promise about an individual.`}</p>
            <Link href={`/advisor?q=${encodeURIComponent(cohortFollowUp(name))}`} className="mt-4 inline-flex items-center gap-1.5 text-xs font-semibold text-accent">Open a workforce check <ArrowRight className="h-3.5 w-3.5" aria-hidden="true" /></Link>
          </div>
        }) : <EmptyState title="No cohort comparison available" />}</div>
      </Surface>}

      <TrustDisclosure title="How to interpret retention cohorts" summary="Cohort history, not an individual prediction">
        <p>The retention curve uses a standard time-to-event method (Kaplan–Meier). A value at 12 months describes the estimated share remaining beyond month 12 from the defined cohort origin; it is not the probability that a currently employed person will stay for the next 12 months.</p>
        <p className="mt-2">When available, the relationship model uses Cox proportional hazards to describe associations in the observed cohort. Those associations are not causal effects and are not used to rank individual employees.</p>
        {warnings.length ? <ul className="mt-3 space-y-1.5">{warnings.slice(0, 4).map((item, index) => <li key={index}>• {item}</li>)}</ul> : null}
      </TrustDisclosure>

      <StateSummary title="PeopleOS analyses cohorts, not individuals" description="Use these patterns for aggregate planning and follow-up investigation. Individual retention ranking is intentionally kept outside the product experience." tone="info" />
    </Page>
  )
}
