'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api-client'
import { Button, EmptyState, MetricCard, Page, PageHeader, SectionHeader, StateSummary, Surface, TrustDisclosure } from '@/components/ui'
import { Activity, Clock3, Layers, ShieldAlert } from 'lucide-react'
import { RetentionCurveChart } from '@/components/charts/retention-curve-chart'
import type { SurvivalAnalysisResult, CohortInsight } from '@/types/api'

type Tab = 'overview' | 'cohorts'

function humanize(value?: string | null) {
  if (!value) return null
  return value.replaceAll('_', ' ').replace(/([a-z0-9])([A-Z])/g, '$1 $2').replace(/\b\w/g, char => char.toUpperCase())
}

export default function RetentionForecastPage() {
  const [tab, setTab] = useState<Tab>('overview')
  const { data, isLoading, isError, error, refetch } = useQuery<SurvivalAnalysisResult>({
    queryKey: ['survival', 'analysis'],
    queryFn: () => api.survival.getAnalysis() as Promise<SurvivalAnalysisResult>,
  })

  const header = <PageHeader eyebrow="Insights · Retention" title="How does retention change with tenure?" description="Explore recorded cohort retention patterns over time and compare the groups your current data can support." />
  if (isLoading) return <Page>{header}<StateSummary title="Preparing retention insights" description="Reading recorded tenure and departure outcomes across your workforce." tone="info" /></Page>
  if (isError) return <Page>{header}<EmptyState title="Retention cohort analysis is unavailable" description={error instanceof Error ? error.message : 'Unable to generate cohort retention analysis.'} action={<Button onClick={() => refetch()}>Retry</Button>} /></Page>

  const summary = data?.summary
  const km = data?.kaplan_meier
  const cohorts: CohortInsight[] = data?.cohort_insights ?? []
  const warnings = data?.warnings ?? []

  return (
    <Page>
      {header}

      <div className="flex flex-wrap gap-2">
        <Button size="sm" variant={tab === 'overview' ? 'primary' : 'secondary'} onClick={() => setTab('overview')}><Activity className="h-4 w-4" />Retention curve</Button>
        <Button size="sm" variant={tab === 'cohorts' ? 'primary' : 'secondary'} onClick={() => setTab('cohorts')}><Layers className="h-4 w-4" />Cohorts</Button>
      </div>

      {tab === 'overview' && <>
        <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <MetricCard label="Cohort midpoint tenure" value={summary?.median_tenure == null ? '—' : `${summary.median_tenure.toFixed(1)}y`} detail="Estimated tenure point where the cohort curve reaches 50%, if observed" icon={Clock3} />
          <MetricCard label="Recorded attrition" value={summary?.overall_attrition_rate == null ? '—' : `${(summary.overall_attrition_rate * 100).toFixed(1)}%`} detail="Known outcomes marked as departed" icon={Activity} />
          <MetricCard label="Relationship model" value={summary?.cox_model_fitted ? 'Available' : 'Not available'} detail="Optional cohort-level association model" icon={ShieldAlert} tone={summary?.cox_model_fitted ? 'info' : 'neutral'} />
          <MetricCard label="Cohorts compared" value={cohorts.length.toLocaleString()} detail="Groups meeting analysis requirements" icon={Layers} />
        </section>

        <Surface padding="lg">
          <SectionHeader title="Cohort retention curve" description="Estimated share remaining beyond each tenure point from the defined cohort start." />
          <div className="mt-6">{km?.overall?.survival_function?.length ? <RetentionCurveChart data={km.overall.survival_function} /> : <EmptyState title="No retention curve available" description="The current dataset does not provide enough valid time-to-event information." />}</div>
        </Surface>
      </>}

      {tab === 'cohorts' && <Surface padding="lg">
        <SectionHeader title="Cohort comparison" description="Compare recorded time-to-exit patterns across the groups available in your data." />
        <div className="mt-5 grid gap-4 lg:grid-cols-2">{cohorts.length ? cohorts.map((cohort, index) => {
          const name = humanize(cohort.cohort_name) || humanize(cohort.cohort_description) || `Cohort ${index + 1}`
          return <div key={`${cohort.cohort_name ?? index}-${index}`} className="rounded-2xl border border-border p-5">
            <div className="font-semibold">{name}</div>
            {cohort.narrative ? <p className="mt-3 text-sm leading-6 text-text-secondary">{cohort.narrative}</p> : null}
            <div className="mt-5 grid grid-cols-3 gap-3 border-t border-border pt-4 text-center">
              <div><div className="text-lg font-semibold">{cohort.cohort_size ?? '—'}</div><div className="text-[11px] text-text-muted">people</div></div>
              <div><div className="text-lg font-semibold">{cohort.avg_tenure_years == null ? '—' : `${cohort.avg_tenure_years.toFixed(1)}y`}</div><div className="text-[11px] text-text-muted">mean recorded tenure</div></div>
              <div><div className="text-lg font-semibold">{cohort.survival_probability_12mo == null ? '—' : `${(cohort.survival_probability_12mo * 100).toFixed(0)}%`}</div><div className="text-[11px] text-text-muted">remain beyond month 12</div></div>
            </div>
          </div>
        }) : <EmptyState title="No cohort comparison available" />}</div>
      </Surface>}

      <TrustDisclosure title="How to interpret retention cohorts" summary="Cohort history, not an individual prediction">
        <p>The retention curve uses Kaplan–Meier time-to-event estimation. A value at 12 months describes the estimated share remaining beyond month 12 from the defined cohort origin; it is not the probability that a currently employed person will stay for the next 12 months.</p>
        <p className="mt-2">When available, the relationship model uses Cox proportional hazards to describe associations in the observed cohort. Those associations are not causal effects and are not used to rank individual employees.</p>
        {warnings.length ? <ul className="mt-3 space-y-1.5">{warnings.slice(0, 4).map((item, index) => <li key={index}>• {item}</li>)}</ul> : null}
      </TrustDisclosure>

      <StateSummary title="PeopleOS analyses cohorts, not individuals" description="Use these patterns for aggregate planning and follow-up investigation. Individual retention ranking is intentionally kept outside the product experience." tone="info" />
    </Page>
  )
}
