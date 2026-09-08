'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api-client'
import { Button, EmptyState, MetricCard, Page, PageHeader, SectionHeader, StateSummary, Surface } from '@/components/ui'
import { Activity, Clock3, Layers, ShieldAlert } from 'lucide-react'
import { RetentionCurveChart } from '@/components/charts/retention-curve-chart'
import type { SurvivalAnalysisResult, CohortInsight } from '@/types/api'

type Tab = 'overview' | 'cohorts'

export default function RetentionForecastPage() {
  const [tab, setTab] = useState<Tab>('overview')
  const { data, isLoading, isError, error, refetch } = useQuery<SurvivalAnalysisResult>({
    queryKey: ['survival', 'analysis'],
    queryFn: () => api.survival.getAnalysis() as Promise<SurvivalAnalysisResult>,
  })

  const header = <PageHeader eyebrow="Plan · Retention Cohort Survival" title="How does observed workforce survival vary across tenure and cohorts?" description="Kaplan–Meier and Cox outputs describe cohort time-to-event patterns under explicit assumptions. They are not individual next-period forecasts or causal intervention effects." />
  if (isLoading) return <Page>{header}<StateSummary title="Building cohort survival view" description="Estimating observed tenure-to-event patterns and aggregate cohort curves." tone="info" /></Page>
  if (isError) return <Page>{header}<EmptyState title="Retention survival analysis is unavailable" description={error instanceof Error ? error.message : 'Unable to generate cohort survival analysis.'} action={<Button onClick={() => refetch()}>Retry</Button>} /></Page>

  const summary = data?.summary
  const km = data?.kaplan_meier
  const cohorts: CohortInsight[] = data?.cohort_insights ?? []
  const warnings = data?.warnings ?? []

  return (
    <Page>
      {header}
      <StateSummary title="Survival probability is cumulative from the cohort origin" description="A Kaplan–Meier value at 12 months estimates the share surviving beyond 12 months from the defined time origin. It is not the probability that a currently employed person stays for the next 12 months." tone="info" />
      <div className="flex flex-wrap gap-2"><Button size="sm" variant={tab === 'overview' ? 'primary' : 'secondary'} onClick={() => setTab('overview')}><Activity className="h-4 w-4" />Cohort curve</Button><Button size="sm" variant={tab === 'cohorts' ? 'primary' : 'secondary'} onClick={() => setTab('cohorts')}><Layers className="h-4 w-4" />Cohorts</Button></div>

      {warnings.length ? <StateSummary title="Model limitations" description={warnings.slice(0, 3).join(' · ')} tone="info" /> : null}

      {tab === 'overview' && <>
        <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <MetricCard label="Median survival tenure" value={summary?.median_tenure == null ? '—' : `${summary.median_tenure.toFixed(1)}y`} detail="Cohort time point where estimated survival reaches 50%, if observed" icon={Clock3} />
          <MetricCard label="Observed attrition share" value={summary?.overall_attrition_rate == null ? '—' : `${(summary.overall_attrition_rate * 100).toFixed(1)}%`} detail="Recorded outcome share in the analysis population" icon={Activity} />
          <MetricCard label="Cox model" value={summary?.cox_model_fitted ? 'Available' : 'Not available'} detail="Association model subject to proportional-hazards and sample assumptions" icon={ShieldAlert} tone={summary?.cox_model_fitted ? 'info' : 'neutral'} />
          <MetricCard label="Cohorts analysed" value={cohorts.length.toLocaleString()} detail="Aggregate comparison groups meeting analysis requirements" icon={Layers} />
        </section>

        <Surface padding="lg"><SectionHeader title="Observed cohort survival curve" description="Kaplan–Meier estimate from tenure/time-to-event data represented in the current dataset." /><div className="mt-6">{km?.overall?.survival_function?.length ? <RetentionCurveChart data={km.overall.survival_function} /> : <EmptyState title="No survival curve available" description="The current dataset does not provide enough valid time-to-event information." />}</div></Surface>
      </>}

      {tab === 'cohorts' && <Surface padding="lg">
        <SectionHeader title="Cohort survival evidence" description="Descriptive group-level time-to-event patterns. Differences may reflect composition and confounding." />
        <div className="mt-5 grid gap-4 lg:grid-cols-2">{cohorts.length ? cohorts.map((cohort, index) => <div key={`${cohort.cohort_name ?? index}-${index}`} className="rounded-2xl border border-border p-5"><div className="font-semibold">{cohort.cohort_name || cohort.cohort_description || `Cohort ${index + 1}`}</div><div className="mt-1 text-xs text-text-muted">{cohort.cohort_description || 'Aggregate cohort'}</div>{cohort.narrative ? <p className="mt-4 text-sm leading-6 text-text-secondary">{cohort.narrative}</p> : null}<div className="mt-5 grid grid-cols-3 gap-3 border-t border-border pt-4 text-center"><div><div className="text-lg font-semibold">{cohort.cohort_size ?? '—'}</div><div className="text-[11px] text-text-muted">people</div></div><div><div className="text-lg font-semibold">{cohort.avg_tenure_years == null ? '—' : `${cohort.avg_tenure_years!.toFixed(1)}y`}</div><div className="text-[11px] text-text-muted">mean observed tenure</div></div><div><div className="text-lg font-semibold">{cohort.survival_probability_12mo == null ? '—' : `${(cohort.survival_probability_12mo * 100).toFixed(0)}%`}</div><div className="text-[11px] text-text-muted">survival beyond month 12</div></div></div></div>) : <EmptyState title="No cohort analysis available" />}</div>
      </Surface>}

      <StateSummary title="Use boundary" description="Cohort survival supports aggregate planning and hypothesis generation. Individual survival ranking is not exposed, and hazard associations are not causal effects." tone="info" />
    </Page>
  )
}
