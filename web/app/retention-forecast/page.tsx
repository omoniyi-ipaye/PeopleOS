'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api-client'
import { Button, EmptyState, MetricCard, Page, PageHeader, SectionHeader, StateSummary, StatusBadge, Surface } from '@/components/ui'
import { Activity, BarChart3, Clock3, Layers, ShieldAlert, TrendingUp } from 'lucide-react'
import { RetentionCurveChart } from '@/components/charts/retention-curve-chart'
import { ForecastTab } from '@/components/diagnostics/forecast-tab'
import type { SurvivalAnalysisResult, CohortInsight } from '@/types/api'

type Tab = 'overview' | 'cohorts' | 'trends'

export default function RetentionForecastPage() {
  const [tab, setTab] = useState<Tab>('overview')
  const { data, isLoading, isError, error, refetch } = useQuery<SurvivalAnalysisResult>({ queryKey: ['survival', 'analysis'], queryFn: () => api.survival.getAnalysis() as Promise<SurvivalAnalysisResult> })

  if (isLoading) return <Page><StateSummary title="Building retention outlook" description="Estimating tenure patterns, cohort survival and forward-looking retention signals." tone="info" /></Page>
  if (isError) return <Page><EmptyState title="Retention Forecast is unavailable" description={error instanceof Error ? error.message : 'Unable to generate retention analysis.'} action={<Button onClick={() => refetch()}>Retry</Button>} /></Page>

  const summary = data?.summary
  const km = data?.kaplan_meier
  const cohorts: CohortInsight[] = data?.cohort_insights ?? []
  const warnings = data?.warnings ?? []
  const recommendations = data?.recommendations ?? []
  const retention12 = summary?.avg_12mo_risk == null ? null : 1 - summary.avg_12mo_risk

  return (
    <Page>
      <PageHeader eyebrow="Plan · Retention Forecast" title="How is retention likely to evolve?" description="A governed forecast view of survival patterns and cohort pressure. Forecasts support planning; they do not determine individual employment action." />
      <div className="flex flex-wrap gap-2">
        <Button size="sm" variant={tab === 'overview' ? 'primary' : 'secondary'} onClick={() => setTab('overview')}><Activity className="h-4 w-4" />Outlook</Button>
        <Button size="sm" variant={tab === 'cohorts' ? 'primary' : 'secondary'} onClick={() => setTab('cohorts')}><Layers className="h-4 w-4" />Cohorts</Button>
        <Button size="sm" variant={tab === 'trends' ? 'primary' : 'secondary'} onClick={() => setTab('trends')}><TrendingUp className="h-4 w-4" />Trends</Button>
      </div>

      {warnings.length ? <StateSummary title="Forecast limitations" description={warnings.slice(0, 2).join(' · ')} tone="warning" /> : null}

      {tab === 'overview' && <>
        <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <MetricCard label="Median tenure" value={summary?.median_tenure == null ? '—' : `${summary.median_tenure.toFixed(1)}y`} detail="Observed workforce tenure" icon={Clock3} />
          <MetricCard label="12-month retention" value={retention12 == null ? '—' : `${(retention12 * 100).toFixed(0)}%`} detail="Estimated probability from current model" icon={Activity} tone={retention12 == null ? 'neutral' : retention12 >= .8 ? 'success' : retention12 >= .7 ? 'warning' : 'danger'} />
          <MetricCard label="High-pressure population" value={(summary?.high_risk_count ?? 0).toLocaleString()} detail="Aggregate forecast signal" icon={ShieldAlert} tone={(summary?.high_risk_count ?? 0) ? 'warning' : 'success'} />
          <MetricCard label="Cohorts analysed" value={cohorts.length.toLocaleString()} detail="Distinct retention patterns" icon={Layers} />
        </section>

        <Surface padding="lg">
          <SectionHeader title="Retention curve" description="Kaplan–Meier survival estimate for the workforce represented in the current dataset." />
          <div className="mt-6">
            {km?.overall?.survival_function?.length ? <RetentionCurveChart data={km.overall.survival_function} /> : <EmptyState title="No survival curve available" description="The current dataset does not provide enough time-to-event information." />}
          </div>
        </Surface>

        {recommendations.length ? <Surface padding="lg"><SectionHeader title="Planning questions" description="Use these as prompts for systemic investigation, not automatic interventions." /><div className="mt-5 grid gap-3 md:grid-cols-2">{recommendations.slice(0, 6).map((item, index) => <div key={`${item}-${index}`} className="rounded-xl border border-border p-4 text-sm leading-6 text-text-secondary"><span className="mr-2 font-semibold text-accent">0{index + 1}</span>{item}</div>)}</div></Surface> : null}
      </>}

      {tab === 'cohorts' && <Surface padding="lg">
        <SectionHeader title="Retention cohorts" description="Compare group-level survival patterns while preserving uncertainty and avoiding individual ranking." />
        <div className="mt-5 grid gap-4 lg:grid-cols-2">{cohorts.length ? cohorts.map((cohort, index) => <div key={`${cohort.cohort_name ?? index}-${index}`} className="rounded-2xl border border-border p-5"><div className="flex items-start justify-between gap-4"><div><div className="font-semibold">{cohort.cohort_name || `Cohort ${index + 1}`}</div><div className="mt-1 text-xs text-text-muted">{cohort.cohort_description || 'Shared retention characteristics'}</div></div><StatusBadge tone={cohort.risk_level === 'High' ? 'danger' : cohort.risk_level === 'Medium' ? 'warning' : 'success'}>{cohort.risk_level || 'Observed'}</StatusBadge></div>{cohort.insight ? <p className="mt-4 text-sm leading-6 text-text-secondary">{cohort.insight}</p> : null}<div className="mt-5 grid grid-cols-3 gap-3 border-t border-border pt-4 text-center"><div><div className="text-lg font-semibold">{cohort.cohort_size ?? '—'}</div><div className="text-[11px] text-text-muted">people</div></div><div><div className="text-lg font-semibold">{(cohort.median_tenure ?? cohort.avg_tenure_years) == null ? '—' : `${(cohort.median_tenure ?? cohort.avg_tenure_years)!.toFixed(1)}y`}</div><div className="text-[11px] text-text-muted">tenure</div></div><div><div className="text-lg font-semibold">{cohort.survival_probability_12mo == null ? '—' : `${(cohort.survival_probability_12mo * 100).toFixed(0)}%`}</div><div className="text-[11px] text-text-muted">12m retention</div></div></div></div>) : <EmptyState title="No cohort analysis available" />}</div>
      </Surface>}

      {tab === 'trends' && <Surface padding="lg"><SectionHeader title="Workforce trends" description="Historical movement used as context for the forecast." /><div className="mt-6"><ForecastTab /></div></Surface>}

      <StateSummary title="Governance boundary" description="Retention forecasts may prioritise investigation and workforce planning. They must not be used as the sole basis for termination, discipline, compensation reduction or other consequential individual decisions." tone="info" />
    </Page>
  )
}
