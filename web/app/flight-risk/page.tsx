'use client'

import { useQuery } from '@tanstack/react-query'
import Link from 'next/link'
import { api } from '@/lib/api-client'
import { EmptyState, MetricCard, Page, PageHeader, SectionHeader, StateSummary, StatusBadge, Surface } from '@/components/ui'
import { Activity, ArrowRight, Brain, Database, ShieldCheck, Target, Users } from 'lucide-react'
import type { ModelMetrics, FeatureImportance, PredictionSummary } from '@/types/api'

interface PlatformStatus {
  workspace?: { active_model?: boolean }
}

export default function RetentionSignalsPage() {
  const platform = useQuery<PlatformStatus>({
    queryKey: ['platform', 'status'],
    queryFn: () => api.getStatus() as Promise<PlatformStatus>,
  })
  const hasActiveModel = Boolean(platform.data?.workspace?.active_model)

  const metrics = useQuery<ModelMetrics>({
    queryKey: ['predictions', 'model-metrics'],
    queryFn: () => api.predictions.getModelMetrics() as Promise<ModelMetrics>,
    retry: false,
    enabled: hasActiveModel,
  })
  const predictions = useQuery<PredictionSummary>({
    queryKey: ['predictions', 'risk'],
    queryFn: () => api.predictions.getRisk(undefined, 100) as Promise<PredictionSummary>,
    retry: false,
    enabled: hasActiveModel && Boolean(metrics.data),
  })
  const importance = useQuery<FeatureImportance>({
    queryKey: ['predictions', 'feature-importance'],
    queryFn: () => api.predictions.getFeatureImportance(8) as Promise<FeatureImportance>,
    retry: false,
    enabled: hasActiveModel && Boolean(metrics.data),
  })

  if (platform.isLoading || (hasActiveModel && metrics.isLoading)) {
    return <Page><StateSummary title="Checking predictive retention capability" description="Reading the governed model lifecycle before requesting predictive outputs." tone="info" /></Page>
  }

  if (!hasActiveModel || platform.isError || metrics.isError || !metrics.data) {
    return <Page>
      <PageHeader eyebrow="Understand · Retention Signals" title="Predictive retention signals are not active" description="PeopleOS keeps predictive risk separate from deterministic analytics. Predictive views appear only after a model has been trained, evaluated and explicitly activated." />
      <StateSummary title="Deterministic analysis remains available" description="No predictive call is made without an active governed model. You can continue with observed workforce evidence now." tone="info" />

      <div className="grid gap-6 lg:grid-cols-3">
        <Surface padding="lg">
          <div className="grid h-10 w-10 place-items-center rounded-xl bg-accent/10 text-accent"><Database className="h-5 w-5" /></div>
          <h2 className="mt-4 font-semibold">Available right now</h2>
          <p className="mt-2 text-sm leading-6 text-text-secondary">Use observed attrition share, tenure, department patterns and other deterministic evidence without waiting for a model.</p>
        </Surface>
        <Surface padding="lg">
          <div className="grid h-10 w-10 place-items-center rounded-xl bg-accent/10 text-accent"><Brain className="h-5 w-5" /></div>
          <h2 className="mt-4 font-semibold">What unlocks this view</h2>
          <p className="mt-2 text-sm leading-6 text-text-secondary">A suitable labelled dataset must pass predictive readiness checks, then a candidate model must be trained, evaluated and activated through the governed lifecycle.</p>
        </Surface>
        <Surface padding="lg">
          <div className="grid h-10 w-10 place-items-center rounded-xl bg-accent/10 text-accent"><ShieldCheck className="h-5 w-5" /></div>
          <h2 className="mt-4 font-semibold">Why the gate matters</h2>
          <p className="mt-2 text-sm leading-6 text-text-secondary">Predictive scores can look authoritative. PeopleOS keeps them unavailable until model state and evaluation evidence justify showing them.</p>
        </Surface>
      </div>

      <Surface padding="md" className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div><div className="font-semibold">Continue with trustworthy evidence</div><div className="text-sm text-text-secondary">Investigate retention today or review model lifecycle details when you are ready.</div></div>
        <div className="flex flex-wrap gap-2">
          <Link href="/platform" className="inline-flex items-center gap-2 rounded-xl border border-border px-4 py-2.5 text-sm font-semibold text-text-primary transition hover:bg-background-secondary"><ShieldCheck className="h-4 w-4" />Trust Center</Link>
          <Link href="/workforce-health" className="inline-flex items-center gap-2 rounded-xl bg-accent px-4 py-2.5 text-sm font-semibold text-white transition hover:opacity-90">Open Workforce Health <ArrowRight className="h-4 w-4" /></Link>
        </div>
      </Surface>
    </Page>
  }

  if (predictions.isLoading) return <Page><StateSummary title="Loading model scores" description="Reading aggregate scores for the active dataset." tone="info" /></Page>
  if (predictions.isError || !predictions.data?.distribution) return <Page><EmptyState title="Aggregate model scores are unavailable" description="No distribution can be shown until valid scores for the current dataset are available." /></Page>

  const model = metrics.data
  const distribution = predictions.data?.distribution
  const total = (distribution?.high_risk ?? 0) + (distribution?.medium_risk ?? 0) + (distribution?.low_risk ?? 0)
  const highPct = distribution?.high_risk_pct ?? 0
  const mediumPct = distribution?.medium_risk_pct ?? 0
  const lowPct = distribution?.low_risk_pct ?? 0
  const features = importance.data?.features ?? []

  return <Page>
    <PageHeader eyebrow="Understand · Retention Signals" title="Where is predictive retention pressure concentrated?" description="Aggregate predictive signals are shown with enough model context to interpret them safely, without turning model diagnostics into the main experience." />
    <StateSummary title="Retrospective model evidence" description="The employee holdout measures classification of recorded outcomes. Accuracy for future departures has not been validated. Use scores for aggregate investigation; they do not establish an employee’s probability of leaving in a defined future period." tone="info" />

    <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
      <MetricCard label="People scored" value={total.toLocaleString()} detail="Aggregate predictive coverage" icon={Users} />
      <MetricCard label="High signal" value={`${highPct.toFixed(1)}%`} detail={`${distribution?.high_risk ?? 0} people in high band`} icon={Target} tone={highPct >= 20 ? 'warning' : 'neutral'} />
      <MetricCard label="Medium signal" value={`${mediumPct.toFixed(1)}%`} detail={`${distribution?.medium_risk ?? 0} people in medium band`} icon={Activity} tone="neutral" />
    </section>

    <div className="grid gap-6 xl:grid-cols-[minmax(0,1.35fr)_minmax(300px,0.65fr)]">
      <Surface padding="lg"><SectionHeader title="Risk distribution" description="Population-level bands from the active predictive model." /><div className="mt-6 space-y-5"><DistributionRow label="High" count={distribution?.high_risk ?? 0} percent={highPct} tone="danger" /><DistributionRow label="Medium" count={distribution?.medium_risk ?? 0} percent={mediumPct} tone="warning" /><DistributionRow label="Low" count={distribution?.low_risk ?? 0} percent={lowPct} tone="success" /></div></Surface>
      <Surface padding="md"><SectionHeader title="Model context" description="Supporting qualification for the active prediction." /><div className="mt-4 space-y-2"><MetricRow label="F1" value={`${(model.f1 * 100).toFixed(1)}%`} /><MetricRow label="Precision" value={`${(model.precision * 100).toFixed(1)}%`} /><MetricRow label="Recall" value={`${(model.recall * 100).toFixed(1)}%`} /><MetricRow label="Average precision / baseline" value={`${model.average_precision?.toFixed(3) ?? '—'} / ${model.baseline_average_precision?.toFixed(3) ?? '—'}`} /><MetricRow label="Brier error / baseline" value={`${model.brier_score?.toFixed(3) ?? '—'} / ${model.baseline_brier_score?.toFixed(3) ?? '—'}`} /><MetricRow label="Reliability" value={model.reliability ?? 'Unknown'} /></div></Surface>
    </div>

    <Surface padding="lg"><SectionHeader title="Strongest model features" description="Feature importance explains model influence, not causal drivers of attrition." /><div className="mt-5 grid gap-3 md:grid-cols-2">{features.length ? features.map((item, index) => <div key={`${item.feature}-${index}`} className="flex items-center justify-between gap-4 rounded-2xl border border-border p-4"><div><div className="font-medium">{item.feature}</div><div className="text-xs text-text-muted">Relative model importance</div></div><StatusBadge tone="neutral">{typeof item.importance === 'number' ? item.importance.toFixed(3) : '—'}</StatusBadge></div>) : <EmptyState title="Feature importance is not available" description="The active model did not expose a feature-importance view." />}</div></Surface>

    <Surface padding="md" className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between"><div><div className="font-semibold">Move from prediction to evidence.</div><div className="text-sm text-text-secondary">Use People Intelligence to compare predictive pressure with deterministic workforce evidence before acting.</div></div><Link href="/advisor" className="inline-flex items-center gap-2 text-sm font-semibold text-accent"><Activity className="h-4 w-4" />Investigate retention <ArrowRight className="h-4 w-4" /></Link></Surface>
  </Page>
}

function DistributionRow({ label, count, percent, tone }: { label: string; count: number; percent: number; tone: 'danger' | 'warning' | 'success' }) {
  const bar = tone === 'danger' ? 'bg-danger' : tone === 'warning' ? 'bg-warning' : 'bg-success'
  return <div><div className="mb-2 flex items-center justify-between gap-4"><div className="flex items-center gap-2"><StatusBadge tone={tone}>{label}</StatusBadge><span className="text-sm text-text-secondary">{count.toLocaleString()} people</span></div><span className="text-sm font-semibold">{percent.toFixed(1)}%</span></div><div className="h-2 overflow-hidden rounded-full bg-background-secondary"><div className={`h-full rounded-full ${bar}`} style={{ width: `${Math.min(100, Math.max(0, percent))}%` }} /></div></div>
}

function MetricRow({ label, value }: { label: string; value: string }) { return <div className="flex items-center justify-between border-b border-border py-3 last:border-0"><span className="text-sm text-text-secondary">{label}</span><span className="font-semibold">{value}</span></div> }
