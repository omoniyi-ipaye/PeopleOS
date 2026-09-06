'use client'

import { useQuery } from '@tanstack/react-query'
import Link from 'next/link'
import { api } from '@/lib/api-client'
import { EmptyState, MetricCard, Page, PageHeader, SectionHeader, StateSummary, StatusBadge, Surface } from '@/components/ui'
import { Activity, ArrowRight, Brain, ShieldCheck, Target, Users } from 'lucide-react'
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
      <PageHeader eyebrow="Understand · Retention Signals" title="Predictive retention signals are not active" description="PeopleOS keeps predictive risk separate from deterministic analytics. Train and activate a governed model before using this surface." />
      <StateSummary title="No active predictive model" description="Workforce Health and People Intelligence remain available without a model. Predictive retention does not silently train or call prediction endpoints until lifecycle state confirms an active model." tone="warning" />
      <Surface padding="lg" className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between"><div><div className="font-semibold">Need retention evidence now?</div><div className="text-sm text-text-secondary">Use deterministic turnover, tenure and department signals while the predictive lifecycle is inactive.</div></div><Link href="/workforce-health" className="inline-flex items-center gap-2 text-sm font-semibold text-accent">Open Workforce Health <ArrowRight className="h-4 w-4" /></Link></Surface>
    </Page>
  }

  const model = metrics.data
  const distribution = predictions.data?.distribution
  const total = (distribution?.high_risk ?? 0) + (distribution?.medium_risk ?? 0) + (distribution?.low_risk ?? 0)
  const highPct = distribution?.high_risk_pct ?? 0
  const mediumPct = distribution?.medium_risk_pct ?? 0
  const lowPct = distribution?.low_risk_pct ?? 0
  const features = importance.data?.features ?? []

  return <Page>
    <PageHeader eyebrow="Understand · Retention Signals" title="Where is predictive retention pressure concentrated?" description="Aggregate predictive signals, model fitness and strongest model features are shown together so risk is never read without model context." />
    <StateSummary title="Predictive signal, not an employment decision" description="Retention scores may help prioritise systemic investigation. They must not be used as the sole basis for termination, discipline, demotion, pay reduction or other consequential individual action." tone="info" />

    <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <MetricCard label="Model quality" value={`${(model.f1 * 100).toFixed(0)}%`} detail="F1 score" icon={Brain} tone={model.f1 >= .7 ? 'success' : 'warning'} />
      <MetricCard label="People scored" value={total.toLocaleString()} detail="Aggregate predictive coverage" icon={Users} />
      <MetricCard label="High signal" value={`${highPct.toFixed(1)}%`} detail={`${distribution?.high_risk ?? 0} people in high band`} icon={Target} tone={highPct >= 20 ? 'warning' : 'neutral'} />
      <MetricCard label="Model reliability" value={model.reliability ?? 'Unknown'} detail={model.best_model ?? 'Active predictive model'} icon={ShieldCheck} tone={model.reliability === 'High' ? 'success' : 'info'} />
    </section>

    <div className="grid gap-6 xl:grid-cols-[minmax(0,1.2fr)_minmax(320px,0.8fr)]">
      <Surface padding="lg"><SectionHeader title="Risk distribution" description="Population-level bands from the active predictive model." /><div className="mt-6 space-y-5"><DistributionRow label="High" count={distribution?.high_risk ?? 0} percent={highPct} tone="danger" /><DistributionRow label="Medium" count={distribution?.medium_risk ?? 0} percent={mediumPct} tone="warning" /><DistributionRow label="Low" count={distribution?.low_risk ?? 0} percent={lowPct} tone="success" /></div></Surface>
      <Surface padding="lg"><SectionHeader title="Model fitness" description="Quality metrics belong beside the predictions they qualify." /><div className="mt-5 space-y-3"><MetricRow label="Accuracy" value={`${(model.accuracy * 100).toFixed(1)}%`} /><MetricRow label="Precision" value={`${(model.precision * 100).toFixed(1)}%`} /><MetricRow label="Recall" value={`${(model.recall * 100).toFixed(1)}%`} /><MetricRow label="F1" value={`${(model.f1 * 100).toFixed(1)}%`} /></div></Surface>
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
