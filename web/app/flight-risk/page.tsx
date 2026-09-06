'use client'

import Link from 'next/link'
import { useQuery } from '@tanstack/react-query'
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  Brain,
  CheckCircle2,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Target,
  Users,
} from 'lucide-react'

interface PlatformStatus {
  data?: { loaded?: boolean; row_count?: number }
  workspace?: { active_model?: boolean; active_dataset?: boolean }
}

interface ModelMetrics {
  accuracy: number
  precision: number
  recall: number
  f1: number
  roc_auc?: number | null
  best_model: string
  reliability: string
  warnings?: string[] | null
}

interface RiskDistribution {
  high_risk: number
  medium_risk: number
  low_risk: number
  total: number
  high_risk_pct: number
  medium_risk_pct: number
  low_risk_pct: number
}

interface PredictionsResponse {
  distribution: RiskDistribution
  model_metrics: ModelMetrics
}

export default function RetentionSignalsPage() {
  const { data: status, isLoading: statusLoading } = useQuery<PlatformStatus>({
    queryKey: ['platform', 'status'],
    queryFn: async () => {
      const response = await fetch('/api/status')
      if (!response.ok) throw new Error('PeopleOS status is unavailable.')
      return response.json()
    },
  })

  const modelActive = Boolean(status?.workspace?.active_model)

  const { data: predictions, isLoading: predictionLoading, error } = useQuery<PredictionsResponse>({
    queryKey: ['retention', 'signals'],
    queryFn: async () => {
      const response = await fetch('/api/predictions/risk?limit=1')
      if (!response.ok) {
        const body = await response.json().catch(() => ({}))
        throw new Error(body.detail ?? 'Retention signals are unavailable.')
      }
      return response.json()
    },
    enabled: modelActive,
  })

  if (statusLoading) {
    return <div className="grid min-h-[55vh] place-items-center text-sm text-slate-500">Checking predictive capability…</div>
  }

  if (!status?.data?.loaded) {
    return <UnavailableState title="Retention signals need workforce data first." detail="Activate a dataset before evaluating whether predictive retention modelling is appropriate." action="Add workforce data" href="/upload" />
  }

  if (!modelActive) {
    return (
      <div className="mx-auto max-w-5xl space-y-6 pb-10">
        <PageHeading />
        <section className="rounded-[30px] border border-slate-200 bg-white p-8 shadow-sm dark:border-white/10 dark:bg-slate-900 md:p-10">
          <div className="grid h-14 w-14 place-items-center rounded-2xl bg-violet-100 text-violet-600 dark:bg-violet-500/10 dark:text-violet-300"><Brain className="h-6 w-6" /></div>
          <div className="mt-6 max-w-2xl">
            <div className="text-xs font-bold uppercase tracking-[0.16em] text-slate-400">Predictive lifecycle</div>
            <h2 className="mt-2 text-2xl font-semibold text-slate-950 dark:text-white">No predictive model is active—and that is a valid state.</h2>
            <p className="mt-3 text-sm leading-6 text-slate-600 dark:text-slate-400">PeopleOS can still analyse observed turnover, tenure and workforce health deterministically. Predictive retention signals should only appear after a model has been trained, evaluated, accepted as a candidate and deliberately activated.</p>
          </div>
          <div className="mt-7 flex flex-wrap gap-3">
            <Link href="/workforce-health" className="inline-flex items-center gap-2 rounded-xl bg-slate-950 px-4 py-2.5 text-sm font-semibold text-white dark:bg-white dark:text-slate-950">Use observed workforce evidence <ArrowRight className="h-4 w-4" /></Link>
            <Link href="/platform" className="inline-flex items-center gap-2 rounded-xl border border-slate-200 px-4 py-2.5 text-sm font-semibold text-slate-700 dark:border-white/10 dark:text-slate-200">Review model lifecycle</Link>
          </div>
        </section>
      </div>
    )
  }

  if (predictionLoading) {
    return <div className="grid min-h-[55vh] place-items-center text-sm text-slate-500">Loading governed retention signals…</div>
  }

  if (error || !predictions) {
    return <UnavailableState title="Retention signals could not be loaded." detail={error instanceof Error ? error.message : 'The active model did not return a valid predictive distribution.'} action="Open Trust Center" href="/platform" />
  }

  const distribution = predictions.distribution
  const metrics = predictions.model_metrics

  return (
    <div className="mx-auto max-w-6xl space-y-6 pb-10">
      <PageHeading />

      <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <Metric icon={ShieldAlert} label="Higher signal" value={`${distribution.high_risk_pct.toFixed(1)}%`} detail={`${distribution.high_risk.toLocaleString()} people`} tone="attention" />
        <Metric icon={Activity} label="Medium signal" value={`${distribution.medium_risk_pct.toFixed(1)}%`} detail={`${distribution.medium_risk.toLocaleString()} people`} tone="watch" />
        <Metric icon={CheckCircle2} label="Lower signal" value={`${distribution.low_risk_pct.toFixed(1)}%`} detail={`${distribution.low_risk.toLocaleString()} people`} tone="stable" />
        <Metric icon={Target} label="Model reliability" value={metrics.reliability} detail={`${(metrics.f1 * 100).toFixed(1)}% F1`} tone="neutral" />
      </section>

      <section className="grid gap-6 lg:grid-cols-[minmax(0,1.2fr)_minmax(320px,0.8fr)]">
        <div className="rounded-[28px] border border-slate-200 bg-white p-6 dark:border-white/10 dark:bg-slate-900">
          <div className="text-xs font-bold uppercase tracking-[0.16em] text-slate-400">Predictive distribution</div>
          <h2 className="mt-1 text-xl font-semibold text-slate-950 dark:text-white">Use the model to locate questions, not make employment decisions.</h2>
          <div className="mt-6 space-y-5">
            <SignalBar label="Higher retention-risk signal" value={distribution.high_risk_pct} count={distribution.high_risk} tone="bg-rose-500" />
            <SignalBar label="Medium retention-risk signal" value={distribution.medium_risk_pct} count={distribution.medium_risk} tone="bg-amber-500" />
            <SignalBar label="Lower retention-risk signal" value={distribution.low_risk_pct} count={distribution.low_risk} tone="bg-emerald-500" />
          </div>
          <div className="mt-6 rounded-2xl bg-slate-50 p-4 text-sm leading-6 text-slate-600 dark:bg-white/[0.04] dark:text-slate-400">A predictive score is not a fact about an individual and should never be used as the sole basis for termination, discipline, demotion, compensation change or other consequential action.</div>
        </div>

        <div className="space-y-6">
          <section className="rounded-[28px] border border-slate-200 bg-white p-6 dark:border-white/10 dark:bg-slate-900">
            <div className="flex items-center gap-2"><ShieldCheck className="h-5 w-5 text-violet-500" /><h2 className="font-semibold text-slate-950 dark:text-white">Model fitness</h2></div>
            <div className="mt-4 space-y-3">
              <FitnessRow label="Accuracy" value={`${(metrics.accuracy * 100).toFixed(1)}%`} />
              <FitnessRow label="Precision" value={`${(metrics.precision * 100).toFixed(1)}%`} />
              <FitnessRow label="Recall" value={`${(metrics.recall * 100).toFixed(1)}%`} />
              <FitnessRow label="F1 balance" value={`${(metrics.f1 * 100).toFixed(1)}%`} />
              <FitnessRow label="Model family" value={metrics.best_model.replaceAll('_', ' ')} />
            </div>
            <Link href="/platform" className="mt-5 inline-flex items-center gap-1 text-xs font-semibold text-violet-600 dark:text-violet-300">Review full model provenance <ArrowRight className="h-3.5 w-3.5" /></Link>
          </section>

          <section className="rounded-[28px] border border-violet-200 bg-violet-50 p-6 dark:border-violet-500/20 dark:bg-violet-500/[0.05]">
            <div className="flex items-center gap-2 text-violet-800 dark:text-violet-300"><Sparkles className="h-5 w-5" /><h2 className="font-semibold">Next investigation</h2></div>
            <p className="mt-2 text-sm leading-6 text-violet-800/80 dark:text-violet-300/80">Ask People Intelligence whether the observed turnover pattern supports the same concern as the predictive distribution.</p>
            <Link href="/advisor" className="mt-4 inline-flex items-center gap-2 rounded-xl bg-violet-600 px-4 py-2.5 text-sm font-semibold text-white">Investigate with evidence <ArrowRight className="h-4 w-4" /></Link>
          </section>
        </div>
      </section>

      {metrics.warnings && metrics.warnings.length > 0 && (
        <section className="rounded-[28px] border border-amber-200 bg-amber-50 p-5 dark:border-amber-500/20 dark:bg-amber-500/[0.05]">
          <div className="flex items-center gap-2 text-amber-800 dark:text-amber-300"><AlertTriangle className="h-5 w-5" /><h2 className="font-semibold">Model limitations</h2></div>
          <ul className="mt-3 space-y-2 text-sm leading-6 text-amber-800/80 dark:text-amber-300/80">{metrics.warnings.map((warning) => <li key={warning}>• {warning}</li>)}</ul>
        </section>
      )}
    </div>
  )
}

function PageHeading() {
  return <section><div className="mb-2 text-xs font-bold uppercase tracking-[0.18em] text-violet-600 dark:text-violet-300">Understand · Retention</div><h1 className="text-3xl font-semibold tracking-tight text-slate-950 dark:text-white">Retention Signals</h1><p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600 dark:text-slate-400">Separate observed retention evidence from model-generated signals, and use predictions only to guide further investigation.</p></section>
}

function Metric({ icon: Icon, label, value, detail, tone }: { icon: React.ElementType; label: string; value: string; detail: string; tone: 'attention' | 'watch' | 'stable' | 'neutral' }) {
  const iconTone = tone === 'attention' ? 'text-rose-500' : tone === 'watch' ? 'text-amber-500' : tone === 'stable' ? 'text-emerald-500' : 'text-violet-500'
  return <div className="rounded-2xl border border-slate-200 bg-white p-5 dark:border-white/10 dark:bg-slate-900"><Icon className={`h-4 w-4 ${iconTone}`} /><div className="mt-3 text-2xl font-semibold text-slate-950 dark:text-white">{value}</div><div className="mt-1 text-xs font-semibold text-slate-600 dark:text-slate-300">{label}</div><div className="mt-1 text-xs text-slate-400">{detail}</div></div>
}

function SignalBar({ label, value, count, tone }: { label: string; value: number; count: number; tone: string }) {
  return <div><div className="mb-2 flex items-center justify-between gap-3 text-sm"><span className="font-medium text-slate-700 dark:text-slate-300">{label}</span><span className="text-xs text-slate-500">{count.toLocaleString()} · {value.toFixed(1)}%</span></div><div className="h-3 overflow-hidden rounded-full bg-slate-100 dark:bg-white/10"><div className={`h-full rounded-full ${tone}`} style={{ width: `${Math.min(100, value)}%` }} /></div></div>
}

function FitnessRow({ label, value }: { label: string; value: string }) {
  return <div className="flex items-center justify-between gap-4 border-b border-slate-100 py-2.5 text-sm last:border-0 dark:border-white/5"><span className="text-slate-500 dark:text-slate-400">{label}</span><span className="font-semibold capitalize text-slate-800 dark:text-slate-200">{value}</span></div>
}

function UnavailableState({ title, detail, action, href }: { title: string; detail: string; action: string; href: string }) {
  return <div className="mx-auto grid min-h-[65vh] max-w-3xl place-items-center"><section className="w-full rounded-[30px] border border-slate-200 bg-white p-8 text-center dark:border-white/10 dark:bg-slate-900"><div className="mx-auto grid h-14 w-14 place-items-center rounded-2xl bg-amber-100 text-amber-700 dark:bg-amber-500/10 dark:text-amber-300"><Users className="h-6 w-6" /></div><h1 className="mt-5 text-2xl font-semibold text-slate-950 dark:text-white">{title}</h1><p className="mx-auto mt-2 max-w-xl text-sm leading-6 text-slate-500 dark:text-slate-400">{detail}</p><Link href={href} className="mt-6 inline-flex items-center gap-2 rounded-xl bg-slate-950 px-4 py-2.5 text-sm font-semibold text-white dark:bg-white dark:text-slate-950">{action}<ArrowRight className="h-4 w-4" /></Link></section></div>
}
