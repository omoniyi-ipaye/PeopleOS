'use client'

import { useCallback, useEffect, useState } from 'react'
import Link from 'next/link'
import { Activity, ArrowRight, Brain, CheckCircle2, Database, RefreshCw, ShieldCheck, Wrench } from 'lucide-react'

interface DatasetRecord { dataset_id: string; version: number; source_name: string; row_count: number; state: string; created_at: string }
interface ModelRecord { model_id: string; version: number; model_family: string; state: string; metrics: Record<string, number> }
interface Workspace { workspace_id: string; name: string; active_dataset_id?: string | null; active_model_id?: string | null; datasets: DatasetRecord[]; models: ModelRecord[]; sessions: Array<{ session_id: string; state: string }> }
interface Health { status: string; checks: Array<{ id: string; healthy: boolean }>; interrupted_jobs?: string[]; adaptation_level?: string; autonomous_recovery_envelope?: string[]; governed_only?: string[] }
interface Fitness { status: string; checks: Record<string, boolean>; observed: { dataset_age_days?: number | null; model_age_days?: number | null; model_auc?: number | null } }
interface Actor { actor_id: string; role: string; permissions: string[] }

function isModelCheck(key: string) {
  return key.includes('model')
}

export default function TrustCenterPage() {
  const [workspace, setWorkspace] = useState<Workspace | null>(null)
  const [health, setHealth] = useState<Health | null>(null)
  const [fitness, setFitness] = useState<Fitness | null>(null)
  const [actor, setActor] = useState<Actor | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [advanced, setAdvanced] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const responses = await Promise.all([
        fetch('/api/platform/workspaces/local'),
        fetch('/api/platform/health'),
        fetch('/api/platform/workspaces/local/fitness'),
        fetch('/api/platform/me'),
      ])
      if (!responses.every((response) => response.ok)) throw new Error('PeopleOS trust state is not available.')
      setWorkspace(await responses[0].json())
      setHealth(await responses[1].json())
      setFitness(await responses[2].json())
      setActor(await responses[3].json())
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to load trust state')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { void load() }, [load])

  if (loading) return <div className="grid min-h-[55vh] place-items-center text-sm text-slate-500">Checking PeopleOS trust state…</div>
  if (error) return <div className="rounded-2xl border border-red-200 bg-red-50 p-5 text-sm text-red-700 dark:border-red-500/20 dark:bg-red-500/[0.05] dark:text-red-300">{error}</div>

  const activeDataset = workspace?.datasets.find((item) => item.dataset_id === workspace.active_dataset_id)
  const activeModel = workspace?.models.find((item) => item.model_id === workspace.active_model_id)
  const relevantChecks = fitness ? Object.entries(fitness.checks).filter(([key]) => activeModel || !isModelCheck(key)) : []
  const failedChecks = relevantChecks.filter(([, passed]) => !passed)
  const trustHealthy = health?.status === 'healthy' && failedChecks.length === 0

  return (
    <div className="mx-auto max-w-6xl space-y-6 pb-10">
      <section className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
        <div>
          <div className="mb-2 text-xs font-bold uppercase tracking-[0.18em] text-violet-600 dark:text-violet-300">Govern · Trust Center</div>
          <h1 className="text-3xl font-semibold tracking-tight text-slate-950 dark:text-white">Can I trust this analysis?</h1>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600 dark:text-slate-400">See the data source, model state, access boundary and recovery controls behind every PeopleOS answer.</p>
        </div>
        <button type="button" onClick={() => void load()} className="inline-flex items-center justify-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 transition hover:bg-slate-50 dark:border-white/10 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-white/5"><RefreshCw className="h-4 w-4" /> Refresh</button>
      </section>

      <section className={`rounded-[30px] border p-6 md:p-7 ${trustHealthy ? 'border-emerald-200 bg-emerald-50/60 dark:border-emerald-500/20 dark:bg-emerald-500/[0.04]' : 'border-amber-200 bg-amber-50/60 dark:border-amber-500/20 dark:bg-amber-500/[0.04]'}`}>
        <div className="flex items-start gap-4">
          <div className={`grid h-12 w-12 shrink-0 place-items-center rounded-2xl ${trustHealthy ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-300' : 'bg-amber-100 text-amber-700 dark:bg-amber-500/10 dark:text-amber-300'}`}><ShieldCheck className="h-6 w-6" /></div>
          <div>
            <div className="text-xs font-bold uppercase tracking-[0.16em] text-slate-500 dark:text-slate-400">Current trust verdict</div>
            <h2 className="mt-1 text-2xl font-semibold text-slate-950 dark:text-white">{trustHealthy ? 'Core evidence path is healthy' : 'Some capabilities need attention'}</h2>
            <p className="mt-2 text-sm leading-6 text-slate-600 dark:text-slate-400">{activeModel ? 'A governed model is active in addition to deterministic evidence.' : 'No predictive model is active. Model-specific checks are not applicable; People Intelligence will use deterministic aggregate evidence and clearly mark predictive capabilities as unavailable.'}</p>
          </div>
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <TrustMetric icon={Database} label="Data source" value={activeDataset ? `Dataset v${activeDataset.version}` : 'None'} detail={activeDataset ? `${activeDataset.row_count.toLocaleString()} rows · ${activeDataset.source_name}` : 'Add data first'} good={Boolean(activeDataset)} />
        <TrustMetric icon={Brain} label="Predictive model" value={activeModel ? `Model v${activeModel.version}` : 'Not active'} detail={activeModel ? activeModel.model_family : 'Optional, governed lifecycle'} good={Boolean(activeModel)} />
        <TrustMetric icon={ShieldCheck} label="Access" value={actor?.role ?? 'unknown'} detail="Server-assigned role boundary" good />
        <TrustMetric icon={Activity} label="Runtime" value={health?.status ?? 'unknown'} detail={health?.adaptation_level ?? 'bounded recovery'} good={health?.status === 'healthy'} />
      </section>

      <section className="grid gap-6 lg:grid-cols-2">
        <Panel title="What PeopleOS can use now" subtitle="Capabilities grounded in the active state">
          <TrustRow label="Aggregate workforce analysis" state={activeDataset ? 'Available' : 'Unavailable'} good={Boolean(activeDataset)} />
          <TrustRow label="Evidence-backed investigations" state={activeDataset ? 'Available' : 'Unavailable'} good={Boolean(activeDataset)} />
          <TrustRow label="Predictive risk analysis" state={activeModel ? 'Model active' : 'Not active'} good={Boolean(activeModel)} />
          <TrustRow label="Investigation history" state={`${workspace?.sessions.length ?? 0} tracked`} good />
        </Panel>

        <Panel title="Fitness checks" subtitle="Deterministic checks, not AI judgement">
          {fitness && Object.entries(fitness.checks).map(([key, passed]) => {
            const notApplicable = !activeModel && isModelCheck(key)
            return <TrustRow key={key} label={key.replaceAll('_', ' ')} state={notApplicable ? 'Not applicable' : passed ? 'Pass' : 'Attention'} good={passed} neutral={notApplicable} />
          })}
          {!fitness && <div className="text-sm text-slate-500">No fitness result available.</div>}
        </Panel>
      </section>

      <section className="rounded-[28px] border border-slate-200 bg-white p-6 dark:border-white/10 dark:bg-slate-900">
        <button type="button" onClick={() => setAdvanced((value) => !value)} className="flex w-full items-center justify-between gap-4 text-left">
          <div><div className="font-semibold text-slate-950 dark:text-white">Advanced governance & recovery</div><div className="mt-1 text-sm text-slate-500 dark:text-slate-400">Technical lifecycle IDs, automatic recovery envelope and governed-only actions.</div></div>
          <span className="text-xs font-semibold text-violet-600 dark:text-violet-300">{advanced ? 'Hide' : 'Show'}</span>
        </button>

        {advanced && (
          <div className="mt-6 grid gap-6 border-t border-slate-200 pt-6 dark:border-white/10 lg:grid-cols-2">
            <div><div className="mb-3 text-xs font-bold uppercase tracking-[0.16em] text-emerald-600 dark:text-emerald-300">May recover automatically</div><ul className="space-y-2 text-sm leading-6 text-slate-600 dark:text-slate-400">{health?.autonomous_recovery_envelope?.map((item) => <li key={item}>• {item}</li>)}</ul></div>
            <div><div className="mb-3 text-xs font-bold uppercase tracking-[0.16em] text-amber-600 dark:text-amber-300">Requires governed action</div><ul className="space-y-2 text-sm leading-6 text-slate-600 dark:text-slate-400">{health?.governed_only?.map((item) => <li key={item}>• {item}</li>)}</ul></div>
            <div className="lg:col-span-2 rounded-2xl bg-slate-50 p-4 text-xs leading-6 text-slate-500 dark:bg-white/[0.04] dark:text-slate-400">Active dataset: {activeDataset?.dataset_id ?? 'none'}<br />Active model: {activeModel?.model_id ?? 'none'}<br />Workspace: {workspace?.workspace_id ?? 'local'}</div>
          </div>
        )}
      </section>

      <section className="flex flex-col gap-3 rounded-2xl border border-slate-200 bg-white p-5 dark:border-white/10 dark:bg-slate-900 md:flex-row md:items-center md:justify-between">
        <div className="flex items-start gap-3"><Wrench className="mt-0.5 h-5 w-5 text-violet-500" /><div><div className="font-semibold">Consequential employment actions remain outside agent autonomy.</div><div className="mt-1 text-sm text-slate-500 dark:text-slate-400">PeopleOS may analyse and explain. It does not terminate, demote, discipline or change pay.</div></div></div>
        <Link href="/advisor" className="inline-flex shrink-0 items-center gap-1 text-sm font-semibold text-violet-600 dark:text-violet-300">Test an investigation <ArrowRight className="h-4 w-4" /></Link>
      </section>
    </div>
  )
}

function Panel({ title, subtitle, children }: { title: string; subtitle: string; children: React.ReactNode }) {
  return <section className="rounded-[28px] border border-slate-200 bg-white p-6 dark:border-white/10 dark:bg-slate-900"><h2 className="font-semibold text-slate-950 dark:text-white">{title}</h2><p className="mb-4 mt-1 text-xs text-slate-500 dark:text-slate-400">{subtitle}</p>{children}</section>
}

function TrustMetric({ icon: Icon, label, value, detail, good }: { icon: React.ElementType; label: string; value: string; detail: string; good: boolean }) {
  return <div className="rounded-2xl border border-slate-200 bg-white p-5 dark:border-white/10 dark:bg-slate-900"><div className="flex items-center justify-between"><Icon className="h-5 w-5 text-violet-500" />{good ? <CheckCircle2 className="h-4 w-4 text-emerald-500" /> : <span className="h-2 w-2 rounded-full bg-slate-300 dark:bg-slate-600" />}</div><div className="mt-4 text-xs font-semibold text-slate-500 dark:text-slate-400">{label}</div><div className="mt-1 text-lg font-semibold text-slate-950 dark:text-white">{value}</div><div className="mt-1 truncate text-xs text-slate-500 dark:text-slate-400">{detail}</div></div>
}

function TrustRow({ label, state, good, neutral = false }: { label: string; state: string; good: boolean; neutral?: boolean }) {
  const tone = neutral ? 'text-slate-500 dark:text-slate-400' : good ? 'text-emerald-600 dark:text-emerald-300' : 'text-amber-600 dark:text-amber-300'
  const dot = neutral ? 'bg-slate-300 dark:bg-slate-600' : good ? 'bg-emerald-500' : 'bg-amber-500'
  return <div className="flex items-center justify-between gap-4 border-b border-slate-100 py-3 last:border-0 dark:border-white/5"><span className="text-sm capitalize text-slate-600 dark:text-slate-300">{label}</span><span className={`inline-flex items-center gap-2 text-xs font-semibold ${tone}`}><span className={`h-2 w-2 rounded-full ${dot}`} />{state}</span></div>
}
