'use client'

import { useCallback, useEffect, useState } from 'react'
import { Activity, Database, ShieldCheck, Brain, RefreshCw, ServerCog } from 'lucide-react'

interface DatasetRecord {
  dataset_id: string
  version: number
  source_name: string
  row_count: number
  state: string
  created_at: string
}

interface ModelRecord {
  model_id: string
  version: number
  model_family: string
  state: string
  metrics: Record<string, number>
}

interface Workspace {
  workspace_id: string
  name: string
  active_dataset_id?: string | null
  active_model_id?: string | null
  datasets: DatasetRecord[]
  models: ModelRecord[]
  sessions: Array<{ session_id: string; state: string }>
}

interface Health {
  status: string
  checks: Array<{ id: string; healthy: boolean }>
  interrupted_jobs?: string[]
  adaptation_level?: string
  autonomous_recovery_envelope?: string[]
  governed_only?: string[]
}

interface Fitness {
  status: string
  checks: Record<string, boolean>
  observed: { dataset_age_days?: number | null; model_age_days?: number | null; model_auc?: number | null }
}

interface Actor {
  actor_id: string
  role: string
  permissions: string[]
}

export default function PlatformPage() {
  const [workspace, setWorkspace] = useState<Workspace | null>(null)
  const [health, setHealth] = useState<Health | null>(null)
  const [fitness, setFitness] = useState<Fitness | null>(null)
  const [actor, setActor] = useState<Actor | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const [workspaceResponse, healthResponse, fitnessResponse, actorResponse] = await Promise.all([
        fetch('/api/platform/workspaces/local'),
        fetch('/api/platform/health'),
        fetch('/api/platform/workspaces/local/fitness'),
        fetch('/api/platform/me'),
      ])
      if (![workspaceResponse, healthResponse, fitnessResponse, actorResponse].every((response) => response.ok)) {
        throw new Error('The platform control plane is not available.')
      }
      setWorkspace(await workspaceResponse.json())
      setHealth(await healthResponse.json())
      setFitness(await fitnessResponse.json())
      setActor(await actorResponse.json())
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to load platform status')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { void load() }, [load])

  if (loading) return <div className="p-8 text-text-secondary">Loading PeopleOS system state…</div>
  if (error) return <div className="p-8 text-red-500">{error}</div>

  const activeDataset = workspace?.datasets.find((item) => item.dataset_id === workspace.active_dataset_id)
  const activeModel = workspace?.models.find((item) => item.model_id === workspace.active_model_id)

  return (
    <div className="p-6 md:p-8 space-y-8 max-w-7xl mx-auto">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.25em] text-text-muted">System Control Plane</p>
          <h1 className="text-3xl font-bold mt-2">PeopleOS System Health</h1>
          <p className="text-text-secondary mt-2 max-w-2xl">Workspace, dataset, model, investigation and recovery state from the governed runtime.</p>
        </div>
        <button onClick={() => void load()} className="inline-flex items-center gap-2 px-4 py-2 rounded-xl border border-border hover:bg-white/5">
          <RefreshCw className="w-4 h-4" /> Refresh
        </button>
      </div>

      <div className="grid md:grid-cols-4 gap-4">
        <Metric icon={Activity} label="Platform" value={health?.status ?? 'unknown'} detail={health?.adaptation_level ?? 'L0'} />
        <Metric icon={Database} label="Dataset" value={activeDataset ? `v${activeDataset.version}` : 'None'} detail={activeDataset ? `${activeDataset.row_count.toLocaleString()} rows` : 'Upload required'} />
        <Metric icon={Brain} label="Active Model" value={activeModel ? `v${activeModel.version}` : 'None'} detail={fitness?.observed.model_auc != null ? `AUC ${fitness.observed.model_auc.toFixed(2)}` : 'No active model'} />
        <Metric icon={ShieldCheck} label="Access" value={actor?.role ?? 'unknown'} detail={actor?.actor_id ?? 'anonymous'} />
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        <Panel title="Lifecycle">
          <LifecycleRow label="Workspace" value={workspace?.name ?? 'Local workspace'} state="active" />
          <LifecycleRow label="Dataset" value={activeDataset?.source_name ?? 'No active dataset'} state={activeDataset?.state ?? 'missing'} />
          <LifecycleRow label="Model" value={activeModel?.model_family ?? 'No active model'} state={activeModel?.state ?? 'optional'} />
          <LifecycleRow label="Investigations" value={`${workspace?.sessions.length ?? 0} sessions`} state="tracked" />
        </Panel>

        <Panel title="Fitness checks">
          {fitness && Object.entries(fitness.checks).map(([key, healthy]) => (
            <div key={key} className="flex items-center justify-between py-2 border-b border-border/50 last:border-0">
              <span className="text-sm">{key.replaceAll('_', ' ')}</span>
              <span className={healthy ? 'text-green-500 text-sm' : 'text-amber-500 text-sm'}>{healthy ? 'Pass' : 'Attention'}</span>
            </div>
          ))}
        </Panel>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        <Panel title="Dataset versions">
          {(workspace?.datasets ?? []).slice().reverse().map((dataset) => (
            <div key={dataset.dataset_id} className="py-3 border-b border-border/50 last:border-0">
              <div className="flex justify-between gap-4"><span className="font-medium">v{dataset.version} · {dataset.source_name}</span><span className="text-xs uppercase text-text-muted">{dataset.state}</span></div>
              <p className="text-xs text-text-muted mt-1">{dataset.row_count.toLocaleString()} rows · {dataset.dataset_id}</p>
            </div>
          ))}
          {workspace?.datasets.length === 0 && <p className="text-sm text-text-muted">No dataset versions registered yet.</p>}
        </Panel>

        <Panel title="Model versions">
          {(workspace?.models ?? []).slice().reverse().map((model) => (
            <div key={model.model_id} className="py-3 border-b border-border/50 last:border-0">
              <div className="flex justify-between gap-4"><span className="font-medium">v{model.version} · {model.model_family}</span><span className="text-xs uppercase text-text-muted">{model.state}</span></div>
              <p className="text-xs text-text-muted mt-1">{model.model_id}</p>
            </div>
          ))}
          {workspace?.models.length === 0 && <p className="text-sm text-text-muted">No governed model versions yet.</p>}
        </Panel>
      </div>

      <Panel title="Bounded self-healing envelope">
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <p className="text-xs uppercase tracking-wider text-green-500 mb-3">May recover automatically</p>
            <ul className="space-y-2 text-sm">{health?.autonomous_recovery_envelope?.map((item) => <li key={item}>• {item}</li>)}</ul>
          </div>
          <div>
            <p className="text-xs uppercase tracking-wider text-amber-500 mb-3">Requires governed action</p>
            <ul className="space-y-2 text-sm">{health?.governed_only?.map((item) => <li key={item}>• {item}</li>)}</ul>
          </div>
        </div>
      </Panel>

      <div className="flex items-center gap-2 text-xs text-text-muted"><ServerCog className="w-4 h-4" /> Consequential employment actions remain outside the agent autonomy boundary.</div>
    </div>
  )
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return <section className="rounded-2xl border border-border bg-surface/60 p-5"><h2 className="font-semibold mb-4">{title}</h2>{children}</section>
}

function Metric({ icon: Icon, label, value, detail }: { icon: React.ElementType; label: string; value: string; detail: string }) {
  return <div className="rounded-2xl border border-border bg-surface/60 p-4"><Icon className="w-5 h-5 text-accent" /><p className="text-xs uppercase tracking-wider text-text-muted mt-4">{label}</p><p className="text-xl font-semibold mt-1 capitalize">{value}</p><p className="text-xs text-text-muted mt-1 truncate">{detail}</p></div>
}

function LifecycleRow({ label, value, state }: { label: string; value: string; state: string }) {
  return <div className="flex items-center justify-between gap-4 py-3 border-b border-border/50 last:border-0"><div><p className="text-xs text-text-muted">{label}</p><p className="text-sm font-medium mt-1">{value}</p></div><span className="text-xs uppercase text-accent">{state}</span></div>
}
