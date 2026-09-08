'use client'

import { useCallback, useEffect, useState } from 'react'
import Link from 'next/link'
import { Activity, ArrowRight, Brain, Database, RefreshCw, ShieldCheck, Wrench } from 'lucide-react'
import { Button, EmptyState, MetricCard, Page, PageHeader, SectionHeader, StateSummary, StatusBadge, Surface } from '@/components/ui'

interface DatasetRecord { dataset_id: string; version: number; source_name: string; row_count: number; state: string; created_at: string }
interface ModelRecord { model_id: string; version: number; model_family: string; state: string; metrics: Record<string, number> }
interface Workspace { workspace_id: string; name: string; active_dataset_id?: string | null; active_model_id?: string | null; datasets: DatasetRecord[]; models: ModelRecord[]; sessions: Array<{ session_id: string; state: string }> }
interface Integrity {status: string; issues: string[]; model_ready: boolean; snapshot?: {dataset_id?: string; source_rows: number; current_rows: number; active_rows: number; unknown_status_rows: number; population_contract: string}}
interface Health { runtime_integrity?: Integrity; status: string; checks: Array<{ id: string; healthy: boolean }>; interrupted_jobs?: string[]; adaptation_level?: string; autonomous_recovery_envelope?: string[]; governed_only?: string[] }
interface Fitness { status: string; checks: Record<string, boolean>; observed: { dataset_age_days?: number | null; model_age_days?: number | null; model_auc?: number | null } }
interface Actor { actor_id: string; role: string; permissions: string[] }

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
      const responses = await Promise.all([fetch('/api/platform/workspaces/local'), fetch('/api/platform/health'), fetch('/api/platform/workspaces/local/fitness'), fetch('/api/platform/me')])
      if (!responses.every((response) => response.ok)) throw new Error('PeopleOS trust state is not available.')
      setWorkspace(await responses[0].json()); setHealth(await responses[1].json()); setFitness(await responses[2].json()); setActor(await responses[3].json()); setError(null)
    } catch (err) { setError(err instanceof Error ? err.message : 'Unable to load trust state') }
    finally { setLoading(false) }
  }, [])

  useEffect(() => { void load() }, [load])

  if (loading) return <Page><StateSummary title="Checking PeopleOS trust state" description="Reading data, model, access and runtime fitness." tone="info" /></Page>
  if (error) return <Page><EmptyState title="Trust state is unavailable" description={error} action={<Button onClick={() => void load()}><RefreshCw className="h-4 w-4" />Retry</Button>} /></Page>

  const activeDataset = workspace?.datasets.find((item) => item.dataset_id === workspace.active_dataset_id)
  const activeModel = health?.runtime_integrity?.model_ready ? workspace?.models.find((item) => item.model_id === workspace.active_model_id) : undefined
  const snapshot = health?.runtime_integrity?.snapshot
  const evidenceReady = health?.runtime_integrity?.status === 'verified'
  const failedChecks = fitness ? Object.entries(fitness.checks).filter(([, passed]) => !passed) : []
  const trustHealthy = health?.status === 'healthy' && evidenceReady && failedChecks.length === 0

  return <Page>
    <PageHeader eyebrow="Govern · Trust Center" title="Can I trust this analysis?" description="See the source, model state, access boundary and recovery controls behind PeopleOS outputs." actions={<Button variant="secondary" size="sm" onClick={() => void load()}><RefreshCw className="h-4 w-4" />Refresh</Button>} />

    <StateSummary title={trustHealthy ? 'Core evidence path is healthy' : 'Some capabilities need attention'} description={activeModel ? 'A governed predictive model is active in addition to deterministic evidence.' : 'No predictive model is active. People Intelligence relies on deterministic aggregate evidence and marks predictive capabilities unavailable.'} tone={trustHealthy ? 'success' : 'warning'} />

    <Surface padding="lg"><SectionHeader title="Dataset integrity" description="Checks that the loaded population and results belong to the selected dataset. This does not certify statistical accuracy." />
      <div className="mt-4"><StatusBadge tone={evidenceReady ? 'success' : 'warning'}>{evidenceReady ? 'Snapshot verified' : 'Evidence unavailable'}</StatusBadge></div>
      {health?.runtime_integrity?.issues.map((issue) => <p key={issue} className="mt-2 text-sm text-text-secondary">{issue}</p>)}
      {snapshot && <p className="mt-3 text-sm text-text-secondary">{snapshot.source_rows.toLocaleString()} source rows · {snapshot.current_rows.toLocaleString()} current employee records · {snapshot.active_rows.toLocaleString()} active · {snapshot.unknown_status_rows.toLocaleString()} unknown statuses. {snapshot.population_contract === 'active_only_input' ? 'Input has no status column and is interpreted as an active-only roster.' : 'Population resolved from recorded statuses.'}</p>}
    </Surface>

    <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <MetricCard label="Data source" value={activeDataset ? `Dataset v${activeDataset.version}` : 'None'} detail={activeDataset ? `${activeDataset.row_count.toLocaleString()} rows · ${activeDataset.source_name}` : 'Add data first'} icon={Database} tone={activeDataset ? 'success' : 'neutral'} />
      <MetricCard label="Predictive model" value={activeModel ? `Model v${activeModel.version}` : 'Not active'} detail={activeModel ? activeModel.model_family : 'Optional governed lifecycle'} icon={Brain} tone={activeModel ? 'info' : 'neutral'} />
      <MetricCard label="Access" value={actor?.role ?? 'Unknown'} detail="Server-assigned role boundary" icon={ShieldCheck} tone="success" />
      <MetricCard label="Runtime" value={health?.status ?? 'Unknown'} detail={health?.adaptation_level ?? 'Bounded recovery'} icon={Activity} tone={health?.status === 'healthy' ? 'success' : 'warning'} />
    </section>

    <div className="grid gap-6 lg:grid-cols-2">
      <Surface padding="lg"><SectionHeader title="What PeopleOS can use now" description="Capabilities grounded in the active lifecycle state." /><div className="mt-5 space-y-2"><TrustRow label="Aggregate workforce analysis" state={evidenceReady ? 'Available' : 'Unavailable'} tone={evidenceReady ? 'success' : 'warning'} /><TrustRow label="Evidence-backed investigations" state={evidenceReady ? 'Available' : 'Unavailable'} tone={evidenceReady ? 'success' : 'warning'} /><TrustRow label="Predictive risk analysis" state={activeModel ? 'Model active' : 'Not active'} tone={activeModel ? 'success' : 'neutral'} /><TrustRow label="Investigation history" state={`${workspace?.sessions.length ?? 0} tracked`} tone="info" /></div></Surface>
      <Surface padding="lg"><SectionHeader title="Fitness checks" description="Deterministic lifecycle checks, not AI judgement." /><div className="mt-5 space-y-2">{fitness ? Object.entries(fitness.checks).map(([key, passed]) => { const modelCheck = key.includes('model_') && !activeModel; return <TrustRow key={key} label={key.replaceAll('_', ' ')} state={modelCheck ? 'Not applicable' : passed ? 'Pass' : 'Attention'} tone={modelCheck ? 'neutral' : passed ? 'success' : 'warning'} /> }) : <EmptyState title="No fitness result available" />}</div></Surface>
    </div>

    <Surface padding="lg">
      <button type="button" onClick={() => setAdvanced((value) => !value)} className="flex w-full items-center justify-between gap-4 text-left"><div><div className="font-semibold">Advanced governance & recovery</div><div className="mt-1 text-sm text-text-secondary">Lifecycle identifiers, bounded recovery and governed-only actions.</div></div><span className="text-xs font-semibold text-accent">{advanced ? 'Hide' : 'Show'}</span></button>
      {advanced && <div className="mt-6 grid gap-6 border-t border-border pt-6 lg:grid-cols-2"><div><div className="text-xs font-semibold uppercase tracking-wider text-success">May recover automatically</div><ul className="mt-3 space-y-2 text-sm leading-6 text-text-secondary">{health?.autonomous_recovery_envelope?.map((item) => <li key={item}>• {item}</li>)}</ul></div><div><div className="text-xs font-semibold uppercase tracking-wider text-warning">Requires governed action</div><ul className="mt-3 space-y-2 text-sm leading-6 text-text-secondary">{health?.governed_only?.map((item) => <li key={item}>• {item}</li>)}</ul></div><div className="lg:col-span-2 rounded-2xl bg-background-secondary p-4 font-mono text-xs leading-6 text-text-muted">Active dataset: {activeDataset?.dataset_id ?? 'none'}<br />Active model: {activeModel?.model_id ?? 'none'}<br />Workspace: {workspace?.workspace_id ?? 'local'}</div></div>}
    </Surface>

    <StateSummary title="Consequential employment actions remain outside agent autonomy" description="PeopleOS may analyse and explain. It does not terminate, demote, discipline or change pay." tone="success" />
    <Link href="/advisor" className="inline-flex items-center gap-2 text-sm font-semibold text-accent"><Wrench className="h-4 w-4" />Test an investigation <ArrowRight className="h-4 w-4" /></Link>
  </Page>
}

function TrustRow({ label, state, tone }: { label: string; state: string; tone: 'success' | 'warning' | 'neutral' | 'info' }) { return <div className="flex items-center justify-between gap-4 border-b border-border py-3 last:border-0"><span className="text-sm capitalize text-text-secondary">{label}</span><StatusBadge tone={tone}>{state}</StatusBadge></div> }
