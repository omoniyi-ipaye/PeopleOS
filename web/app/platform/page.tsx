'use client'

import { useCallback, useEffect, useState } from 'react'
import Link from 'next/link'
import { Activity, ArrowRight, Brain, Database, RefreshCw, ShieldCheck } from 'lucide-react'
import { Button, EmptyState, MetricCard, Page, PageHeader, SectionHeader, StateSummary, StatusBadge, Surface } from '@/components/ui'

interface DatasetRecord { dataset_id: string; version: number; source_name: string; row_count: number; state: string; created_at: string }
interface ModelRecord { model_id: string; version: number; model_family: string; state: string; metrics: Record<string, number> }
interface Workspace { workspace_id: string; name: string; active_dataset_id?: string | null; active_model_id?: string | null; datasets: DatasetRecord[]; models: ModelRecord[]; sessions: Array<{ session_id: string; state: string }> }
interface Integrity { status: string; issues: string[]; model_ready: boolean; snapshot?: { dataset_id?: string; source_rows: number; current_rows: number; active_rows: number; unknown_status_rows: number; population_contract: string } }
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
      if (!responses.every(response => response.ok)) throw new Error('PeopleOS could not verify its trust state.')
      setWorkspace(await responses[0].json()); setHealth(await responses[1].json()); setFitness(await responses[2].json()); setActor(await responses[3].json()); setError(null)
    } catch (err) { setError(err instanceof Error ? err.message : 'Unable to load trust information') }
    finally { setLoading(false) }
  }, [])

  useEffect(() => { void load() }, [load])
  if (loading) return <Page><StateSummary title="Checking your PeopleOS setup" description="Verifying the active data, privacy boundary and app health." tone="info" /></Page>
  if (error) return <Page><EmptyState title="Trust information is unavailable" description={error} action={<Button onClick={() => void load()}><RefreshCw className="h-4 w-4" />Retry</Button>} /></Page>

  const activeDataset = workspace?.datasets.find(item => item.dataset_id === workspace.active_dataset_id)
  const activeModel = health?.runtime_integrity?.model_ready ? workspace?.models.find(item => item.model_id === workspace.active_model_id) : undefined
  const snapshot = health?.runtime_integrity?.snapshot
  const evidenceReady = health?.runtime_integrity?.status === 'verified'
  const failedChecks = fitness ? Object.entries(fitness.checks).filter(([key, passed]) => !passed && !(key.includes('model_') && !activeModel)) : []
  const trustHealthy = health?.status === 'healthy' && evidenceReady && failedChecks.length === 0

  return <Page>
    <PageHeader eyebrow="Trust & Privacy" title="Can I rely on PeopleOS?" description="See the few things that matter: which data is being used, whether it is healthy, whether predictive insights are active, and what stays outside AI control." actions={<Button variant="secondary" size="sm" onClick={() => void load()}><RefreshCw className="h-4 w-4" />Refresh</Button>} />

    <StateSummary title={trustHealthy ? 'Your PeopleOS setup looks healthy' : 'Something needs attention'} description={trustHealthy ? 'PeopleOS is using a verified local data source and the core evidence path is working.' : 'Review the items below before relying on new analysis.'} tone={trustHealthy ? 'success' : 'warning'} />

    <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <MetricCard label="Workforce data" value={activeDataset ? `Ready · v${activeDataset.version}` : 'Not added'} detail={activeDataset ? `${activeDataset.row_count.toLocaleString()} rows · ${activeDataset.source_name}` : 'Add data to begin'} icon={Database} tone={activeDataset ? 'success' : 'neutral'} />
      <MetricCard label="Predictive insights" value={activeModel ? 'Experimental model active' : 'Not enabled'} detail={activeModel ? activeModel.model_family : 'Descriptive analysis still works'} icon={Brain} tone={activeModel ? 'info' : 'neutral'} />
      <MetricCard label="Privacy & access" value="Local" detail={`Current access: ${actor?.role ?? 'owner'}`} icon={ShieldCheck} tone="success" />
      <MetricCard label="App health" value={health?.status === 'healthy' ? 'Healthy' : 'Needs attention'} detail="Local analysis service" icon={Activity} tone={health?.status === 'healthy' ? 'success' : 'warning'} />
    </section>

    <div className="grid gap-6 lg:grid-cols-2">
      <Surface padding="lg"><SectionHeader title="Your current data" description="The workforce source behind every new answer." />{snapshot ? <div className="mt-5 space-y-3"><TrustRow label="Data check" state={evidenceReady ? 'Verified' : 'Needs attention'} tone={evidenceReady ? 'success' : 'warning'} /><TrustRow label="Current employee records" state={snapshot.current_rows.toLocaleString()} tone="info" /><TrustRow label="Active employees" state={snapshot.active_rows.toLocaleString()} tone="info" /><TrustRow label="Unknown statuses" state={snapshot.unknown_status_rows.toLocaleString()} tone={snapshot.unknown_status_rows ? 'warning' : 'success'} /></div> : <EmptyState title="No active workforce data" />}</Surface>
      <Surface padding="lg"><SectionHeader title="What PeopleOS can do now" description="Capabilities are shown only when the current setup supports them." /><div className="mt-5 space-y-2"><TrustRow label="Workforce analysis" state={evidenceReady ? 'Available' : 'Unavailable'} tone={evidenceReady ? 'success' : 'warning'} /><TrustRow label="Ask PeopleOS" state={evidenceReady ? 'Available' : 'Unavailable'} tone={evidenceReady ? 'success' : 'warning'} /><TrustRow label="Predictive insights" state={activeModel ? 'Experimental' : 'Off'} tone={activeModel ? 'info' : 'neutral'} /><TrustRow label="Saved investigations" state={`${workspace?.sessions.length ?? 0}`} tone="info" /></div></Surface>
    </div>

    <Surface padding="lg" className="border-emerald-200/70 bg-emerald-50/40 dark:border-emerald-500/20 dark:bg-emerald-500/[0.03]">
      <div className="flex gap-4"><ShieldCheck className="mt-0.5 h-5 w-5 shrink-0 text-emerald-600" /><div><div className="font-semibold">PeopleOS analyses. People decide.</div><p className="mt-1 text-sm leading-6 text-text-secondary">PeopleOS does not terminate, demote, discipline, reduce pay or make consequential employee decisions. Individual risk ranking is kept outside the governed product experience.</p></div></div>
    </Surface>

    <Surface padding="lg">
      <button type="button" onClick={() => setAdvanced(value => !value)} aria-expanded={advanced} aria-controls="advanced-trust-details" className="flex w-full items-center justify-between gap-4 text-left"><div><div className="font-semibold">Advanced trust details</div><div className="mt-1 text-sm text-text-secondary">Technical lifecycle checks, recovery boundaries and identifiers.</div></div><span className="text-xs font-semibold text-accent">{advanced ? 'Hide' : 'Show'}</span></button>
      {advanced && <div id="advanced-trust-details" className="mt-6 border-t border-border pt-6"><div className="grid gap-6 lg:grid-cols-2"><div><div className="text-xs font-semibold uppercase tracking-wider text-success">Can recover automatically</div><ul className="mt-3 space-y-2 text-sm leading-6 text-text-secondary">{health?.autonomous_recovery_envelope?.map(item => <li key={item}>• {item}</li>)}</ul></div><div><div className="text-xs font-semibold uppercase tracking-wider text-warning">Needs an explicit action</div><ul className="mt-3 space-y-2 text-sm leading-6 text-text-secondary">{health?.governed_only?.map(item => <li key={item}>• {item}</li>)}</ul></div></div><div className="mt-6 rounded-2xl bg-background-secondary p-4 font-mono text-xs leading-6 text-text-muted">Dataset: {activeDataset?.dataset_id ?? 'none'}<br />Model: {activeModel?.model_id ?? 'none'}<br />Workspace: {workspace?.workspace_id ?? 'local'}</div></div>}
    </Surface>

    <Link href="/advisor" className="inline-flex items-center gap-2 text-sm font-semibold text-accent">Ask PeopleOS a question <ArrowRight className="h-4 w-4" /></Link>
  </Page>
}

function TrustRow({ label, state, tone }: { label: string; state: string; tone: 'success' | 'warning' | 'neutral' | 'info' }) { return <div className="flex items-center justify-between gap-4 border-b border-border py-3 last:border-0"><span className="text-sm text-text-secondary">{label}</span><StatusBadge tone={tone}>{state}</StatusBadge></div> }
