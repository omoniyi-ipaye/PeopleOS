'use client'

import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api-client'
import { EmptyState, MetricCard, Page, PageHeader, SectionHeader, StateSummary, StatusBadge, Surface } from '@/components/ui'
import { Activity, Brain, Cpu, Database, Search, Server, ShieldCheck } from 'lucide-react'
import type { UploadStatus } from '@/types/api'

export default function SettingsPage() {
  const { data: status, isLoading: statusLoading, isError } = useQuery<UploadStatus>({ queryKey: ['api', 'status'], queryFn: () => api.getStatus() as Promise<UploadStatus> })
  const { data: health, isLoading: healthLoading } = useQuery<{ status: string }>({ queryKey: ['api', 'health'], queryFn: () => api.getHealth() as Promise<{ status: string }> })

  if (statusLoading || healthLoading) return <Page><StateSummary title="Reading PeopleOS configuration" description="Checking runtime, dataset and capability state." tone="info" /></Page>
  if (isError) return <Page><EmptyState title="System state is unavailable" description="PeopleOS could not read its current runtime configuration." /></Page>

  const engines = status?.engines ?? {}
  const engineRows = [
    ['Analytics core', 'analytics', Activity],
    ['Predictive engine', 'ml', Brain],
    ['Compensation', 'compensation', Database],
    ['Succession', 'succession', Database],
    ['Team dynamics', 'team_dynamics', Cpu],
    ['Fairness', 'fairness', ShieldCheck],
    ['Semantic search', 'vector_search', Search],
    ['Local AI', 'llm', Brain],
  ] as const

  return (
    <Page>
      <PageHeader eyebrow="Govern · Settings" title="System configuration and capability state" description="Operational configuration belongs here. Trust, evidence fitness and recovery controls remain in Trust Center." />

      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="Runtime" value={health?.status === 'healthy' ? 'Healthy' : health?.status ?? 'Unknown'} detail="Backend health state" icon={Server} tone={health?.status === 'healthy' ? 'success' : 'warning'} />
        <MetricCard label="Dataset" value={status?.data?.loaded ? 'Active' : 'Not loaded'} detail={status?.data?.loaded ? `${status.data.row_count.toLocaleString()} rows` : 'Add data in Data & Sources'} icon={Database} tone={status?.data?.loaded ? 'success' : 'neutral'} />
        <MetricCard label="Predictive capability" value={status?.engines?.ml ? 'Available' : 'Not active'} detail="Separate governed model lifecycle" icon={Brain} tone={status?.engines?.ml ? 'info' : 'neutral'} />
        <MetricCard label="Local AI" value={status?.engines?.llm ? 'Available' : 'Fallback mode'} detail={status?.engines?.llm ? 'Contextual synthesis available' : 'Deterministic synthesis remains available'} icon={ShieldCheck} tone={status?.engines?.llm ? 'info' : 'success'} />
      </section>

      <Surface padding="lg">
        <SectionHeader title="Capability registry" description="Which analytical engines are currently available in this runtime." />
        <div className="mt-5 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          {engineRows.map(([label, key, Icon]) => { const active = Boolean((engines as Record<string, boolean>)[key]); return <div key={key} className="rounded-2xl border border-border p-4"><div className="flex items-center justify-between"><Icon className="h-4 w-4 text-accent" /><StatusBadge tone={active ? 'success' : 'neutral'}>{active ? 'Available' : 'Unavailable'}</StatusBadge></div><div className="mt-4 text-sm font-semibold">{label}</div><div className="mt-1 text-xs text-text-muted">{active ? 'Ready for supported workflows' : 'Not part of the current active capability set'}</div></div> })}
        </div>
      </Surface>

      {status?.features_enabled ? <Surface padding="lg"><SectionHeader title="Dataset-enabled features" description="Features inferred from the active source schema; this does not mean every optional model is trained." /><div className="mt-5 flex flex-wrap gap-2">{Object.entries(status.features_enabled).map(([key, enabled]) => <StatusBadge key={key} tone={enabled ? 'success' : 'neutral'}>{key.replaceAll('_', ' ')} · {enabled ? 'ready' : 'not available'}</StatusBadge>)}</div></Surface> : null}

      <div className="grid gap-6 lg:grid-cols-2">
        <Surface padding="lg"><SectionHeader title="Runtime architecture" description="Current product stack" /><div className="mt-5 space-y-3 text-sm text-text-secondary"><div className="flex justify-between border-b border-border py-2"><span>Web application</span><span className="font-medium text-text-primary">Next.js 16 · React 19</span></div><div className="flex justify-between border-b border-border py-2"><span>API</span><span className="font-medium text-text-primary">FastAPI</span></div><div className="flex justify-between border-b border-border py-2"><span>Default install</span><span className="font-medium text-text-primary">Core analytics runtime</span></div><div className="flex justify-between py-2"><span>Advanced NLP/vector</span><span className="font-medium text-text-primary">Optional capability tier</span></div></div></Surface>
        <Surface padding="lg"><SectionHeader title="Privacy boundary" description="How local-first operation is represented" /><StateSummary title="Workforce data remains inside the configured PeopleOS runtime" description="External model or connector use must be explicitly configured and governed. The UI should never imply a capability is active solely because source fields exist." tone="success" /></Surface>
      </div>
    </Page>
  )
}
