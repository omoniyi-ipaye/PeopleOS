'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api-client'
import { Button, EmptyState, MetricCard, Page, PageHeader, SectionHeader, StateSummary, StatusBadge, Surface, TrustDisclosure } from '@/components/ui'
import { Activity, Brain, Database, ExternalLink, Power, RefreshCw, Server, ShieldCheck } from 'lucide-react'

interface PlatformStatus {
  data?: { loaded?: boolean; row_count?: number }
  capabilities?: Record<string, boolean>
}
interface UploadState { features_enabled?: Record<string, boolean> }
interface DesktopState { desktop: boolean; restart_supported?: boolean; quit_supported?: boolean }

export default function SettingsPage() {
  const [desktopAction, setDesktopAction] = useState<'open' | 'restart' | 'quit' | null>(null)
  const platform = useQuery<PlatformStatus>({ queryKey: ['api', 'status'], queryFn: () => api.getStatus() as Promise<PlatformStatus> })
  const health = useQuery<{ status: string }>({ queryKey: ['api', 'health'], queryFn: () => api.getHealth() as Promise<{ status: string }> })
  const upload = useQuery<UploadState>({ queryKey: ['upload', 'status'], queryFn: () => api.upload.getStatus() as Promise<UploadState> })
  const desktop = useQuery<DesktopState>({
    queryKey: ['desktop', 'status'], retry: false,
    queryFn: async () => { const response = await fetch('/api/desktop/status'); return response.ok ? response.json() : { desktop: false } },
  })

  async function runDesktop(action: 'open' | 'restart' | 'quit') {
    setDesktopAction(action)
    try {
      const response = await fetch(`/api/desktop/${action}`, { method: 'POST' })
      if (!response.ok) throw new Error('Desktop control unavailable')
      if (action === 'open') window.setTimeout(() => setDesktopAction(null), 800)
    } catch {
      setDesktopAction(null)
    }
  }

  if (platform.isLoading || health.isLoading) return <Page><StateSummary title="Opening settings" description="Checking your PeopleOS app and workforce state." tone="info" /></Page>
  if (platform.isError) return <Page><EmptyState title="Settings are temporarily unavailable" description="PeopleOS could not read the current app state." /></Page>

  const capabilities = platform.data?.capabilities ?? {}
  const features = upload.data?.features_enabled ?? {}
  const hasData = Boolean(platform.data?.data?.loaded)
  const appHealthy = health.data?.status === 'healthy'

  return <Page>
    <PageHeader eyebrow="Settings" title="PeopleOS settings" description="Manage the local app and see which capabilities are available. Technical details stay out of the way unless you need them." />

    {desktop.data?.desktop && <Surface padding="lg" className="border-violet-200/70 bg-gradient-to-br from-white to-violet-50/30 dark:border-violet-500/20 dark:from-slate-950 dark:to-violet-500/[0.03]">
      <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
        <div><div className="text-xs font-bold uppercase tracking-[0.16em] text-violet-600 dark:text-violet-300">Desktop app</div><h2 className="mt-2 text-xl font-semibold">PeopleOS is running on this computer</h2><p className="mt-2 max-w-2xl text-sm leading-6 text-text-secondary">Open another PeopleOS tab, restart the local app cleanly, or quit it completely. No terminal or Task Manager needed.</p></div>
        <div className="flex flex-wrap gap-2">
          <Button variant="secondary" disabled={desktopAction !== null} onClick={() => void runDesktop('open')}><ExternalLink className="h-4 w-4" />Open PeopleOS</Button>
          <Button variant="secondary" disabled={desktopAction !== null || !desktop.data.restart_supported} onClick={() => void runDesktop('restart')}><RefreshCw className={`h-4 w-4 ${desktopAction === 'restart' ? 'animate-spin' : ''}`} />Restart app</Button>
          <Button variant="danger" disabled={desktopAction !== null || !desktop.data.quit_supported} onClick={() => void runDesktop('quit')}><Power className="h-4 w-4" />Quit PeopleOS</Button>
        </div>
      </div>
      {desktopAction === 'restart' && <p className="mt-4 text-sm text-text-secondary">PeopleOS is restarting and will reopen when it is ready.</p>}
      {desktopAction === 'quit' && <p className="mt-4 text-sm text-text-secondary">PeopleOS is closing. You can close this browser tab.</p>}
    </Surface>}

    <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <MetricCard label="PeopleOS app" value={appHealthy ? 'Ready' : health.data?.status ?? 'Needs attention'} detail="Local application status" icon={Server} tone={appHealthy ? 'success' : 'warning'} />
      <MetricCard label="Workforce data" value={hasData ? 'Ready' : 'Not added'} detail={hasData ? `${platform.data?.data?.row_count?.toLocaleString() ?? ''} source records` : 'Add a workforce file to begin'} icon={Database} tone={hasData ? 'success' : 'neutral'} />
      <MetricCard label="Predictive insights" value={capabilities.predictive_model ? 'Available' : 'Not active'} detail="Shown only after a governed model is activated" icon={Brain} tone={capabilities.predictive_model ? 'info' : 'neutral'} />
      <MetricCard label="AI assistance" value={capabilities.llm ? 'Available' : 'Deterministic mode'} detail={capabilities.llm ? 'AI can organise verified evidence' : 'Calculations and verified answers still work'} icon={ShieldCheck} tone="success" />
    </section>

    <Surface padding="lg">
      <SectionHeader title="What your current data supports" description="PeopleOS only enables analysis when the required data is present and usable." />
      <div className="mt-5 flex flex-wrap gap-2">
        {Object.keys(features).length ? Object.entries(features).map(([key, enabled]) => <StatusBadge key={key} tone={enabled ? 'success' : 'neutral'}>{key.replaceAll('_', ' ')} · {enabled ? 'ready' : 'not available'}</StatusBadge>) : <span className="text-sm text-text-secondary">Add workforce data to see available analysis areas.</span>}
      </div>
    </Surface>

    <Surface padding="lg"><SectionHeader title="Privacy" description="Your default PeopleOS boundary" /><StateSummary title="Your workforce data stays in this PeopleOS installation" description="External AI or connectors are never implied to be active just because your file contains a field. Any external capability must be explicitly configured." tone="success" /></Surface>

    <TrustDisclosure title="Advanced system details" summary="For technical review and troubleshooting">
      <div className="grid gap-3 md:grid-cols-2">
        {Object.entries(capabilities).map(([key, enabled]) => <div key={key} className="flex items-center justify-between rounded-xl border border-border px-4 py-3"><div className="flex items-center gap-2 text-sm"><Activity className="h-4 w-4 text-accent" />{key.replaceAll('_', ' ')}</div><StatusBadge tone={enabled ? 'success' : 'neutral'}>{enabled ? 'Available' : 'Unavailable'}</StatusBadge></div>)}
      </div>
      <div className="mt-5 text-sm leading-6 text-text-secondary">PeopleOS uses a local Next.js interface with a FastAPI analytical runtime. Advanced model and retrieval capabilities remain optional and governed separately from deterministic workforce calculations.</div>
    </TrustDisclosure>
  </Page>
}
