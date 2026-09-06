'use client'

import Link from 'next/link'
import { useQuery } from '@tanstack/react-query'
import { Brain, CheckCircle2, Clock3, Database, FileClock } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { EmptyState, MetricCard } from '@/components/ui/data-display'
import { Page, PageHeader, SectionHeader } from '@/components/ui/page'
import { StatusBadge } from '@/components/ui/status'
import { Surface } from '@/components/ui/surface'

interface InvestigationSession {
  session_id: string
  dataset_id?: string | null
  model_id?: string | null
  state: 'open' | 'complete' | 'failed' | 'archived'
  request_ids: string[]
  question_hashes: string[]
  created_at: string
  updated_at: string
}

interface Workspace {
  active_dataset_id?: string | null
  active_model_id?: string | null
  sessions: InvestigationSession[]
}

export default function SavedInvestigationsPage() {
  const { data, isLoading, error } = useQuery<Workspace>({
    queryKey: ['platform', 'workspace', 'local'],
    queryFn: async () => {
      const response = await fetch('/api/platform/workspaces/local')
      if (!response.ok) throw new Error('Saved investigations are not available.')
      return response.json()
    },
  })

  if (isLoading) return <div role="status" className="grid min-h-[55vh] place-items-center text-sm text-slate-500">Loading investigation history…</div>

  if (error) {
    return <Page><EmptyState tone="danger" title="Investigation history is unavailable" description={error instanceof Error ? error.message : 'Unable to load investigations.'} /></Page>
  }

  const sessions = [...(data?.sessions ?? [])].sort((a, b) => b.updated_at.localeCompare(a.updated_at))

  return (
    <Page>
      <PageHeader
        eyebrow="Investigate · Saved work"
        title="Saved Investigations"
        description="Review the governed investigation trail tied to the datasets and models that were active when each analysis ran."
        actions={<Link href="/advisor"><Button><Brain className="h-4 w-4" />New investigation</Button></Link>}
      />

      <section className="grid gap-3 sm:grid-cols-3">
        <MetricCard icon={<FileClock className="h-4 w-4" />} label="Investigations" value={sessions.length.toLocaleString()} />
        <MetricCard icon={<Database className="h-4 w-4" />} label="Active dataset" value={data?.active_dataset_id ? 'Available' : 'None'} status={<StatusBadge tone={data?.active_dataset_id ? 'success' : 'warning'}>{data?.active_dataset_id ? 'Active' : 'Missing'}</StatusBadge>} />
        <MetricCard icon={<CheckCircle2 className="h-4 w-4" />} label="Open investigations" value={sessions.filter((session) => session.state === 'open').length.toLocaleString()} />
      </section>

      {sessions.length === 0 ? (
        <EmptyState
          title="No investigation history yet"
          description="Ask People Intelligence a workforce question. PeopleOS preserves investigation identity and provenance without storing the raw question in the control-plane registry."
          icon={<Brain className="h-5 w-5" />}
          action={<Link href="/advisor"><Button>Start investigating</Button></Link>}
        />
      ) : (
        <Surface padding="none" className="overflow-hidden">
          <div className="p-5">
            <SectionHeader title="Investigation history" description="Raw questions are not persisted here; the registry stores hashes, request IDs and provenance." />
          </div>
          <div className="divide-y divide-slate-100 border-t border-slate-200 dark:divide-white/5 dark:border-white/10">
            {sessions.map((session, index) => (
              <article key={session.session_id} className="grid gap-4 p-5 md:grid-cols-[minmax(0,1fr)_auto] md:items-center">
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-sm font-semibold text-slate-950 dark:text-white">Investigation {sessions.length - index}</span>
                    <SessionStatus state={session.state} />
                  </div>
                  <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-500 dark:text-slate-400">
                    <span className="inline-flex items-center gap-1.5"><Clock3 className="h-3.5 w-3.5" />{new Date(session.updated_at).toLocaleString()}</span>
                    <span>{session.request_ids.length} request{session.request_ids.length === 1 ? '' : 's'}</span>
                    <span>{session.model_id ? 'Model-backed' : 'Deterministic evidence'}</span>
                  </div>
                </div>
                <div className="text-right text-[11px] text-slate-400">
                  Dataset provenance<br />
                  <span className="font-mono">{session.dataset_id ? `${session.dataset_id.slice(0, 18)}…` : 'none'}</span>
                </div>
              </article>
            ))}
          </div>
        </Surface>
      )}
    </Page>
  )
}

function SessionStatus({ state }: { state: InvestigationSession['state'] }) {
  const tone = state === 'failed' ? 'danger' : state === 'open' ? 'accent' : state === 'complete' ? 'success' : 'neutral'
  return <StatusBadge tone={tone}>{state}</StatusBadge>
}
