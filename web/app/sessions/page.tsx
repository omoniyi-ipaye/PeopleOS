'use client'

import Link from 'next/link'
import { useQuery } from '@tanstack/react-query'
import {
  ArrowRight,
  Brain,
  CheckCircle2,
  Clock3,
  Database,
  FileClock,
} from 'lucide-react'

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

  if (isLoading) {
    return <div className="grid min-h-[55vh] place-items-center text-sm text-slate-500">Loading investigation history…</div>
  }

  if (error) {
    return <div className="rounded-2xl border border-red-200 bg-red-50 p-5 text-sm text-red-700 dark:border-red-500/20 dark:bg-red-500/[0.05] dark:text-red-300">{error instanceof Error ? error.message : 'Unable to load investigations.'}</div>
  }

  const sessions = [...(data?.sessions ?? [])].sort((a, b) => b.updated_at.localeCompare(a.updated_at))

  return (
    <div className="mx-auto max-w-6xl space-y-6 pb-10">
      <section className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
        <div>
          <div className="mb-2 text-xs font-bold uppercase tracking-[0.18em] text-violet-600 dark:text-violet-300">Investigate · Saved work</div>
          <h1 className="text-3xl font-semibold tracking-tight text-slate-950 dark:text-white">Saved Investigations</h1>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600 dark:text-slate-400">
            Review the governed investigation trail tied to the datasets and models that were active when each analysis ran.
          </p>
        </div>
        <Link href="/advisor" className="inline-flex items-center justify-center gap-2 rounded-xl bg-violet-600 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-violet-600/20 transition hover:bg-violet-500">
          <Brain className="h-4 w-4" /> New investigation
        </Link>
      </section>

      <section className="grid gap-3 sm:grid-cols-3">
        <Metric icon={FileClock} label="Investigations" value={sessions.length.toLocaleString()} />
        <Metric icon={Database} label="Active dataset" value={data?.active_dataset_id ? 'Available' : 'None'} />
        <Metric icon={CheckCircle2} label="Open investigations" value={sessions.filter((session) => session.state === 'open').length.toLocaleString()} />
      </section>

      {sessions.length === 0 ? (
        <section className="rounded-[30px] border border-dashed border-slate-300 bg-white p-10 text-center dark:border-white/15 dark:bg-slate-900">
          <div className="mx-auto grid h-14 w-14 place-items-center rounded-2xl bg-violet-100 text-violet-600 dark:bg-violet-500/10 dark:text-violet-300"><Brain className="h-6 w-6" /></div>
          <h2 className="mt-5 text-xl font-semibold text-slate-950 dark:text-white">No investigation history yet</h2>
          <p className="mx-auto mt-2 max-w-lg text-sm leading-6 text-slate-500 dark:text-slate-400">Ask People Intelligence a workforce question. PeopleOS will preserve the investigation identity and provenance without storing the raw question in the control-plane registry.</p>
          <Link href="/advisor" className="mt-6 inline-flex items-center gap-2 rounded-xl bg-slate-950 px-4 py-2.5 text-sm font-semibold text-white dark:bg-white dark:text-slate-950">Start investigating <ArrowRight className="h-4 w-4" /></Link>
        </section>
      ) : (
        <section className="rounded-[28px] border border-slate-200 bg-white shadow-sm dark:border-white/10 dark:bg-slate-900">
          <div className="border-b border-slate-200 p-5 dark:border-white/10">
            <h2 className="font-semibold text-slate-950 dark:text-white">Investigation history</h2>
            <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">Raw questions are not persisted here; the registry stores hashes, request IDs and provenance.</p>
          </div>
          <div className="divide-y divide-slate-100 dark:divide-white/5">
            {sessions.map((session, index) => (
              <article key={session.session_id} className="grid gap-4 p-5 md:grid-cols-[minmax(0,1fr)_auto] md:items-center">
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-sm font-semibold text-slate-950 dark:text-white">Investigation {sessions.length - index}</span>
                    <StateBadge state={session.state} />
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
        </section>
      )}
    </div>
  )
}

function Metric({ icon: Icon, label, value }: { icon: React.ElementType; label: string; value: string }) {
  return <div className="rounded-2xl border border-slate-200 bg-white p-5 dark:border-white/10 dark:bg-slate-900"><Icon className="h-4 w-4 text-violet-500" /><div className="mt-3 text-2xl font-semibold text-slate-950 dark:text-white">{value}</div><div className="mt-1 text-xs text-slate-500 dark:text-slate-400">{label}</div></div>
}

function StateBadge({ state }: { state: InvestigationSession['state'] }) {
  const tone = state === 'failed'
    ? 'bg-red-100 text-red-700 dark:bg-red-500/10 dark:text-red-300'
    : state === 'open'
      ? 'bg-violet-100 text-violet-700 dark:bg-violet-500/10 dark:text-violet-300'
      : 'bg-slate-100 text-slate-600 dark:bg-white/5 dark:text-slate-300'
  return <span className={`rounded-full px-2.5 py-1 text-[11px] font-semibold capitalize ${tone}`}>{state}</span>
}
