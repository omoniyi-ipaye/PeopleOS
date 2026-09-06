'use client'

import Link from 'next/link'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api-client'
import {
  Activity,
  ArrowRight,
  Brain,
  Database,
  Heart,
  ShieldCheck,
  Sparkles,
  TrendingDown,
  Users,
} from 'lucide-react'
import type { AnalyticsSummary, DepartmentList } from '@/types/api'

interface PlatformStatus {
  data?: { loaded?: boolean; row_count?: number; active_dataset?: boolean }
  capabilities?: { predictive_model?: boolean; llm?: boolean }
  workspace?: { active_dataset?: boolean; active_model?: boolean; dataset_versions?: number }
}

function percentage(value?: number) {
  if (value === undefined || value === null) return '—'
  return `${(value * 100).toFixed(1)}%`
}

export default function DecisionCockpitPage() {
  const { data: status, isLoading: statusLoading } = useQuery<PlatformStatus>({
    queryKey: ['platform', 'status'],
    queryFn: api.getStatus as never,
  })

  const hasData = Boolean(status?.data?.loaded)

  const { data: summary, isLoading: summaryLoading } = useQuery<AnalyticsSummary>({
    queryKey: ['analytics', 'summary'],
    queryFn: api.analytics.getSummary as never,
    enabled: hasData,
  })

  const { data: departmentData } = useQuery<DepartmentList>({
    queryKey: ['analytics', 'departments'],
    queryFn: api.analytics.getDepartments as never,
    enabled: hasData,
  })

  if (statusLoading) {
    return (
      <div className="grid min-h-[65vh] place-items-center">
        <div className="text-center">
          <div className="mx-auto mb-4 grid h-12 w-12 place-items-center rounded-2xl bg-violet-100 text-violet-600 dark:bg-violet-500/10 dark:text-violet-300">
            <Sparkles className="h-6 w-6 animate-pulse" />
          </div>
          <div className="font-medium text-slate-700 dark:text-slate-200">Preparing your workforce briefing…</div>
        </div>
      </div>
    )
  }

  if (!hasData) {
    return (
      <div className="mx-auto flex min-h-[70vh] max-w-4xl items-center justify-center p-4">
        <div className="w-full rounded-[32px] border border-slate-200 bg-white p-8 shadow-xl shadow-slate-200/40 dark:border-white/10 dark:bg-slate-900 dark:shadow-none md:p-12">
          <div className="mb-6 grid h-14 w-14 place-items-center rounded-2xl bg-violet-600 text-white shadow-lg shadow-violet-600/20">
            <Database className="h-7 w-7" />
          </div>
          <div className="max-w-2xl">
            <div className="mb-2 text-xs font-bold uppercase tracking-[0.18em] text-violet-600 dark:text-violet-300">Start here</div>
            <h1 className="text-3xl font-semibold tracking-tight text-slate-950 dark:text-white md:text-4xl">Give PeopleOS a workforce dataset.</h1>
            <p className="mt-4 text-base leading-7 text-slate-600 dark:text-slate-300">
              PeopleOS will validate the data, tell you what can be analysed immediately, explain which capabilities need a trained model, and then build your first decision briefing.
            </p>
          </div>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link href="/upload" className="inline-flex items-center gap-2 rounded-xl bg-violet-600 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-violet-600/20 transition hover:bg-violet-500">
              Add workforce data <ArrowRight className="h-4 w-4" />
            </Link>
            <Link href="/advisor" className="inline-flex items-center gap-2 rounded-xl border border-slate-200 px-5 py-3 text-sm font-semibold text-slate-700 transition hover:bg-slate-50 dark:border-white/10 dark:text-slate-200 dark:hover:bg-white/5">
              See how investigations work
            </Link>
          </div>
        </div>
      </div>
    )
  }

  const departments = departmentData?.departments ?? []
  const largestDepartment = [...departments].sort((a, b) => b.headcount - a.headcount)[0]
  const turnover = summary?.turnover_rate ?? 0
  const rating = summary?.lastrating_mean ?? 0
  const tenure = summary?.tenure_mean ?? 0
  const activeCount = summary?.active_count ?? summary?.headcount ?? status?.data?.row_count ?? 0

  const signals = [
    {
      tone: turnover >= 0.15 ? 'attention' : 'stable',
      icon: TrendingDown,
      title: turnover >= 0.15 ? 'Turnover deserves attention' : 'Turnover is within the current watch range',
      detail: `${percentage(turnover)} departure rate across the active dataset.`,
      href: '/workforce-health',
      action: 'Understand the pattern',
    },
    {
      tone: rating > 0 && rating < 3 ? 'attention' : 'stable',
      icon: Heart,
      title: rating > 0 && rating < 3 ? 'Performance experience may need investigation' : 'Performance signals look broadly stable',
      detail: rating ? `${rating.toFixed(1)} / 5 average recorded rating.` : 'No reliable rating signal is available.',
      href: '/employee-experience',
      action: 'Explore experience',
    },
    {
      tone: 'context',
      icon: Users,
      title: largestDepartment ? `${largestDepartment.dept} is your largest workforce segment` : 'Workforce structure is available',
      detail: largestDepartment ? `${largestDepartment.headcount.toLocaleString()} people are in this department.` : 'Open workforce health to inspect department structure.',
      href: '/workforce-health',
      action: 'Explore structure',
    },
  ] as const

  return (
    <div className="mx-auto max-w-[1440px] space-y-6 pb-10">
      <section className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <div className="mb-2 text-xs font-bold uppercase tracking-[0.18em] text-violet-600 dark:text-violet-300">Decision cockpit</div>
          <h1 className="text-3xl font-semibold tracking-tight text-slate-950 dark:text-white md:text-4xl">What deserves your attention?</h1>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-600 dark:text-slate-400">
            Start with the signal, then move into evidence, investigation, or planning. PeopleOS keeps model and data state visible throughout.
          </p>
        </div>
        <Link href="/advisor" className="inline-flex items-center justify-center gap-2 rounded-xl bg-slate-950 px-5 py-3 text-sm font-semibold text-white shadow-lg transition hover:bg-slate-800 dark:bg-white dark:text-slate-950 dark:hover:bg-slate-100">
          <Brain className="h-4 w-4" /> Ask PeopleOS
        </Link>
      </section>

      <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <Metric label="Active workforce" value={activeCount.toLocaleString()} note="people in the current dataset" icon={Users} />
        <Metric label="Departure rate" value={percentage(turnover)} note="observed workforce outcome" icon={TrendingDown} />
        <Metric label="Average tenure" value={tenure ? `${tenure.toFixed(1)}y` : '—'} note="workforce continuity" icon={Activity} />
        <Metric label="Trust state" value={status?.workspace?.active_model ? 'Model active' : 'Evidence-first'} note={status?.capabilities?.llm ? 'local AI synthesis available' : 'deterministic synthesis'} icon={ShieldCheck} />
      </section>

      <section className="grid gap-6 xl:grid-cols-[minmax(0,1.35fr)_minmax(340px,0.65fr)]">
        <div className="rounded-[28px] border border-slate-200 bg-white p-6 shadow-sm dark:border-white/10 dark:bg-slate-900 md:p-7">
          <div className="mb-5 flex items-start justify-between gap-4">
            <div>
              <div className="text-xs font-bold uppercase tracking-[0.16em] text-slate-400">Priority briefing</div>
              <h2 className="mt-1 text-xl font-semibold text-slate-950 dark:text-white">Signals worth reviewing now</h2>
            </div>
            <span className="rounded-full bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-300">Dataset active</span>
          </div>
          <div className="space-y-3">
            {signals.map((signal) => (
              <Link key={signal.title} href={signal.href} className="group flex items-start gap-4 rounded-2xl border border-slate-200/80 p-4 transition hover:border-violet-200 hover:bg-violet-50/40 dark:border-white/10 dark:hover:border-violet-500/20 dark:hover:bg-violet-500/[0.04]">
                <div className={`mt-0.5 grid h-10 w-10 shrink-0 place-items-center rounded-xl ${signal.tone === 'attention' ? 'bg-amber-100 text-amber-700 dark:bg-amber-500/10 dark:text-amber-300' : signal.tone === 'stable' ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-300' : 'bg-slate-100 text-slate-600 dark:bg-white/5 dark:text-slate-300'}`}>
                  <signal.icon className="h-5 w-5" />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="font-semibold text-slate-900 dark:text-white">{signal.title}</div>
                  <div className="mt-1 text-sm leading-6 text-slate-600 dark:text-slate-400">{signal.detail}</div>
                  <div className="mt-2 inline-flex items-center gap-1 text-xs font-semibold text-violet-600 group-hover:gap-2 dark:text-violet-300">{signal.action}<ArrowRight className="h-3.5 w-3.5" /></div>
                </div>
              </Link>
            ))}
          </div>
        </div>

        <div className="space-y-6">
          <div className="rounded-[28px] border border-slate-200 bg-gradient-to-br from-slate-950 to-slate-900 p-6 text-white shadow-xl dark:border-white/10">
            <div className="mb-5 grid h-11 w-11 place-items-center rounded-2xl bg-violet-500/15 text-violet-300 ring-1 ring-violet-400/20">
              <Brain className="h-5 w-5" />
            </div>
            <div className="text-xs font-bold uppercase tracking-[0.16em] text-violet-300">Investigate</div>
            <h2 className="mt-2 text-xl font-semibold">Ask why, not just what.</h2>
            <p className="mt-2 text-sm leading-6 text-slate-300">People Intelligence selects governed tools, shows the evidence it used, and tells you when the evidence is incomplete.</p>
            <Link href="/advisor" className="mt-5 inline-flex items-center gap-2 rounded-xl bg-white px-4 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-slate-100">Start an investigation <ArrowRight className="h-4 w-4" /></Link>
          </div>

          <div className="rounded-[28px] border border-slate-200 bg-white p-6 dark:border-white/10 dark:bg-slate-900">
            <div className="text-xs font-bold uppercase tracking-[0.16em] text-slate-400">Capability state</div>
            <div className="mt-4 space-y-3 text-sm">
              <StateRow label="Dataset" value="Active" good />
              <StateRow label="Predictive model" value={status?.workspace?.active_model ? 'Active' : 'Not active'} good={Boolean(status?.workspace?.active_model)} />
              <StateRow label="AI synthesis" value={status?.capabilities?.llm ? 'Local model' : 'Deterministic fallback'} good />
            </div>
            <Link href="/platform" className="mt-5 inline-flex items-center gap-1 text-xs font-semibold text-violet-600 dark:text-violet-300">Open Trust Center <ArrowRight className="h-3.5 w-3.5" /></Link>
          </div>
        </div>
      </section>

      {summaryLoading && <div className="text-xs text-slate-400">Refreshing workforce signals…</div>}
    </div>
  )
}

function Metric({ label, value, note, icon: Icon }: { label: string; value: string; note: string; icon: typeof Users }) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm dark:border-white/10 dark:bg-slate-900">
      <div className="flex items-center justify-between gap-3">
        <div className="text-xs font-semibold text-slate-500 dark:text-slate-400">{label}</div>
        <Icon className="h-4 w-4 text-violet-500" />
      </div>
      <div className="mt-3 text-2xl font-semibold tracking-tight text-slate-950 dark:text-white">{value}</div>
      <div className="mt-1 text-xs text-slate-500 dark:text-slate-400">{note}</div>
    </div>
  )
}

function StateRow({ label, value, good }: { label: string; value: string; good: boolean }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <span className="text-slate-500 dark:text-slate-400">{label}</span>
      <span className="inline-flex items-center gap-2 font-medium text-slate-800 dark:text-slate-200">
        <span className={`h-2 w-2 rounded-full ${good ? 'bg-emerald-500' : 'bg-slate-300 dark:bg-slate-600'}`} />
        {value}
      </span>
    </div>
  )
}
