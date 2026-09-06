'use client'

import Link from 'next/link'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api-client'
import { Activity, ArrowRight, Brain, Database, Sparkles, TrendingDown, Users } from 'lucide-react'
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
  const { data: status, isLoading: statusLoading } = useQuery<PlatformStatus>({ queryKey: ['platform', 'status'], queryFn: () => api.getStatus() as Promise<PlatformStatus> })
  const hasData = Boolean(status?.data?.loaded)
  const { data: summary, isLoading: summaryLoading } = useQuery<AnalyticsSummary>({ queryKey: ['analytics', 'summary'], queryFn: () => api.analytics.getSummary() as Promise<AnalyticsSummary>, enabled: hasData })
  const { data: departmentData } = useQuery<DepartmentList>({ queryKey: ['analytics', 'departments'], queryFn: () => api.analytics.getDepartments() as Promise<DepartmentList>, enabled: hasData })

  if (statusLoading) return <div className="grid min-h-[65vh] place-items-center"><div className="text-center"><div className="mx-auto mb-4 grid h-12 w-12 place-items-center rounded-2xl bg-violet-100 text-violet-600 dark:bg-violet-500/10 dark:text-violet-300"><Sparkles className="h-6 w-6 animate-pulse" /></div><div className="font-medium text-slate-700 dark:text-slate-200">Preparing your workforce briefing…</div></div></div>

  if (!hasData) return <div className="mx-auto flex min-h-[70vh] max-w-4xl items-center justify-center p-4"><div className="w-full rounded-3xl border border-slate-200 bg-white p-8 shadow-xl shadow-slate-200/40 dark:border-white/10 dark:bg-slate-900 dark:shadow-none md:p-12"><div className="mb-6 grid h-14 w-14 place-items-center rounded-2xl bg-violet-600 text-white shadow-lg shadow-violet-600/20"><Database className="h-7 w-7" /></div><div className="max-w-2xl"><div className="mb-2 text-xs font-bold uppercase tracking-[0.18em] text-violet-600 dark:text-violet-300">Start here</div><h1 className="text-3xl font-semibold tracking-tight text-slate-950 dark:text-white md:text-4xl">Give PeopleOS a workforce dataset.</h1><p className="mt-4 text-base leading-7 text-slate-600 dark:text-slate-300">PeopleOS will validate the data, establish the current population, tell you which evidence is available immediately, and keep predictive or causal capabilities behind explicit governed boundaries.</p></div><div className="mt-8 flex flex-wrap gap-3"><Link href="/upload" className="inline-flex items-center gap-2 rounded-xl bg-violet-600 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-violet-600/20 transition hover:bg-violet-500">Add workforce data <ArrowRight className="h-4 w-4" /></Link><Link href="/advisor" className="inline-flex items-center gap-2 rounded-xl border border-slate-200 px-5 py-3 text-sm font-semibold text-slate-700 transition hover:bg-slate-50 dark:border-white/10 dark:text-slate-200 dark:hover:bg-white/5">See how investigations work</Link></div></div></div>

  const departments = departmentData?.departments ?? []
  const largestDepartment = [...departments].sort((a, b) => b.headcount - a.headcount)[0]
  const observedAttritionShare = summary?.observed_attrition_share ?? summary?.turnover_rate ?? 0
  const rating = summary?.lastrating_mean ?? 0
  const tenure = summary?.tenure_mean ?? 0
  const activeCount = summary?.active_count ?? summary?.headcount ?? status?.data?.row_count ?? 0

  const signals = [
    {
      tone: observedAttritionShare >= 0.15 ? 'attention' : 'stable',
      icon: TrendingDown,
      title: observedAttritionShare >= 0.15 ? 'Recorded attrition share deserves investigation' : 'Recorded attrition share is below the current watch threshold',
      detail: `${percentage(observedAttritionShare)} of current employee outcome records are marked departed. This is not a period turnover rate.`,
      href: '/workforce-health',
      action: 'Understand the pattern',
    },
    {
      tone: 'context',
      icon: Activity,
      title: rating ? 'Performance rating context is available' : 'No reliable rating context is available',
      detail: rating ? `${rating.toFixed(1)} / 5 average recorded rating among active employees. Treat this as performance context, not an experience measure.` : 'Add valid performance measurements if this context is needed.',
      href: '/workforce-health',
      action: 'Review workforce evidence',
    },
    {
      tone: 'context',
      icon: Users,
      title: largestDepartment ? `${largestDepartment.dept} is the largest active workforce segment` : 'Workforce structure is available',
      detail: largestDepartment ? `${largestDepartment.headcount.toLocaleString()} active people are represented in this department.` : 'Open Workforce Health to inspect current structure.',
      href: '/workforce-health',
      action: 'Explore structure',
    },
  ] as const

  return <div className="mx-auto max-w-[1440px] space-y-6 pb-10">
    <section className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between"><div><div className="mb-2 text-xs font-bold uppercase tracking-[0.18em] text-violet-600 dark:text-violet-300">Decision cockpit</div><h1 className="text-3xl font-semibold tracking-tight text-slate-950 dark:text-white md:text-4xl">What deserves your attention?</h1><p className="mt-2 max-w-2xl text-sm leading-6 text-slate-600 dark:text-slate-400">Start with the workforce signals that matter. Evidence boundaries stay available without competing with the decision itself.</p></div><Link href="/advisor" className="inline-flex items-center justify-center gap-2 rounded-xl bg-slate-950 px-5 py-3 text-sm font-semibold text-white shadow-lg transition hover:bg-slate-800 dark:bg-white dark:text-slate-950 dark:hover:bg-slate-100"><Brain className="h-4 w-4" /> Ask PeopleOS</Link></section>

    <section className="grid gap-3 sm:grid-cols-3"><Metric label="Active workforce" value={activeCount.toLocaleString()} note="current employees in the resolved population" icon={Users} /><Metric label="Observed attrition share" value={percentage(observedAttritionShare)} note="recorded outcome share; not period turnover" icon={TrendingDown} /><Metric label="Average active tenure" value={tenure ? `${tenure.toFixed(1)}y` : '—'} note="current active workforce" icon={Activity} /></section>

    <section className="grid gap-6 xl:grid-cols-[minmax(0,1.45fr)_minmax(320px,0.55fr)]"><div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm dark:border-white/10 dark:bg-slate-900 md:p-7"><div className="mb-5"><div className="text-xs font-bold uppercase tracking-[0.16em] text-slate-400">Priority briefing</div><h2 className="mt-1 text-xl font-semibold text-slate-950 dark:text-white">Signals worth reviewing now</h2></div><div className="space-y-3">{signals.map((signal) => <Link key={signal.title} href={signal.href} className="group flex items-start gap-4 rounded-2xl border border-slate-200/80 p-4 transition hover:border-violet-200 hover:bg-violet-50/40 dark:border-white/10 dark:hover:border-violet-500/20 dark:hover:bg-violet-500/[0.04]"><div className={`mt-0.5 grid h-10 w-10 shrink-0 place-items-center rounded-xl ${signal.tone === 'attention' ? 'bg-amber-100 text-amber-700 dark:bg-amber-500/10 dark:text-amber-300' : 'bg-slate-100 text-slate-600 dark:bg-white/5 dark:text-slate-300'}`}><signal.icon className="h-5 w-5" /></div><div className="min-w-0 flex-1"><div className="font-semibold text-slate-900 dark:text-white">{signal.title}</div><div className="mt-1 text-sm leading-6 text-slate-600 dark:text-slate-400">{signal.detail}</div><div className="mt-2 inline-flex items-center gap-1 text-xs font-semibold text-violet-600 group-hover:gap-2 dark:text-violet-300">{signal.action}<ArrowRight className="h-3.5 w-3.5" /></div></div></Link>)}</div></div>

      <div className="rounded-3xl border border-slate-200 bg-gradient-to-br from-slate-950 to-slate-900 p-6 text-white shadow-xl dark:border-white/10"><div className="mb-5 grid h-11 w-11 place-items-center rounded-2xl bg-violet-500/15 text-violet-300 ring-1 ring-violet-400/20"><Brain className="h-5 w-5" /></div><div className="text-xs font-bold uppercase tracking-[0.16em] text-violet-300">Investigate</div><h2 className="mt-2 text-xl font-semibold">Ask what the evidence supports.</h2><p className="mt-2 text-sm leading-6 text-slate-300">Move from a signal into a governed investigation with evidence, coverage and limitations available when you need them.</p><Link href="/advisor" className="mt-5 inline-flex items-center gap-2 rounded-xl bg-white px-4 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-slate-100">Start an investigation <ArrowRight className="h-4 w-4" /></Link></div>
    </section>

    <div className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-200 pt-4 text-xs text-slate-400 dark:border-white/10"><span>{status?.workspace?.active_model ? 'Predictive model active' : 'Evidence-first mode'} · {status?.capabilities?.llm ? 'Local synthesis available' : 'Deterministic synthesis'}</span><Link href="/platform" className="font-semibold text-slate-500 transition hover:text-violet-600 dark:text-slate-400 dark:hover:text-violet-300">Trust details <ArrowRight className="inline h-3 w-3" /></Link></div>
    {summaryLoading && <div className="text-xs text-slate-400">Refreshing workforce evidence…</div>}
  </div>
}

function Metric({ label, value, note, icon: Icon }: { label: string; value: string; note: string; icon: typeof Users }) { return <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm dark:border-white/10 dark:bg-slate-900"><div className="flex items-center justify-between gap-3"><div className="text-xs font-semibold text-slate-500 dark:text-slate-400">{label}</div><Icon className="h-4 w-4 text-violet-500" /></div><div className="mt-3 text-2xl font-semibold tracking-tight text-slate-950 dark:text-white">{value}</div><div className="mt-1 text-xs text-slate-500 dark:text-slate-400">{note}</div></div> }
