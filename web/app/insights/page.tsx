'use client'

import Link from 'next/link'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api-client'
import { Activity, ArrowRight, BarChartHorizontal, Heart, Sparkles, UserPlus, Users } from 'lucide-react'
import { EmptyState, Page, PageHeader, StateSummary, Surface } from '@/components/ui'

interface UploadStatus {
  has_data: boolean
  features_enabled?: Record<string, boolean>
}

interface PlatformStatus {
  capabilities?: { predictive_model?: boolean; llm?: boolean }
  data?: { loaded?: boolean }
}

type InsightCard = {
  title: string
  description: string
  href: string
  icon: typeof Users
  available: boolean
  unavailableLabel?: string
}

export default function InsightsPage() {
  const upload = useQuery<UploadStatus>({ queryKey: ['upload', 'status'], queryFn: () => api.upload.getStatus() as Promise<UploadStatus> })
  const platform = useQuery<PlatformStatus>({ queryKey: ['platform', 'status'], queryFn: () => api.getStatus() as Promise<PlatformStatus> })
  const hasData = Boolean(upload.data?.has_data || platform.data?.data?.loaded)
  const features = upload.data?.features_enabled ?? {}

  const cards: InsightCard[] = [
    {
      title: 'Workforce',
      description: 'Headcount, departments, tenure, ratings and recorded attrition patterns.',
      href: '/workforce-health',
      icon: Users,
      available: hasData,
    },
    {
      title: 'Experience',
      description: 'Measured employee-experience signals, response coverage and lifecycle patterns.',
      href: '/employee-experience',
      icon: Heart,
      available: hasData,
      unavailableLabel: 'Add workforce data first',
    },
    {
      title: 'Retention',
      description: platform.data?.capabilities?.predictive_model ? 'Aggregate model-score patterns and cohort retention evidence.' : 'Recorded attrition and cohort retention evidence. Predictive insights remain optional.',
      href: '/flight-risk',
      icon: BarChartHorizontal,
      available: hasData,
    },
    {
      title: 'Compensation',
      description: 'Salary distribution, dispersion and pay-gap screening with explicit coverage.',
      href: '/workforce-health',
      icon: Activity,
      available: hasData && features.compensation !== false,
      unavailableLabel: hasData ? 'Confirm annual pay and currency in Data' : 'Add workforce data first',
    },
    {
      title: 'Hiring',
      description: 'Quality-of-hire cohorts and observed pre-hire/post-hire relationships when those fields are present.',
      href: '/quality-of-hire',
      icon: UserPlus,
      available: hasData,
    },
    {
      title: 'Explore more',
      description: 'Ask PeopleOS to combine available evidence across workforce, pay, structure, fairness and retention.',
      href: '/advisor',
      icon: Sparkles,
      available: hasData,
    },
  ]

  return <Page>
    <PageHeader eyebrow="Insights" title="What would you like to understand?" description="PeopleOS adapts to the data you have. Open an available area, or ask a question and let PeopleOS choose the right evidence." />

    {upload.isLoading || platform.isLoading ? <StateSummary title="Checking what your data supports" description="PeopleOS is preparing the insight areas available for this dataset." tone="info" /> : !hasData ? <EmptyState title="Add data to unlock your insights" description="Start with the fictional sample or add your own workforce file. PeopleOS only shows conclusions your data can support." action={<Link href="/upload" className="inline-flex items-center gap-2 rounded-xl bg-violet-600 px-4 py-2.5 text-sm font-semibold text-white">Add workforce data <ArrowRight className="h-4 w-4" /></Link>} /> : null}

    <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
      {cards.map((card) => {
        const Icon = card.icon
        const body = <Surface padding="lg" className={`h-full transition ${card.available ? 'hover:-translate-y-0.5 hover:shadow-lg hover:shadow-slate-200/40 dark:hover:shadow-none' : 'opacity-60'}`}>
          <div className="flex h-full flex-col">
            <div className="grid h-11 w-11 place-items-center rounded-2xl bg-violet-50 text-violet-600 dark:bg-violet-500/10 dark:text-violet-300"><Icon className="h-5 w-5" /></div>
            <h2 className="mt-5 text-lg font-semibold text-slate-950 dark:text-white">{card.title}</h2>
            <p className="mt-2 flex-1 text-sm leading-6 text-slate-600 dark:text-slate-400">{card.description}</p>
            <div className="mt-5 text-sm font-semibold text-violet-600 dark:text-violet-300">{card.available ? <>Open insight <ArrowRight className="ml-1 inline h-4 w-4" /></> : card.unavailableLabel ?? 'Not available for this dataset'}</div>
          </div>
        </Surface>
        return card.available ? <Link key={card.title} href={card.href} className="block rounded-2xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500">{body}</Link> : <div key={card.title}>{body}</div>
      })}
    </section>

    {hasData && <Surface padding="lg" className="flex flex-col gap-4 bg-slate-950 text-white dark:bg-white dark:text-slate-950 sm:flex-row sm:items-center sm:justify-between">
      <div><div className="text-xs font-bold uppercase tracking-[0.16em] text-violet-300 dark:text-violet-700">Not sure where to start?</div><h2 className="mt-2 text-xl font-semibold">Ask PeopleOS what deserves attention.</h2><p className="mt-1 text-sm text-slate-300 dark:text-slate-600">It will use only the evidence available in your current dataset.</p></div>
      <Link href="/advisor" className="inline-flex shrink-0 items-center justify-center gap-2 rounded-xl bg-white px-4 py-2.5 text-sm font-semibold text-slate-950 dark:bg-slate-950 dark:text-white"><Sparkles className="h-4 w-4" />Ask PeopleOS</Link>
    </Surface>}
  </Page>
}
