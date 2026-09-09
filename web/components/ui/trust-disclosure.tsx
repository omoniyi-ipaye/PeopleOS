'use client'

import { ChevronDown, ShieldCheck } from 'lucide-react'
import { ReactNode } from 'react'

export function TrustDisclosure({ title = 'About this analysis', summary, children }: { title?: string; summary?: string; children: ReactNode }) {
  return <details className="group rounded-2xl border border-slate-200/80 bg-white/70 dark:border-white/10 dark:bg-white/[0.02]">
    <summary className="flex cursor-pointer list-none items-center justify-between gap-4 px-4 py-3 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500">
      <div className="flex min-w-0 items-center gap-2.5"><ShieldCheck className="h-4 w-4 shrink-0 text-emerald-600 dark:text-emerald-400" /><div className="min-w-0"><span className="font-semibold text-slate-800 dark:text-slate-100">{title}</span>{summary && <span className="ml-2 text-slate-500 dark:text-slate-400">{summary}</span>}</div></div>
      <ChevronDown className="h-4 w-4 shrink-0 text-slate-400 transition group-open:rotate-180" />
    </summary>
    <div className="border-t border-slate-200/70 px-4 py-4 text-sm leading-6 text-slate-600 dark:border-white/10 dark:text-slate-400">{children}</div>
  </details>
}
