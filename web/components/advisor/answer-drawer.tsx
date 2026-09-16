'use client'

import { Bot, CheckCircle2, X } from 'lucide-react'
import { Button, StateSummary, Surface } from '@/components/ui'
import { AgentAnswer, AgentNextAction, InvestigationResult } from '@/components/advisor/investigation-result'

interface AnswerDrawerProps {
  result: { answer: AgentAnswer; source: string; reportingCurrency?: string } | null
  loading: boolean
  elapsedSeconds?: number
  error: string | null
  followUps: AgentNextAction[]
  onClose: () => void
  onCancel: () => void
  onInvestigate: (question: string) => void
}

export function AnswerDrawer({ result, loading, elapsedSeconds = 0, error, followUps, onClose, onCancel, onInvestigate }: AnswerDrawerProps) {
  if (!loading && !result && !error) return null

  return <div className="fixed inset-0 z-[80]" role="presentation">
    <button type="button" aria-label="Close answer panel" onClick={onClose} className="absolute inset-0 h-full w-full cursor-default bg-slate-950/35 backdrop-blur-[2px] dark:bg-black/55" />
    <section role="dialog" aria-modal="true" aria-labelledby="peopleos-answer-drawer-title" aria-describedby="peopleos-answer-drawer-description" className="relative flex h-full w-full max-w-[min(760px,100vw)] animate-answer-drawer-in flex-col overflow-hidden border-r border-slate-200 bg-slate-50 shadow-2xl dark:border-white/10 dark:bg-slate-950">
      <header className="flex shrink-0 items-start justify-between gap-4 border-b border-slate-200/80 bg-white/95 px-5 py-4 backdrop-blur-xl dark:border-white/10 dark:bg-slate-950/95 sm:px-7">
        <div className="flex min-w-0 items-start gap-3">
          <span className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-violet-100 text-violet-700 dark:bg-violet-500/15 dark:text-violet-200"><Bot className="h-4 w-4" aria-hidden="true" /></span>
          <div className="min-w-0"><p className="text-xs font-bold uppercase tracking-[0.16em] text-violet-600 dark:text-violet-300">PeopleOS workspace</p><h2 id="peopleos-answer-drawer-title" className="mt-1 truncate text-lg font-semibold text-slate-950 dark:text-white">Your answer</h2><p id="peopleos-answer-drawer-description" className="mt-1 text-xs leading-5 text-slate-500 dark:text-slate-400">Review the result, its evidence and the next useful question without leaving the conversation.</p></div>
        </div>
        <Button type="button" variant="ghost" size="icon" aria-label="Close answer details" title="Close answer panel" onClick={onClose}><X className="h-5 w-5" aria-hidden="true" /></Button>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-5 sm:px-7 sm:py-7">
        {loading && <><Surface padding="md" className="border-violet-200/70 bg-violet-50/40 dark:border-violet-500/20 dark:bg-violet-500/[0.04]"><div role="status" aria-live="polite"><div className="flex items-start gap-3"><span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-violet-100 text-violet-700 dark:bg-violet-500/15 dark:text-violet-200"><Bot className="h-4 w-4 animate-pulse" aria-hidden="true" /></span><div><p className="font-semibold text-text-primary">PeopleOS is working through it…</p><p className="mt-1 text-sm leading-6 text-text-secondary">The agent is gathering approved checks, checking the evidence and preparing a useful answer. {elapsedSeconds > 0 ? `${elapsedSeconds}s elapsed.` : 'This has just started.'}</p></div></div><ol className="mt-4 space-y-2 text-sm text-text-secondary"><li className="rounded-xl border border-violet-100 bg-white/70 px-3 py-2 dark:border-violet-500/15 dark:bg-white/[0.04]">1. Understand the question</li><li className="rounded-xl border border-violet-100 bg-white/70 px-3 py-2 dark:border-violet-500/15 dark:bg-white/[0.04]">2. Gather verified evidence</li><li className="rounded-xl border border-violet-100 bg-white/70 px-3 py-2 dark:border-violet-500/15 dark:bg-white/[0.04]">3. Check whether it is enough</li><li className="rounded-xl border border-violet-100 bg-white/70 px-3 py-2 dark:border-violet-500/15 dark:bg-white/[0.04]">4. Explain and suggest a next check</li></ol></div></Surface><Button className="mt-4" type="button" variant="secondary" size="sm" onClick={onCancel}>Stop waiting</Button></>}
        {error && <div role="alert"><StateSummary title="No answer shown" description={error} tone="warning" /></div>}
        {result && !loading && <>
          <InvestigationResult result={result.answer} source={result.source} reportingCurrency={result.reportingCurrency} />
          {followUps.length > 0 && <Surface padding="md" className="mt-4 border-violet-100/80 bg-violet-50/30 dark:border-violet-500/15 dark:bg-violet-500/[0.03]"><div><div className="text-sm font-semibold text-text-primary">Keep exploring</div><div className="mt-1 text-xs leading-5 text-text-secondary">Ask a bounded follow-up without losing this conversation.</div></div><div className="mt-3 flex flex-wrap gap-2">{followUps.map(action => <Button key={`${action.label}-${action.question}`} type="button" variant="secondary" size="sm" disabled={loading} title={action.reason} onClick={() => onInvestigate(action.question)}>{action.label}</Button>)}</div></Surface>}
          <div className="mt-5 flex items-center gap-2 text-xs text-text-muted"><CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" aria-hidden="true" />The conversation stays open behind this panel.</div>
        </>}
      </div>
    </section>
  </div>
}
