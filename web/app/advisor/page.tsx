'use client'

import { FormEvent, type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { ArrowRight, Bot, Clock3, Search, Sparkles, Square } from 'lucide-react'
import { Button, Page, PageHeader, StateSummary, Surface, Textarea } from '@/components/ui'
import { api } from '@/lib/api-client'
import { AgentAnswer, AgentNextAction } from '@/components/advisor/investigation-result'
import { AnswerDrawer } from '@/components/advisor/answer-drawer'

interface RuntimeStatus { data?: { loaded?: boolean }; integrity?: { status: string; snapshot?: { generation: string; dataset_id: string; dataset_version?: number; source_name?: string; reporting_currency?: string } } }

const suggestedQuestions = [
  ['Start with workforce structure', 'Headcount by department'],
  ['Where is recorded attrition?', 'Recorded attrition share by department'],
  ['How does pay vary?', 'Average salary by department'],
  ['How are teams structured?', 'Headcount by role'],
]

function deeperQuestions(question: string) {
  const q = question.toLowerCase()
  if (/salary|pay|compensation/.test(q)) return [
    ['By department', 'Average salary by department'],
    ['By location', 'Average salary by location'],
    ['By job level', 'Average salary by job level'],
    ['Salary vs tenure', 'Correlation between salary and tenure'],
  ]
  if (/attrition|departure|retention|turnover/.test(q)) return [
    ['By department', 'Recorded attrition share by department'],
    ['By location', 'Recorded attrition share by location'],
    ['By job level', 'Recorded attrition share by job level'],
    ['Compare workforce size', 'Headcount by department'],
  ]
  if (/headcount|how many|workforce size|employee count/.test(q)) return [
    ['By department', 'Headcount by department'],
    ['By location', 'Headcount by location'],
    ['By job level', 'Headcount by job level'],
    ['Role mix', 'Headcount by role'],
  ]
  if (/experience|engagement|pulse|enps/.test(q)) return [
    ['Workforce by department', 'Headcount by department'],
    ['Tenure relationship', 'Correlation between tenure and performance rating'],
    ['Location mix', 'Headcount by location'],
  ]
  return [
    ['Workforce by department', 'Headcount by department'],
    ['Pay by department', 'Average salary by department'],
    ['Attrition by department', 'Recorded attrition share by department'],
    ['Location mix', 'Headcount by location'],
  ]
}

interface QuestionComposerProps {
  question: string
  onQuestionChange: (value: string) => void
  onSubmit: (event: FormEvent<HTMLFormElement>) => void
  onInvestigate: (prompt: string) => void
  onCancel: () => void
  loading: boolean
  elapsedSeconds: number
  ready: boolean
}

function investigationProgress(elapsedSeconds: number) {
  if (elapsedSeconds >= 30) return { title: 'Still checking the local analysis service…', detail: 'This can happen when the local model is waking up. You can keep waiting or stop and retry; your workforce data is not changed.' }
  if (elapsedSeconds >= 12) return { title: 'The evidence is readying the answer…', detail: 'PeopleOS is finishing the approved checks before it asks the language model to explain them.' }
  if (elapsedSeconds >= 5) return { title: 'Gathering verified evidence…', detail: 'The analytical checks run before any optional AI explanation.' }
  return { title: 'Starting the investigation…', detail: 'PeopleOS is checking the question against your verified workforce snapshot.' }
}

function elapsedLabel(seconds: number) {
  return seconds < 1 ? 'just started' : `${seconds}s elapsed`
}

function QuestionComposer({ question, onQuestionChange, onSubmit, onInvestigate, onCancel, loading, elapsedSeconds, ready }: QuestionComposerProps) {
  const progress = investigationProgress(elapsedSeconds)
  return <Surface padding="md" className="border-slate-200/90 bg-white shadow-lg shadow-slate-200/30 dark:border-white/10 dark:bg-slate-950 dark:shadow-black/20">
    <form onSubmit={onSubmit} className="space-y-3" aria-busy={loading}>
      <Textarea label="Message PeopleOS" value={question} onChange={event => onQuestionChange(event.target.value)} placeholder="Ask a workforce question in plain language…" rows={3} minLength={3} maxLength={2000} disabled={loading || !ready} helperText="PeopleOS will choose approved checks, show the evidence and explain what it means." leading={<Search className="h-4 w-4" />} className="min-h-24 resize-none" />
      {loading && <div className="flex flex-col gap-3 rounded-2xl border border-violet-200/80 bg-violet-50/70 px-3.5 py-3 dark:border-violet-500/20 dark:bg-violet-500/[0.07] sm:flex-row sm:items-center sm:justify-between" role="status" aria-live="polite">
        <div className="flex min-w-0 items-start gap-2.5"><Clock3 className="mt-0.5 h-4 w-4 shrink-0 animate-pulse text-violet-600 dark:text-violet-300" /><div className="min-w-0"><p className="text-sm font-semibold text-violet-950 dark:text-violet-100">{progress.title}</p><p className="mt-0.5 text-xs leading-5 text-violet-800/80 dark:text-violet-200/80">{progress.detail} <span className="font-medium">{elapsedLabel(elapsedSeconds)}.</span></p></div></div>
        <Button type="button" variant="secondary" size="sm" onClick={onCancel}><Square className="h-3.5 w-3.5" />Stop waiting</Button>
      </div>}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex flex-wrap gap-2">{suggestedQuestions.map(([label, prompt]) => <Button key={label} type="button" variant="ghost" size="sm" disabled={loading || !ready} title={`Ask: ${prompt}`} onClick={() => onInvestigate(prompt)}>{label}</Button>)}</div>
        <Button type="submit" isLoading={loading} disabled={!ready || loading || question.trim().length < 3}><Sparkles className="h-4 w-4" />Ask PeopleOS</Button>
      </div>
    </form>
  </Surface>
}

function ConversationMessage({ role, children }: { role: 'assistant' | 'user'; children: ReactNode }) {
  const user = role === 'user'
  return <div className={`flex items-start gap-3 ${user ? 'justify-end' : 'justify-start'}`}>
    {!user && <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-violet-100 text-violet-700 dark:bg-violet-500/15 dark:text-violet-200"><Bot className="h-4 w-4" aria-hidden="true" /></span>}
    <div className={`max-w-[min(760px,92%)] rounded-2xl px-4 py-3.5 shadow-sm ${user ? 'rounded-tr-md bg-violet-600 text-white' : 'rounded-tl-md border border-slate-200/90 bg-white text-slate-800 dark:border-white/10 dark:bg-slate-900 dark:text-slate-100'}`}>
      <p className={`mb-1 text-[11px] font-bold uppercase tracking-[0.14em] ${user ? 'text-violet-100' : 'text-violet-600 dark:text-violet-300'}`}>{user ? 'You' : 'PeopleOS'}</p>
      <div className="text-[15px] leading-7">{children}</div>
    </div>
  </div>
}

type ConversationTurn = { id: string; question: string; status: 'loading' | 'complete' | 'cancelled' | 'error'; answer?: AgentAnswer; error?: string }

function Conversation({ history, activeTurnId, onOpenAnswer, onCancel }: { history: ConversationTurn[]; activeTurnId: string | null; onOpenAnswer: () => void; onCancel: () => void }) {
  return <div className="space-y-6">
    {!history.length && <ConversationMessage role="assistant"><p>Ask me anything about the workforce snapshot. I’ll work through the relevant approved checks, show what the data supports and tell you where the evidence stops.</p></ConversationMessage>}
    {history.map((turn, index) => {
      const isCurrent = turn.id === activeTurnId
      const isLatestCompleted = turn.status === 'complete' && index === history.length - 1
      return <div key={turn.id} className="space-y-3">
        <ConversationMessage role="user">{turn.question}</ConversationMessage>
        {turn.status === 'loading' && <ConversationMessage role="assistant"><div className="flex items-center gap-2"><span className="h-2 w-2 animate-pulse rounded-full bg-violet-500" /><span>Working through the evidence…</span></div><p className="mt-1 text-sm text-slate-500 dark:text-slate-400">The answer panel will open as soon as the investigation is ready.</p>{isCurrent && <Button type="button" variant="secondary" size="sm" className="mt-3" onClick={onCancel}><Square className="h-3.5 w-3.5" />Stop waiting</Button>}</ConversationMessage>}
        {turn.status === 'complete' && <ConversationMessage role="assistant"><p>{isLatestCompleted ? 'I’ve prepared a governed workforce readout for that question.' : 'This earlier workforce readout remains part of the conversation.'} {isLatestCompleted ? 'Open the answer panel to see the plain-language result, supporting checks, visuals and bounded follow-ups.' : 'Ask a follow-up below to continue from the current workforce evidence.'}</p>{isLatestCompleted && <div className="mt-3 flex flex-wrap items-center gap-2"><Button type="button" variant="secondary" size="sm" onClick={onOpenAnswer}>Open answer panel <ArrowRight className="h-4 w-4" /></Button><span className="text-xs text-slate-500 dark:text-slate-400">The conversation remains here.</span></div>}</ConversationMessage>}
        {turn.status === 'cancelled' && <ConversationMessage role="assistant"><p className="text-slate-600 dark:text-slate-300">I stopped waiting for this answer. Nothing in your workforce data was changed; you can retry the question when ready.</p></ConversationMessage>}
        {turn.status === 'error' && <ConversationMessage role="assistant"><p className="text-amber-800 dark:text-amber-200">I couldn’t complete this investigation. {turn.error ?? 'Please retry the question.'}</p></ConversationMessage>}
      </div>
    })}
  </div>
}

function PeopleIntelligencePageContent() {
  const status = useQuery<RuntimeStatus>({ queryKey: ['platform', 'status'], queryFn: () => api.getStatus() as Promise<RuntimeStatus>, refetchOnMount: 'always' })
  const snapshot = status.data?.integrity?.snapshot
  const ready = !status.isError && status.data?.data?.loaded && status.data.integrity?.status === 'verified' && Boolean(snapshot)
  const pathname = usePathname()
  const [question, setQuestion] = useState('')
  const [requestedQuestion, setRequestedQuestion] = useState('')
  const [result, setResult] = useState<{ answer: AgentAnswer; generation: string; source: string; reportingCurrency?: string } | null>(null)
  const [conversationHistory, setConversationHistory] = useState<ConversationTurn[]>([])
  const [activeTurnId, setActiveTurnId] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [elapsedSeconds, setElapsedSeconds] = useState(0)
  const [answerOpen, setAnswerOpen] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const controller = useRef<AbortController | null>(null)
  const activeTurnRef = useRef<string | null>(null)
  const initialQueryLoaded = useRef<string | null>(null)
  const initialQueryRun = useRef<string | null>(null)

  useEffect(() => () => controller.current?.abort(), [])
  useEffect(() => {
    if (!loading) return
    const startedAt = Date.now()
    const interval = window.setInterval(() => setElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000)), 1000)
    return () => window.clearInterval(interval)
  }, [loading])
  useEffect(() => {
    if (typeof window === 'undefined') return
    const timer = window.setTimeout(() => {
      setRequestedQuestion(new URLSearchParams(window.location.search).get('q')?.trim().slice(0, 2000) ?? '')
    }, 0)
    return () => window.clearTimeout(timer)
  }, [pathname])
  useEffect(() => {
    if (!requestedQuestion || initialQueryLoaded.current === requestedQuestion) return
    const timer = window.setTimeout(() => {
      initialQueryLoaded.current = requestedQuestion
      setQuestion(requestedQuestion)
    }, 0)
    return () => window.clearTimeout(timer)
  }, [requestedQuestion])
  useEffect(() => {
    if (!answerOpen) return
    const previous = document.activeElement as HTMLElement | null
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    window.setTimeout(() => document.querySelector<HTMLButtonElement>('[aria-label="Close answer details"]')?.focus(), 0)
    function handleKeyDown(event: KeyboardEvent) { if (event.key === 'Escape') setAnswerOpen(false) }
    window.addEventListener('keydown', handleKeyDown)
    return () => {
      window.removeEventListener('keydown', handleKeyDown)
      document.body.style.overflow = previousOverflow
      previous?.focus()
    }
  }, [answerOpen])

  const currentResult = ready && result?.generation === snapshot?.generation ? result : null
  const followUps = useMemo<AgentNextAction[]>(() => {
    if (!currentResult) return []
    if (currentResult.answer.next_actions?.length) return currentResult.answer.next_actions
    return deeperQuestions(currentResult.answer.question).map(([label, prompt]) => ({ label, question: prompt, reason: 'Continue with another verified workforce view.' }))
  }, [currentResult])

  const investigate = useCallback(async (selected?: string) => {
    const prompt = (selected ?? question).trim()
    if (prompt.length < 3 || prompt.length > 2000 || !ready || !snapshot || loading) return
    const abort = new AbortController()
    const turnId = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
    let timedOut = false
    const timeout = window.setTimeout(() => { timedOut = true; abort.abort() }, 120_000)
    controller.current = abort
    activeTurnRef.current = turnId
    setActiveTurnId(turnId)
    setConversationHistory(current => [...current, { id: turnId, question: prompt, status: 'loading' }])
    setQuestion(prompt)
    setAnswerOpen(true)
    window.history.replaceState(null, '', `/advisor?q=${encodeURIComponent(prompt)}`)
    setLoading(true); setElapsedSeconds(0); setError(null); setResult(null)
    try {
      const refreshed = await status.refetch()
      if (abort.signal.aborted) return
      const captured = refreshed.data?.integrity?.snapshot
      if (refreshed.isError || !captured || refreshed.data?.data?.loaded !== true || refreshed.data.integrity?.status !== 'verified') {
        throw new Error('PeopleOS could not verify the current workforce snapshot. Refresh the connection and retry.')
      }
      const answer = await api.intelligence.investigate({ question: prompt, dataset_version: captured.dataset_id, agentic: true }, abort.signal) as AgentAnswer
      if (!abort.signal.aborted) {
        setResult({ answer, generation: captured.generation, reportingCurrency: captured.reporting_currency, source: `${captured.source_name ?? 'Current workforce'}${captured.dataset_version == null ? '' : ` · version ${captured.dataset_version}`}` })
        setConversationHistory(current => current.map(turn => turn.id === turnId ? { ...turn, status: 'complete', answer } : turn))
      }
    } catch (err) {
      if (timedOut) {
        const message = 'This investigation took longer than PeopleOS could keep the request open. Nothing was changed; retry when the local analysis service is ready.'
        setError(message)
        setConversationHistory(current => current.map(turn => turn.id === turnId ? { ...turn, status: 'error', error: message } : turn))
      } else if (!abort.signal.aborted) {
        const message = err instanceof Error ? err.message : 'PeopleOS could not answer this question.'
        setError(message)
        setConversationHistory(current => current.map(turn => turn.id === turnId ? { ...turn, status: 'error', error: message } : turn))
      }
    } finally {
      window.clearTimeout(timeout)
      if (controller.current === abort) { controller.current = null; activeTurnRef.current = null; setActiveTurnId(null); setLoading(false); setElapsedSeconds(0) }
    }
  }, [loading, question, ready, snapshot, status])

  useEffect(() => {
    if (!requestedQuestion || !ready || question !== requestedQuestion || initialQueryRun.current === requestedQuestion) return
    const timer = window.setTimeout(() => {
      initialQueryRun.current = requestedQuestion
      void investigate(requestedQuestion)
    }, 0)
    return () => window.clearTimeout(timer)
  }, [investigate, question, ready, requestedQuestion])

  function submit(event: FormEvent<HTMLFormElement>) { event.preventDefault(); void investigate() }
  function cancel() {
    const activeTurnId = activeTurnRef.current
    controller.current?.abort()
    controller.current = null
    activeTurnRef.current = null
    setActiveTurnId(null)
    if (activeTurnId) setConversationHistory(current => current.map(turn => turn.id === activeTurnId ? { ...turn, status: 'cancelled' } : turn))
    setLoading(false)
    setElapsedSeconds(0)
    setError('Stopped waiting for this answer.')
  }

  return <Page>
    <PageHeader eyebrow="Ask PeopleOS" title="Ask your workforce a question" description="Have a conversation with your verified workforce snapshot. PeopleOS chooses the relevant checks, explains the result in People language and keeps the supporting evidence one step away." actions={currentResult && !answerOpen ? <Button type="button" variant="secondary" onClick={() => setAnswerOpen(true)}>Open answer panel <ArrowRight className="h-4 w-4" /></Button> : undefined} />

    {status.isLoading ? <StateSummary title="Connecting to your workforce" description="PeopleOS is checking the active data source." tone="info" /> : status.isError ? <div role="alert"><StateSummary title="Your workforce data could not be checked" description="Retry before asking a question." tone="warning" /><Button variant="secondary" onClick={() => void status.refetch()}>Retry</Button></div> : !ready ? <StateSummary title="Add workforce data first" description="Use the fictional sample or add your own file, then come back and ask anything your data can support." tone="info" /> : <div className="flex flex-wrap items-center gap-2 text-sm text-text-secondary"><span className="h-2 w-2 rounded-full bg-emerald-500" />Using <strong className="text-text-primary">{snapshot?.source_name ?? 'your current workforce'}</strong>{snapshot?.dataset_version != null ? ` · version ${snapshot.dataset_version}` : ''}<span className="text-text-muted">·</span><span>Conversation mode</span></div>}
    {!status.isLoading && !status.isError && !ready && <Link href="/upload" className="inline-flex items-center gap-2 text-sm font-semibold text-accent">Add workforce data <ArrowRight className="h-4 w-4" /></Link>}

    {ready ? <div className="mx-auto flex w-full max-w-4xl flex-col gap-6">
      <Conversation history={conversationHistory} activeTurnId={activeTurnId} onOpenAnswer={() => setAnswerOpen(true)} onCancel={cancel} />
      <QuestionComposer question={question} onQuestionChange={setQuestion} onSubmit={submit} onInvestigate={prompt => void investigate(prompt)} onCancel={cancel} loading={loading} elapsedSeconds={elapsedSeconds} ready={Boolean(ready)} />
      {!conversationHistory.length && !loading && <p className="text-center text-xs leading-5 text-text-muted">Ask a question to begin. AI can organise and explain verified results, but it does not create the underlying facts.</p>}
    </div> : <div className="mx-auto w-full max-w-4xl"><QuestionComposer question={question} onQuestionChange={setQuestion} onSubmit={submit} onInvestigate={prompt => void investigate(prompt)} onCancel={cancel} loading={loading} elapsedSeconds={elapsedSeconds} ready={false} /></div>}

    {result && !currentResult && !loading && <StateSummary title="Your workforce data changed" description="The previous answer has been hidden. Ask again to use the current data." tone="info" />}
    {error && !answerOpen && <div role="alert"><StateSummary title="No answer shown" description={error} tone="warning" /></div>}
    {answerOpen && <AnswerDrawer result={currentResult} loading={loading} elapsedSeconds={elapsedSeconds} error={error} followUps={followUps} onClose={() => setAnswerOpen(false)} onCancel={cancel} onInvestigate={prompt => void investigate(prompt)} />}
  </Page>
}

export default function PeopleIntelligencePage() {
  return <PeopleIntelligencePageContent />
}
