'use client'

import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import Link from 'next/link'
import { ArrowRight, Search, Sparkles } from 'lucide-react'
import { Button, Page, PageHeader, StateSummary, Surface, Textarea } from '@/components/ui'
import { api } from '@/lib/api-client'
import { AgentAnswer, InvestigationResult, userFacingWarning } from '@/components/advisor/investigation-result'

interface RuntimeStatus { data?: { loaded?: boolean }; integrity?: { status: string; snapshot?: { generation: string; dataset_id: string; dataset_version?: number; source_name?: string } } }
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

export default function PeopleIntelligencePage() {
  const status = useQuery<RuntimeStatus>({ queryKey: ['platform', 'status'], queryFn: () => api.getStatus() as Promise<RuntimeStatus>, refetchOnMount: 'always' })
  const snapshot = status.data?.integrity?.snapshot
  const ready = !status.isError && status.data?.data?.loaded && status.data.integrity?.status === 'verified' && Boolean(snapshot)
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState<{ answer: AgentAnswer; generation: string; source: string } | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const controller = useRef<AbortController | null>(null)
  const initialQueryLoaded = useRef(false)
  const initialQueryRun = useRef(false)
  useEffect(() => () => controller.current?.abort(), [])
  useEffect(() => {
    if (initialQueryLoaded.current || typeof window === 'undefined') return
    initialQueryLoaded.current = true
    const prompt = new URLSearchParams(window.location.search).get('q')?.trim()
    if (!prompt) return
    const timer = window.setTimeout(() => setQuestion(prompt.slice(0, 2000)), 0)
    return () => window.clearTimeout(timer)
  }, [])
  const currentResult = ready && result?.generation === snapshot?.generation ? result : null
  const followUps = useMemo(() => currentResult ? deeperQuestions(currentResult.answer.question) : [], [currentResult])

  const investigate = useCallback(async (selected?: string) => {
    const prompt = (selected ?? question).trim()
    if (prompt.length < 3 || prompt.length > 2000 || !ready || !snapshot || loading) return
    const abort = new AbortController()
    controller.current = abort
    setQuestion(prompt)
    window.history.replaceState(null, '', `/advisor?q=${encodeURIComponent(prompt)}`)
    setLoading(true); setError(null); setResult(null)
    try {
      // A local backend restart can create a fresh runtime generation while
      // React Query still holds the previous status response. Refresh just
      // before the investigation so a valid answer is not mistaken for stale
      // data; the server-side snapshot guard still rejects a real mid-request
      // dataset/model change.
      const refreshed = await status.refetch()
      const captured = refreshed.data?.integrity?.snapshot
      if (refreshed.isError || !captured || refreshed.data?.data?.loaded !== true || refreshed.data.integrity?.status !== 'verified') {
        throw new Error('PeopleOS could not verify the current workforce snapshot. Refresh the connection and retry.')
      }
      const answer = await api.intelligence.investigate({ question: prompt, dataset_version: captured.dataset_id }, abort.signal) as AgentAnswer
      if (!abort.signal.aborted) setResult({ answer, generation: captured.generation, source: `${captured.source_name ?? 'Current workforce'}${captured.dataset_version == null ? '' : ` · version ${captured.dataset_version}`}` })
    } catch (err) { if (!abort.signal.aborted) setError(userFacingWarning(err instanceof Error ? err.message : 'PeopleOS could not answer this question.')) }
    finally { if (controller.current === abort) { controller.current = null; setLoading(false) } }
  }, [loading, question, ready, snapshot, status])

  useEffect(() => {
    if (initialQueryRun.current || !ready || typeof window === 'undefined') return
    const prompt = new URLSearchParams(window.location.search).get('q')?.trim()?.slice(0, 2000)
    if (!prompt || question !== prompt) return
    initialQueryRun.current = true
    window.setTimeout(() => void investigate(prompt), 0)
  }, [investigate, ready, question])

  function submit(event: FormEvent) { event.preventDefault(); void investigate() }
  function cancel() { controller.current?.abort(); controller.current = null; setLoading(false); setError('Stopped waiting for this answer.') }

  return <Page>
    <PageHeader eyebrow="Ask PeopleOS" title="What would you like to understand?" description="Ask in normal People language. PeopleOS will choose the available analysis, calculate from your workforce data, and tell you when the evidence is not enough." />

    {status.isLoading ? <StateSummary title="Connecting to your workforce" description="PeopleOS is checking the active data source." tone="info" /> : status.isError ? <div role="alert"><StateSummary title="Your workforce data could not be checked" description="Retry before asking a question." tone="warning" /><Button variant="secondary" onClick={() => void status.refetch()}>Retry</Button></div> : !ready ? <StateSummary title="Add workforce data first" description="Use the fictional sample or add your own file, then come back and ask anything your data can support." tone="info" /> : <div className="flex items-center gap-2 text-sm text-text-secondary"><span className="h-2 w-2 rounded-full bg-emerald-500" />Using <strong className="text-text-primary">{snapshot?.source_name ?? 'your current workforce'}</strong>{snapshot?.dataset_version != null ? ` · version ${snapshot.dataset_version}` : ''}</div>}
    {!status.isLoading && !status.isError && !ready && <Link href="/upload" className="inline-flex items-center gap-2 text-sm font-semibold text-accent">Add workforce data <ArrowRight className="h-4 w-4" /></Link>}

    <Surface padding="lg" className="border-violet-200/70 bg-gradient-to-br from-white to-violet-50/30 dark:border-violet-500/20 dark:from-slate-950 dark:to-violet-500/[0.03]">
      <form onSubmit={submit} className="space-y-4" aria-busy={loading}>
        <Textarea label="Ask PeopleOS" value={question} onChange={event => setQuestion(event.target.value)} placeholder="For example: What should I be paying attention to in this workforce?" rows={4} minLength={3} maxLength={2000} disabled={loading || !ready} helperText="Ask about workforce, pay, retention, experience, structure, fairness or hiring. If the data cannot answer reliably, PeopleOS will say so." leading={<Search className="h-4 w-4" />} />
        <div className="flex flex-wrap items-center justify-between gap-3"><div className="flex flex-wrap gap-2">{suggestedQuestions.map(([label, prompt]) => <Button key={label} type="button" variant="secondary" size="sm" disabled={loading || !ready} title={`Run: ${prompt}`} onClick={() => void investigate(prompt)}>{label}</Button>)}</div><Button type="submit" isLoading={loading} disabled={!ready || question.trim().length < 3}><Sparkles className="h-4 w-4" />Ask PeopleOS</Button></div>
      </form>
    </Surface>

    {loading && <Surface padding="md"><div role="status" aria-live="polite"><p className="font-semibold">Looking through your workforce data…</p><p className="mt-2 text-sm text-text-secondary">PeopleOS is running the relevant calculations and checking whether there is enough evidence to answer.</p></div><Button className="mt-3" variant="secondary" size="sm" onClick={cancel}>Stop</Button></Surface>}
    {error && <div role="alert"><StateSummary title="No answer shown" description={error} tone="warning" /></div>}
    {result && !currentResult && !loading && <StateSummary title="Your workforce data changed" description="The previous answer has been hidden. Ask again to use the current data." tone="info" />}
    {currentResult && !loading && <>
      <InvestigationResult result={currentResult.answer} source={currentResult.source} />
      <Surface padding="md" className="border-violet-100/80 bg-violet-50/30 dark:border-violet-500/15 dark:bg-violet-500/[0.03]">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div><div className="text-sm font-semibold text-text-primary">Explore deeper</div><div className="mt-1 text-xs text-text-secondary">Run another verified calculation from the same workforce data.</div></div>
          <div className="flex flex-wrap gap-2">{followUps.map(([label, prompt]) => <Button key={`${label}-${prompt}`} type="button" variant="secondary" size="sm" disabled={loading} onClick={() => void investigate(prompt)}>{label}</Button>)}</div>
        </div>
      </Surface>
    </>}

    {ready && !currentResult && !loading && <div className="text-center text-xs text-text-muted">PeopleOS calculates from your active workforce data. AI can organise and explain verified results, but it does not create the underlying facts.</div>}
  </Page>
}
