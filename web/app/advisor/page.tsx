'use client'

import { FormEvent, useEffect, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import Link from 'next/link'
import { ArrowRight, Search, Sparkles } from 'lucide-react'
import { Button, Page, PageHeader, StateSummary, Surface, Textarea } from '@/components/ui'
import { api } from '@/lib/api-client'
import { AgentAnswer, InvestigationResult, userFacingWarning } from '@/components/advisor/investigation-result'

interface RuntimeStatus { data?: { loaded?: boolean }; integrity?: { status: string; snapshot?: { generation: string; dataset_id: string; dataset_version?: number; source_name?: string } } }
const suggestedQuestions = [
  ['What should I look at first?', 'What are the most important workforce signals I should look at right now?'],
  ['Where are people leaving?', 'Where is recorded attrition elevated across departments?'],
  ['How does pay look?', 'What does our compensation data show about salary distribution and pay-gap screening?'],
  ['How are teams structured?', 'Where do we have aggregate span-of-control or role-tenure pressure?'],
]

export default function PeopleIntelligencePage() {
  const status = useQuery<RuntimeStatus>({ queryKey: ['platform', 'status'], queryFn: () => api.getStatus() as Promise<RuntimeStatus> })
  const snapshot = status.data?.integrity?.snapshot
  const ready = !status.isError && status.data?.data?.loaded && status.data.integrity?.status === 'verified' && Boolean(snapshot)
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState<{ answer: AgentAnswer; generation: string; source: string } | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const controller = useRef<AbortController | null>(null)
  useEffect(() => () => controller.current?.abort(), [])
  const currentResult = ready && result?.generation === snapshot?.generation ? result : null

  async function investigate(selected?: string) {
    const prompt = (selected ?? question).trim()
    if (prompt.length < 3 || prompt.length > 2000 || !ready || !snapshot || loading) return
    const captured = snapshot
    const abort = new AbortController()
    controller.current = abort
    setQuestion(prompt); setLoading(true); setError(null); setResult(null)
    try {
      const response = await fetch('/api/intelligence/investigate', { method: 'POST', signal: abort.signal, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: prompt, dataset_version: captured.dataset_id }) })
      if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(typeof body.detail === 'string' ? body.detail : 'PeopleOS could not answer that question. Try asking it another way.') }
      if (response.headers.get('X-PeopleOS-Snapshot') !== captured.generation) throw new Error('Your workforce data changed while PeopleOS was analysing it. Ask the question again to use the latest data.')
      const answer: AgentAnswer = await response.json()
      if (!abort.signal.aborted) setResult({ answer, generation: captured.generation, source: `${captured.source_name ?? 'Current workforce'}${captured.dataset_version == null ? '' : ` · version ${captured.dataset_version}`}` })
    } catch (err) { if (!abort.signal.aborted) setError(userFacingWarning(err instanceof Error ? err.message : 'PeopleOS could not answer this question.')) }
    finally { if (controller.current === abort) { controller.current = null; setLoading(false) } }
  }

  function submit(event: FormEvent) { event.preventDefault(); void investigate() }
  function cancel() { controller.current?.abort(); controller.current = null; setLoading(false); setError('Stopped waiting for this answer.') }

  return <Page>
    <PageHeader eyebrow="Ask PeopleOS" title="What would you like to understand?" description="Ask in normal People language. PeopleOS will choose the available analysis, calculate from your workforce data, and tell you when the evidence is not enough." />

    {status.isLoading ? <StateSummary title="Connecting to your workforce" description="PeopleOS is checking the active data source." tone="info" /> : status.isError ? <div role="alert"><StateSummary title="Your workforce data could not be checked" description="Retry before asking a question." tone="warning" /><Button variant="secondary" onClick={() => void status.refetch()}>Retry</Button></div> : !ready ? <StateSummary title="Add workforce data first" description="Use the fictional sample or add your own file, then come back and ask anything your data can support." tone="info" /> : <div className="flex items-center gap-2 text-sm text-text-secondary"><span className="h-2 w-2 rounded-full bg-emerald-500" />Using <strong className="text-text-primary">{snapshot?.source_name ?? 'your current workforce'}</strong>{snapshot?.dataset_version != null ? ` · version ${snapshot.dataset_version}` : ''}</div>}
    {!status.isLoading && !status.isError && !ready && <Link href="/upload" className="inline-flex items-center gap-2 text-sm font-semibold text-accent">Add workforce data <ArrowRight className="h-4 w-4" /></Link>}

    <Surface padding="lg" className="border-violet-200/70 bg-gradient-to-br from-white to-violet-50/30 dark:border-violet-500/20 dark:from-slate-950 dark:to-violet-500/[0.03]">
      <form onSubmit={submit} className="space-y-4" aria-busy={loading}>
        <Textarea label="Ask PeopleOS" value={question} onChange={event => setQuestion(event.target.value)} placeholder="For example: What should I be paying attention to in this workforce?" rows={4} minLength={3} maxLength={2000} disabled={loading || !ready} helperText="Ask about workforce, pay, retention, experience, structure, fairness or hiring. If the data cannot answer reliably, PeopleOS will say so." leading={<Search className="h-4 w-4" />} />
        <div className="flex flex-wrap items-center justify-between gap-3"><div className="flex flex-wrap gap-2">{suggestedQuestions.map(([label, prompt]) => <Button key={label} type="button" variant="secondary" size="sm" disabled={loading || !ready} title={prompt} onClick={() => setQuestion(prompt)}>{label}</Button>)}</div><Button type="submit" isLoading={loading} disabled={!ready || question.trim().length < 3}><Sparkles className="h-4 w-4" />Ask PeopleOS</Button></div>
      </form>
    </Surface>

    {loading && <Surface padding="md"><div role="status" aria-live="polite"><p className="font-semibold">Looking through your workforce data…</p><p className="mt-2 text-sm text-text-secondary">PeopleOS is running the relevant calculations and checking whether there is enough evidence to answer.</p></div><Button className="mt-3" variant="secondary" size="sm" onClick={cancel}>Stop</Button></Surface>}
    {error && <div role="alert"><StateSummary title="No answer shown" description={error} tone="warning" /></div>}
    {result && !currentResult && !loading && <StateSummary title="Your workforce data changed" description="The previous answer has been hidden. Ask again to use the current data." tone="info" />}
    {currentResult && !loading && <InvestigationResult result={currentResult.answer} source={currentResult.source} />}

    {ready && !currentResult && !loading && <div className="text-center text-xs text-text-muted">PeopleOS calculates from your active workforce data. AI can organise and explain verified results, but it does not create the underlying facts.</div>}
  </Page>
}
