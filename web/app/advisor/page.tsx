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
  ['Workforce health', 'What are the most important workforce health signals right now?'],
  ['Recorded attrition', 'Where is observed attrition share elevated and which departments need investigation?'],
  ['Compensation', 'What does our compensation data say about salary dispersion and pay-gap screening?'],
  ['Organisation structure', 'Where do we have aggregate span-of-control or role-tenure pressure?'],
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
      if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(typeof body.detail === 'string' ? body.detail : 'The investigation could not complete. Check the question and try again.') }
      if (response.headers.get('X-PeopleOS-Snapshot') !== captured.generation) throw new Error('The dataset changed during the investigation. Run your question again against the current source.')
      const answer: AgentAnswer = await response.json()
      if (!abort.signal.aborted) setResult({ answer, generation: captured.generation, source: `${captured.source_name ?? 'Current dataset'}${captured.dataset_version == null ? '' : ` · version ${captured.dataset_version}`}` })
    } catch (err) { if (!abort.signal.aborted) setError(userFacingWarning(err instanceof Error ? err.message : 'Unable to investigate this question.')) }
    finally { if (controller.current === abort) { controller.current = null; setLoading(false) } }
  }
  function submit(event: FormEvent) { event.preventDefault(); void investigate() }
  function cancel() { controller.current?.abort(); controller.current = null; setLoading(false); setError('Stopped waiting for this investigation. The server may finish its read-only analysis; no answer from this request will be displayed.') }
  return <Page>
    <PageHeader eyebrow="Investigate · People Intelligence" title="Ask a workforce question and inspect the evidence" description="Explore workforce patterns with an aggregate, read-only investigation. Each answer shows its evidence, tool outcomes and remaining gaps." />
    {status.isLoading ? <div role="status"><StateSummary title="Checking your data source" description="Confirming which workforce dataset is available for investigation." tone="info" /></div> : status.isError ? <div role="alert"><StateSummary title="Data source unavailable" description="PeopleOS could not check the current dataset. Retry before investigating." tone="warning" /><Button variant="secondary" onClick={() => void status.refetch()}>Retry connection</Button></div> : !ready ? <StateSummary title="Add a verified workforce dataset first" description="Upload data or load the sample to start an investigation." tone="info" /> : <p className="text-sm text-text-secondary">Investigating: <strong>{snapshot?.source_name ?? 'Current workforce dataset'}</strong>{snapshot?.dataset_version != null ? ` · version ${snapshot.dataset_version}` : ''}</p>}
    {!status.isLoading && !status.isError && !ready && <Link href="/upload" className="inline-flex items-center gap-2 text-sm font-semibold text-accent">Add workforce data <ArrowRight className="h-4 w-4" /></Link>}
    <Surface padding="lg"><form onSubmit={submit} className="space-y-4" aria-busy={loading}><Textarea label="Investigation question" value={question} onChange={event => setQuestion(event.target.value)} placeholder="Ask about observed attrition, workforce health, compensation or organisation structure…" rows={3} minLength={3} maxLength={2000} disabled={loading || !ready} helperText="3–2,000 characters. Each submission starts a new investigation of the selected dataset." leading={<Search className="h-4 w-4" />} /><div className="flex flex-wrap items-center justify-between gap-3"><div className="flex flex-wrap gap-2">{suggestedQuestions.map(([label, prompt]) => <Button key={label} type="button" variant="secondary" size="sm" disabled={loading || !ready} title={prompt} onClick={() => setQuestion(prompt)}>{label}</Button>)}</div><Button type="submit" isLoading={loading} disabled={!ready || question.trim().length < 3}><Sparkles className="h-4 w-4" />Investigate</Button></div></form></Surface>
    {loading && <Surface padding="md"><div role="status" aria-live="polite"><p className="font-semibold">Investigation in progress</p><p className="mt-2 text-sm text-text-secondary">Waiting for tool evidence and synthesis. Execution outcomes will appear when the server responds.</p></div><Button className="mt-3" variant="secondary" size="sm" onClick={cancel}>Stop waiting</Button></Surface>}
    {error && <div role="alert"><StateSummary title="Investigation not displayed" description={error} tone="warning" /></div>}
    {result && !currentResult && !loading && <StateSummary title="The data source has changed" description="The previous answer has been hidden. Run your question again to use the current dataset." tone="info" />}
    {currentResult && !loading && <InvestigationResult result={currentResult.answer} source={currentResult.source} />}
    <Link href="/platform" className="inline-flex items-center gap-2 text-sm font-semibold text-accent">Review data integrity and capability status <ArrowRight className="h-4 w-4" /></Link>
  </Page>
}
