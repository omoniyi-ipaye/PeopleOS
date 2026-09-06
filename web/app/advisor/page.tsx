'use client'

import { FormEvent, useMemo, useState } from 'react'
import Link from 'next/link'
import { AlertTriangle, ArrowRight, Brain, Database, Search, ShieldCheck, Sparkles, Target, Wrench } from 'lucide-react'
import { Button, Page, PageHeader, SectionHeader, StateSummary, StatusBadge, Surface, Textarea } from '@/components/ui'
import { EvidenceQuality } from '@/components/patterns'

interface EvidenceItem { evidence_id: string; kind: string; claim: string; source_tool: string; value?: unknown; metric?: string | null; confidence: number; metadata?: Record<string, unknown> }
interface ToolResult { result_id: string; tool_id: string; status: 'success' | 'partial' | 'blocked' | 'failed'; summary: string; evidence: EvidenceItem[]; warnings: string[]; error?: string | null }
interface EvidenceBundle { overall_confidence?: number | null; coverage_score?: number; sufficiency?: string; unknowns: string[]; contradictions?: string[]; verification_notes: string[]; tool_results: ToolResult[] }
interface AgentAnswer { request_id: string; question: string; answer: string; status: 'complete' | 'partial' | 'unavailable' | 'insufficient'; confidence: number; tools_used: string[]; model?: string | null; evidence: EvidenceBundle; warnings: string[] }

const suggestedQuestions = [
  'What are the most important workforce health signals right now?',
  'Why is turnover high and which departments need investigation?',
  'What does our compensation data say about pay equity?',
  'Where do we have manager span or role-stagnation risk?',
]

function pct(value?: number | null) { return value == null ? '—' : `${Math.round(value * 100)}%` }

function userFacingWarning(warning: string) {
  if (/numpy|dtype|traceback|attributeerror|typeerror|valueerror|exception/i.test(warning)) return 'One analytical capability could not contribute evidence. Verified evidence from other tools remains valid.'
  return warning
}

function evidenceClaim(item: EvidenceItem) {
  const value = typeof item.value === 'number' ? item.value : Number(item.value)
  if (!Number.isFinite(value)) return item.claim
  const label = item.claim.split(':', 1)[0]
  if (['turnover_rate', 'department_turnover_rate', 'mean_risk_score', 'model_f1'].includes(item.metric ?? '')) return `${label}: ${(value * 100).toFixed(1)}%`
  if (item.metric === 'salary_mean') return `${label}: ${Math.round(value).toLocaleString()}`
  if (item.metric === 'tenure_mean') return `${label}: ${value.toFixed(1)} years`
  if (item.metric === 'lastrating_mean') return `${label}: ${value.toFixed(1)}/5`
  if (['headcount', 'active_count', 'department_count'].includes(item.metric ?? '')) return `${label}: ${Math.round(value).toLocaleString()}`
  return item.claim
}

export default function PeopleIntelligencePage() {
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState<AgentAnswer | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showEvidence, setShowEvidence] = useState(true)
  const [showTrace, setShowTrace] = useState(false)

  const evidenceItems = useMemo(() => result?.evidence.tool_results.flatMap((tool) => tool.evidence) ?? [], [result])

  async function investigate(selected?: string) {
    const prompt = (selected ?? question).trim()
    if (!prompt) return
    setQuestion(prompt); setLoading(true); setError(null)
    try {
      const response = await fetch('/api/intelligence/investigate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: prompt }) })
      if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail ?? 'People Intelligence investigation failed.') }
      setResult(await response.json())
    } catch (err) { setResult(null); setError(err instanceof Error ? err.message : 'Unable to investigate this question.') }
    finally { setLoading(false) }
  }

  function submit(event: FormEvent) { event.preventDefault(); void investigate() }

  return (
    <Page>
      <PageHeader eyebrow="Investigate · People Intelligence" title="Ask a workforce question and inspect the evidence" description="PeopleOS plans deterministic tools, measures evidence quality and exposes gaps before synthesis. The agent is analytical and read-only." />

      <Surface padding="lg">
        <form onSubmit={submit} className="space-y-4">
          <Textarea label="Investigation question" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask about turnover, workforce health, compensation equity or organisation structure…" rows={3} leading={<Search className="h-4 w-4" />} />
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap gap-2">{suggestedQuestions.map((item) => <Button key={item} type="button" variant="secondary" size="sm" disabled={loading} onClick={() => void investigate(item)}>{item.length > 42 ? `${item.slice(0, 42)}…` : item}</Button>)}</div>
            <Button type="submit" isLoading={loading} disabled={!question.trim()}><Sparkles className="h-4 w-4" />Investigate</Button>
          </div>
        </form>
      </Surface>

      {error && <StateSummary title="Investigation unavailable" description={userFacingWarning(error)} tone="danger" />}

      {result && <div className="grid gap-6 xl:grid-cols-[minmax(0,1.5fr)_minmax(320px,0.7fr)]">
        <main className="space-y-6">
          <Surface padding="lg">
            <div className="flex flex-wrap items-start justify-between gap-4"><div><div className="text-xs font-semibold uppercase tracking-wider text-text-muted">Finding</div><h2 className="mt-1 text-xl font-semibold">{result.question}</h2></div><div className="flex gap-2"><StatusBadge tone={result.status === 'complete' ? 'success' : 'warning'}>{result.status === 'complete' ? 'Complete coverage' : result.status === 'partial' ? 'Partial coverage' : 'Insufficient coverage'}</StatusBadge><StatusBadge tone="info">{pct(result.confidence)} confidence</StatusBadge></div></div>
            <div className="mt-5 whitespace-pre-wrap rounded-2xl bg-background-secondary p-5 text-sm leading-7 text-text-primary">{result.answer}</div>
            <div className="mt-5 flex flex-wrap gap-2">{result.tools_used.map((tool) => <StatusBadge key={tool} tone="neutral"><Wrench className="h-3 w-3" />{tool}</StatusBadge>)}<StatusBadge tone="neutral"><Brain className="h-3 w-3" />{result.model ? `Synthesized by ${result.model}` : 'Deterministic synthesis'}</StatusBadge></div>
          </Surface>

          <Surface padding="none">
            <button type="button" onClick={() => setShowEvidence((value) => !value)} className="flex w-full items-center justify-between p-5 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/40"><div className="flex items-center gap-3"><Database className="h-5 w-5 text-accent" /><div><div className="font-semibold">Evidence ledger</div><div className="text-xs text-text-muted">{evidenceItems.length} items from {result.evidence.tool_results.length} governed tool runs</div></div></div><span className="text-xs font-semibold text-accent">{showEvidence ? 'Hide' : 'Show'}</span></button>
            {showEvidence && <div className="space-y-3 border-t border-border p-5">{evidenceItems.length ? evidenceItems.map((item) => <div key={item.evidence_id} className="rounded-2xl border border-border bg-background-secondary p-4"><div className="flex items-start justify-between gap-4"><div><div className="text-sm font-medium">{evidenceClaim(item)}</div><div className="mt-2 text-[11px] text-text-muted">{item.source_tool}{item.metric ? ` · ${item.metric}` : ''} · {item.kind}</div></div><StatusBadge tone="neutral">{pct(item.confidence)}</StatusBadge></div></div>) : <div className="text-sm text-text-muted">No positive evidence was available.</div>}</div>}
          </Surface>

          <Surface padding="lg"><SectionHeader title="Continue the investigation" description="Use follow-up questions to reduce uncertainty rather than jumping directly to action." /><div className="mt-4 flex flex-wrap gap-2">{['What evidence is missing before I act?', 'Which departments should I compare first?', 'What should I validate with the People owner?'].map((item) => <Button key={item} variant="secondary" size="sm" onClick={() => void investigate(item)} disabled={loading}>{item}<ArrowRight className="h-3.5 w-3.5" /></Button>)}</div></Surface>
        </main>

        <aside className="space-y-4">
          <EvidenceQuality confidence={result.confidence} coverage={result.evidence.coverage_score ?? (result.status === 'complete' ? 1 : result.status === 'partial' ? .6 : .2)} sufficiency={(result.evidence.sufficiency as any) ?? (result.status === 'complete' ? 'SUFFICIENT' : result.status === 'partial' ? 'LIMITED' : 'INSUFFICIENT')} />
          {result.warnings.length > 0 && <StateSummary title="Limitations & controls" description={result.warnings.map(userFacingWarning).slice(0, 3).join(' · ')} tone="warning" />}
          <StateSummary title="Agent boundary" description="Read-only aggregate analysis · allowlisted tools · consequential employment actions remain outside autonomy." tone="success" />
          <Surface padding="md"><button type="button" onClick={() => setShowTrace((value) => !value)} className="flex w-full items-center justify-between text-left"><div><div className="font-semibold">Advanced trace</div><div className="text-xs text-text-muted">Request and tool execution detail</div></div><span className="text-xs font-semibold text-accent">{showTrace ? 'Hide' : 'Show'}</span></button>{showTrace && <div className="mt-4 rounded-xl bg-background-secondary p-3 font-mono text-[11px] leading-5 text-text-muted">Request: {result.request_id}<br />Tools: {result.tools_used.join(', ') || 'none'}<br />Synthesis: {result.model ?? 'deterministic fallback'}</div>}</Surface>
          <Link href="/platform" className="inline-flex items-center gap-1 text-xs font-semibold text-accent"><ShieldCheck className="h-3.5 w-3.5" />Review Trust Center <ArrowRight className="h-3.5 w-3.5" /></Link>
        </aside>
      </div>}
    </Page>
  )
}
