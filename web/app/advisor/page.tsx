'use client'

import { FormEvent, useMemo, useState } from 'react'
import Link from 'next/link'
import {
  AlertTriangle,
  ArrowRight,
  Brain,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Database,
  Loader2,
  Search,
  ShieldCheck,
  Sparkles,
  Target,
  Wrench,
} from 'lucide-react'

interface EvidenceItem {
  evidence_id: string
  kind: string
  claim: string
  source_tool: string
  metric?: string | null
  confidence: number
}

interface ToolResult {
  result_id: string
  tool_id: string
  status: 'success' | 'partial' | 'blocked' | 'failed'
  summary: string
  evidence: EvidenceItem[]
  warnings: string[]
  error?: string | null
}

interface EvidenceBundle {
  overall_confidence?: number | null
  unknowns: string[]
  verification_notes: string[]
  tool_results: ToolResult[]
}

interface AgentAnswer {
  request_id: string
  question: string
  answer: string
  status: 'complete' | 'partial' | 'unavailable' | 'insufficient'
  confidence: number
  tools_used: string[]
  model?: string | null
  evidence: EvidenceBundle
  warnings: string[]
}

const suggestedQuestions = [
  'What are the most important workforce health signals right now?',
  'Why is turnover high and which departments need investigation?',
  'What does our compensation data say about pay equity?',
  'Where do we have manager span or role-stagnation risk?',
]

const followUps = [
  'Which part of this finding should I investigate next?',
  'What evidence is missing before I act on this?',
  'Which departments should I compare first?',
]

function pct(value: number) {
  return `${Math.round(value * 100)}%`
}

function coverageLabel(status: AgentAnswer['status']) {
  if (status === 'complete') return 'Complete coverage'
  if (status === 'partial') return 'Partial coverage'
  return 'Insufficient coverage'
}

function userFacingWarning(warning: string) {
  if (/numpy|dtype|traceback|attributeerror|typeerror|valueerror|exception/i.test(warning)) {
    const tool = warning.split(':')[0]?.trim()
    return tool
      ? `${tool} could not contribute evidence. The verified evidence from other tools is still valid.`
      : 'One analytical capability could not contribute evidence. The verified evidence from other tools is still valid.'
  }
  return warning
}

export default function PeopleIntelligencePage() {
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState<AgentAnswer | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showEvidence, setShowEvidence] = useState(true)
  const [showTrace, setShowTrace] = useState(false)

  const evidenceItems = useMemo(
    () => result?.evidence.tool_results.flatMap((tool) => tool.evidence) ?? [],
    [result]
  )

  async function investigate(selected?: string) {
    const prompt = (selected ?? question).trim()
    if (!prompt) return
    setQuestion(prompt)
    setLoading(true)
    setError(null)
    try {
      const response = await fetch('/api/intelligence/investigate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: prompt }),
      })
      if (!response.ok) {
        const body = await response.json().catch(() => ({}))
        throw new Error(body.detail ?? 'People Intelligence investigation failed.')
      }
      setResult(await response.json())
    } catch (err) {
      setResult(null)
      setError(err instanceof Error ? err.message : 'Unable to investigate this question.')
    } finally {
      setLoading(false)
    }
  }

  function onSubmit(event: FormEvent) {
    event.preventDefault()
    void investigate()
  }

  return (
    <div className="mx-auto max-w-7xl space-y-6 pb-10">
      <section className="relative overflow-hidden rounded-[30px] bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 p-6 text-white shadow-xl md:p-8">
        <div className="absolute -right-24 -top-24 h-72 w-72 rounded-full bg-violet-500/10 blur-3xl" />
        <div className="relative flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
          <div className="max-w-3xl">
            <div className="mb-3 flex items-center gap-2 text-xs font-bold uppercase tracking-[0.18em] text-violet-300"><ShieldCheck className="h-4 w-4" /> Investigate · Governed evidence</div>
            <h1 className="text-3xl font-semibold tracking-tight">People Intelligence Agent</h1>
            <p className="mt-3 text-sm leading-6 text-slate-300">Ask a business question. PeopleOS selects deterministic tools, shows what evidence supports the finding, separates coverage from confidence, and makes gaps visible before you act.</p>
          </div>
          <div className="grid grid-cols-3 gap-2 text-center text-[11px] text-slate-300">
            <Boundary icon={Database} label="Aggregate evidence" />
            <Boundary icon={Wrench} label="Allowlisted tools" />
            <Boundary icon={ShieldCheck} label="Policy gated" />
          </div>
        </div>
      </section>

      <section className="rounded-[28px] border border-slate-200 bg-white p-5 shadow-sm dark:border-white/10 dark:bg-slate-900">
        <form onSubmit={onSubmit} className="space-y-4">
          <div className="relative">
            <Search className="absolute left-4 top-4 h-5 w-5 text-slate-400" />
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask about turnover, workforce health, compensation equity, manager structure…"
              rows={3}
              className="w-full resize-none rounded-2xl border border-slate-200 bg-slate-50 py-3.5 pl-12 pr-4 text-sm text-slate-900 outline-none transition focus:border-violet-400 focus:bg-white focus:ring-4 focus:ring-violet-400/10 dark:border-white/10 dark:bg-white/[0.04] dark:text-white dark:focus:bg-white/[0.06]"
            />
          </div>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap gap-2">
              {suggestedQuestions.map((item) => (
                <button key={item} type="button" onClick={() => void investigate(item)} disabled={loading} className="rounded-full border border-slate-200 px-3 py-1.5 text-xs text-slate-600 transition hover:border-violet-300 hover:text-violet-700 disabled:opacity-50 dark:border-white/10 dark:text-slate-400 dark:hover:border-violet-500/30 dark:hover:text-violet-300">
                  {item.length > 44 ? `${item.slice(0, 44)}…` : item}
                </button>
              ))}
            </div>
            <button type="submit" disabled={loading || !question.trim()} className="inline-flex items-center gap-2 rounded-xl bg-violet-600 px-5 py-2.5 text-sm font-semibold text-white shadow-lg shadow-violet-600/20 transition hover:bg-violet-500 disabled:cursor-not-allowed disabled:opacity-50">
              {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />} Investigate
            </button>
          </div>
        </form>
      </section>

      {error && (
        <section className="flex items-start gap-3 rounded-2xl border border-red-200 bg-red-50 p-4 text-sm text-red-800 dark:border-red-500/20 dark:bg-red-500/[0.05] dark:text-red-300">
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" />
          <div><div className="font-semibold">Investigation unavailable</div><div className="mt-1">{userFacingWarning(error)}</div></div>
        </section>
      )}

      {result && (
        <div className="grid gap-6 xl:grid-cols-[minmax(0,1.5fr)_minmax(330px,0.7fr)]">
          <div className="space-y-6">
            <section className="rounded-[28px] border border-slate-200 bg-white p-6 shadow-sm dark:border-white/10 dark:bg-slate-900">
              <div className="mb-5 flex flex-wrap items-start justify-between gap-4">
                <div><div className="text-xs font-bold uppercase tracking-[0.16em] text-slate-400">Finding</div><h2 className="mt-1 text-xl font-semibold text-slate-950 dark:text-white">{result.question}</h2></div>
                <div className="flex flex-wrap gap-2">
                  <span className={`rounded-full px-3 py-1 text-xs font-semibold ${result.status === 'complete' ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-300' : 'bg-amber-100 text-amber-700 dark:bg-amber-500/10 dark:text-amber-300'}`}>{coverageLabel(result.status)}</span>
                  <span className="rounded-full bg-violet-100 px-3 py-1 text-xs font-semibold text-violet-700 dark:bg-violet-500/10 dark:text-violet-300">{pct(result.confidence)} confidence</span>
                </div>
              </div>
              <div className="whitespace-pre-wrap rounded-2xl bg-slate-50 p-5 text-sm leading-7 text-slate-800 dark:bg-white/[0.04] dark:text-slate-200">{result.answer}</div>
              <div className="mt-5 flex flex-wrap gap-2">
                {result.tools_used.map((tool) => <span key={tool} className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 px-2.5 py-1 text-xs text-slate-500 dark:border-white/10 dark:text-slate-400"><Wrench className="h-3 w-3" />{tool}</span>)}
                <span className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 px-2.5 py-1 text-xs text-slate-500 dark:border-white/10 dark:text-slate-400"><Brain className="h-3 w-3" />{result.model ? `Synthesized by ${result.model}` : 'Deterministic synthesis'}</span>
              </div>
            </section>

            <section className="rounded-[28px] border border-slate-200 bg-white shadow-sm dark:border-white/10 dark:bg-slate-900">
              <button type="button" onClick={() => setShowEvidence((value) => !value)} className="flex w-full items-center justify-between p-5 text-left">
                <div className="flex items-center gap-3"><div className="grid h-10 w-10 place-items-center rounded-xl bg-sky-100 text-sky-700 dark:bg-sky-500/10 dark:text-sky-300"><Database className="h-5 w-5" /></div><div><h3 className="font-semibold text-slate-950 dark:text-white">Evidence ledger</h3><p className="mt-1 text-xs text-slate-500 dark:text-slate-400">{evidenceItems.length} traceable evidence item(s) from {result.evidence.tool_results.length} tool run(s)</p></div></div>
                {showEvidence ? <ChevronUp className="h-5 w-5 text-slate-400" /> : <ChevronDown className="h-5 w-5 text-slate-400" />}
              </button>
              {showEvidence && (
                <div className="space-y-3 border-t border-slate-200 p-5 dark:border-white/10">
                  {evidenceItems.length ? evidenceItems.map((item) => (
                    <div key={item.evidence_id} className="rounded-2xl border border-slate-200 bg-slate-50 p-4 dark:border-white/10 dark:bg-white/[0.03]">
                      <div className="flex items-start justify-between gap-4"><div><div className="text-sm font-medium text-slate-900 dark:text-white">{item.claim}</div><div className="mt-2 flex flex-wrap gap-2 text-[11px] text-slate-500 dark:text-slate-400"><span>{item.source_tool}</span><span>·</span><span>{item.kind}</span>{item.metric && <><span>·</span><span>{item.metric}</span></>}</div></div><span className="shrink-0 rounded-full bg-white px-2 py-1 text-[11px] font-medium text-slate-600 dark:bg-white/5 dark:text-slate-300">{pct(item.confidence)}</span></div>
                    </div>
                  )) : <div className="rounded-2xl border border-dashed border-slate-200 p-6 text-center text-sm text-slate-500 dark:border-white/10 dark:text-slate-400">No positive evidence was available for this investigation.</div>}
                </div>
              )}
            </section>

            <section className="rounded-[28px] border border-slate-200 bg-white p-5 dark:border-white/10 dark:bg-slate-900">
              <div className="text-xs font-bold uppercase tracking-[0.16em] text-slate-400">Continue the investigation</div>
              <div className="mt-3 flex flex-wrap gap-2">{followUps.map((item) => <button key={item} type="button" onClick={() => void investigate(item)} disabled={loading} className="inline-flex items-center gap-2 rounded-xl border border-slate-200 px-3 py-2 text-xs font-semibold text-slate-700 transition hover:border-violet-300 hover:text-violet-700 disabled:opacity-50 dark:border-white/10 dark:text-slate-300 dark:hover:border-violet-500/30 dark:hover:text-violet-300">{item}<ArrowRight className="h-3.5 w-3.5" /></button>)}</div>
            </section>
          </div>

          <aside className="space-y-4">
            <section className="rounded-[28px] border border-slate-200 bg-white p-5 shadow-sm dark:border-white/10 dark:bg-slate-900">
              <div className="mb-4 flex items-center gap-2"><Target className="h-5 w-5 text-violet-500" /><h3 className="font-semibold text-slate-950 dark:text-white">Investigation health</h3></div>
              <div><div className="mb-1 flex justify-between text-xs"><span className="text-slate-500 dark:text-slate-400">Confidence in available evidence</span><span className="font-semibold">{pct(result.confidence)}</span></div><div className="h-2 overflow-hidden rounded-full bg-slate-100 dark:bg-white/10"><div className="h-full rounded-full bg-violet-500" style={{ width: pct(result.confidence) }} /></div></div>
              <div className="mt-4 grid grid-cols-2 gap-3"><HealthMetric value={result.evidence.verification_notes.length} label="verified runs" /><HealthMetric value={result.evidence.unknowns.length} label="known gaps" /></div>
            </section>

            {result.warnings.length > 0 && <section className="rounded-[28px] border border-amber-200 bg-amber-50 p-5 dark:border-amber-500/20 dark:bg-amber-500/[0.05]"><div className="mb-3 flex items-center gap-2 text-amber-800 dark:text-amber-300"><AlertTriangle className="h-5 w-5" /><h3 className="font-semibold">Limitations & controls</h3></div><ul className="space-y-2 text-xs leading-5 text-amber-800/90 dark:text-amber-300/90">{result.warnings.map((warning, index) => <li key={`${warning}-${index}`}>• {userFacingWarning(warning)}</li>)}</ul></section>}

            <section className="rounded-[28px] border border-emerald-200 bg-emerald-50 p-5 dark:border-emerald-500/20 dark:bg-emerald-500/[0.05]"><div className="mb-3 flex items-center gap-2 text-emerald-800 dark:text-emerald-300"><CheckCircle2 className="h-5 w-5" /><h3 className="font-semibold">Agent boundary</h3></div><ul className="space-y-2 text-xs leading-5 text-emerald-800/90 dark:text-emerald-300/90"><li>• Read-only analytical execution</li><li>• Aggregate evidence only</li><li>• Allowlisted tools only</li><li>• Consequential employment actions remain outside autonomy</li></ul></section>

            <section className="rounded-[28px] border border-slate-200 bg-white p-5 dark:border-white/10 dark:bg-slate-900">
              <button type="button" onClick={() => setShowTrace((value) => !value)} className="flex w-full items-center justify-between text-left"><div><div className="font-semibold text-slate-950 dark:text-white">Advanced trace</div><div className="mt-1 text-xs text-slate-500 dark:text-slate-400">Request and tool execution detail</div></div><span className="text-xs font-semibold text-violet-600 dark:text-violet-300">{showTrace ? 'Hide' : 'Show'}</span></button>
              {showTrace && <div className="mt-4 rounded-xl bg-slate-50 p-3 text-[11px] leading-5 text-slate-500 dark:bg-white/[0.04] dark:text-slate-400">Request: {result.request_id}<br />Tools: {result.tools_used.join(', ') || 'none'}<br />Synthesis: {result.model ?? 'deterministic fallback'}</div>}
            </section>

            <Link href="/platform" className="inline-flex items-center gap-1 text-xs font-semibold text-violet-600 dark:text-violet-300">Review Trust Center <ArrowRight className="h-3.5 w-3.5" /></Link>
          </aside>
        </div>
      )}
    </div>
  )
}

function Boundary({ icon: Icon, label }: { icon: React.ElementType; label: string }) {
  return <div className="rounded-2xl border border-white/10 bg-white/5 px-3 py-3"><Icon className="mx-auto mb-1.5 h-4 w-4 text-violet-300" />{label}</div>
}

function HealthMetric({ value, label }: { value: number; label: string }) {
  return <div className="rounded-2xl bg-slate-50 p-3 dark:bg-white/[0.04]"><div className="text-2xl font-semibold text-slate-950 dark:text-white">{value}</div><div className="text-xs text-slate-500 dark:text-slate-400">{label}</div></div>
}
