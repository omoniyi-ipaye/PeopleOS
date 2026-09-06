'use client'

import { FormEvent, useMemo, useState } from 'react'
import {
  AlertTriangle,
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
  value?: unknown
  metric?: string | null
  confidence: number
  metadata?: Record<string, unknown>
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
  bundle_id: string
  overall_confidence?: number | null
  contradictions: string[]
  unknowns: string[]
  verification_notes: string[]
  tool_results: ToolResult[]
}

interface AgentAnswer {
  request_id: string
  question: string
  answer: string
  status: 'complete' | 'partial' | 'unavailable'
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

function pct(value: number) {
  return `${Math.round(value * 100)}%`
}

function userFacingWarning(warning: string) {
  const technicalPatterns = [
    /numpy/i,
    /dtype/i,
    /traceback/i,
    /attributeerror/i,
    /typeerror/i,
    /valueerror/i,
    /exception/i,
  ]
  if (technicalPatterns.some((pattern) => pattern.test(warning))) {
    const tool = warning.split(':')[0]?.trim()
    return tool
      ? `${tool} could not contribute evidence to this investigation. Other verified evidence is still shown.`
      : 'One analytical capability could not contribute evidence to this investigation. Other verified evidence is still shown.'
  }
  return warning
}

function coverageLabel(status: AgentAnswer['status']) {
  if (status === 'complete') return 'Complete coverage'
  if (status === 'partial') return 'Partial coverage'
  return 'Insufficient coverage'
}

export default function AdvisorPage() {
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState<AgentAnswer | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showEvidence, setShowEvidence] = useState(true)

  const evidenceItems = useMemo(
    () => result?.evidence.tool_results.flatMap((tool) => tool.evidence) ?? [],
    [result]
  )

  async function investigate(selectedQuestion?: string) {
    const prompt = (selectedQuestion ?? question).trim()
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
    <div className="min-h-screen bg-background p-6 md:p-8">
      <div className="mx-auto max-w-7xl space-y-6">
        <section className="relative overflow-hidden rounded-3xl border border-white/10 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 p-7 text-white shadow-2xl">
          <div className="absolute -right-20 -top-20 h-64 w-64 rounded-full bg-violet-500/10 blur-3xl" />
          <div className="relative grid gap-6 lg:grid-cols-[1fr_auto] lg:items-end">
            <div>
              <div className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.2em] text-violet-300">
                <ShieldCheck className="h-4 w-4" />
                Governed · Read-only · Evidence-backed
              </div>
              <div className="flex items-center gap-3">
                <div className="rounded-2xl bg-violet-500/15 p-3 ring-1 ring-violet-400/20">
                  <Brain className="h-7 w-7 text-violet-300" />
                </div>
                <div>
                  <h1 className="text-3xl font-semibold tracking-tight">People Intelligence Agent</h1>
                  <p className="mt-1 max-w-3xl text-sm text-slate-300">
                    Ask a workforce question. PeopleOS selects governed analytics tools, builds a traceable evidence bundle, checks confidence and limitations, then synthesizes the result.
                  </p>
                </div>
              </div>
            </div>
            <div className="grid grid-cols-3 gap-2 text-center text-xs">
              <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3"><Database className="mx-auto mb-1 h-4 w-4 text-sky-300" />Aggregate data</div>
              <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3"><Wrench className="mx-auto mb-1 h-4 w-4 text-emerald-300" />Allowlisted tools</div>
              <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3"><ShieldCheck className="mx-auto mb-1 h-4 w-4 text-violet-300" />Policy gated</div>
            </div>
          </div>
        </section>

        <section className="rounded-3xl border border-border bg-card p-5 shadow-sm">
          <form onSubmit={onSubmit} className="space-y-4">
            <div className="relative">
              <Search className="absolute left-4 top-4 h-5 w-5 text-muted-foreground" />
              <textarea value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask about turnover, workforce health, compensation equity, manager structure…" rows={3} className="w-full resize-none rounded-2xl border border-border bg-background py-3.5 pl-12 pr-4 text-sm outline-none transition focus:border-violet-400 focus:ring-4 focus:ring-violet-400/10" />
            </div>
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex flex-wrap gap-2">
                {suggestedQuestions.map((item) => (
                  <button key={item} type="button" onClick={() => void investigate(item)} disabled={loading} className="rounded-full border border-border bg-muted/40 px-3 py-1.5 text-xs text-muted-foreground transition hover:border-violet-300 hover:text-foreground disabled:opacity-50">
                    {item.length > 42 ? `${item.slice(0, 42)}…` : item}
                  </button>
                ))}
              </div>
              <button type="submit" disabled={loading || !question.trim()} className="inline-flex items-center gap-2 rounded-xl bg-violet-600 px-5 py-2.5 text-sm font-semibold text-white shadow-lg shadow-violet-600/20 transition hover:bg-violet-500 disabled:cursor-not-allowed disabled:opacity-50">
                {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
                Investigate
              </button>
            </div>
          </form>
        </section>

        {error && (
          <div className="flex items-start gap-3 rounded-2xl border border-red-200 bg-red-50 p-4 text-sm text-red-800 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-200">
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" />
            <div><div className="font-semibold">Investigation unavailable</div><div className="mt-1">{userFacingWarning(error)}</div></div>
          </div>
        )}

        {result && (
          <div className="grid gap-6 xl:grid-cols-[minmax(0,1.5fr)_minmax(340px,0.7fr)]">
            <div className="space-y-6">
              <section className="rounded-3xl border border-border bg-card p-6 shadow-sm">
                <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
                  <div><div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Investigation</div><h2 className="mt-1 text-xl font-semibold">{result.question}</h2></div>
                  <div className="flex items-center gap-2">
                    <span className={`rounded-full px-3 py-1 text-xs font-semibold ${result.status === 'complete' ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-300' : 'bg-amber-100 text-amber-700 dark:bg-amber-950/50 dark:text-amber-300'}`}>{coverageLabel(result.status)}</span>
                    <span className="rounded-full bg-violet-100 px-3 py-1 text-xs font-semibold text-violet-700 dark:bg-violet-950/50 dark:text-violet-300">{pct(result.confidence)} confidence in available evidence</span>
                  </div>
                </div>

                <div className="whitespace-pre-wrap rounded-2xl bg-muted/30 p-5 text-sm leading-7 text-foreground">{result.answer}</div>

                <div className="mt-5 flex flex-wrap gap-2">
                  {result.tools_used.map((tool) => <span key={tool} className="inline-flex items-center gap-1.5 rounded-lg border border-border px-2.5 py-1 text-xs text-muted-foreground"><Wrench className="h-3 w-3" />{tool}</span>)}
                  <span className="inline-flex items-center gap-1.5 rounded-lg border border-border px-2.5 py-1 text-xs text-muted-foreground"><Brain className="h-3 w-3" />{result.model ? `Synthesized by ${result.model}` : 'Deterministic synthesis'}</span>
                </div>
              </section>

              <section className="rounded-3xl border border-border bg-card shadow-sm">
                <button type="button" onClick={() => setShowEvidence((value) => !value)} className="flex w-full items-center justify-between p-5 text-left">
                  <div className="flex items-center gap-3"><div className="rounded-xl bg-sky-100 p-2 text-sky-700 dark:bg-sky-950/40 dark:text-sky-300"><Database className="h-5 w-5" /></div><div><h3 className="font-semibold">Evidence ledger</h3><p className="text-xs text-muted-foreground">{evidenceItems.length} traceable evidence item(s) from {result.evidence.tool_results.length} tool run(s)</p></div></div>
                  {showEvidence ? <ChevronUp className="h-5 w-5 text-muted-foreground" /> : <ChevronDown className="h-5 w-5 text-muted-foreground" />}
                </button>
                {showEvidence && <div className="border-t border-border p-5"><div className="space-y-3">{evidenceItems.length ? evidenceItems.map((item) => <div key={item.evidence_id} className="rounded-2xl border border-border bg-background p-4"><div className="flex items-start justify-between gap-4"><div><div className="text-sm font-medium">{item.claim}</div><div className="mt-2 flex flex-wrap gap-2 text-[11px] text-muted-foreground"><span>{item.source_tool}</span><span>·</span><span>{item.kind}</span>{item.metric && <><span>·</span><span>{item.metric}</span></>}</div></div><span className="shrink-0 rounded-full bg-muted px-2 py-1 text-[11px] font-medium">{pct(item.confidence)}</span></div></div>) : <div className="rounded-2xl border border-dashed border-border p-6 text-center text-sm text-muted-foreground">No positive evidence was available for this investigation.</div>}</div></div>}
              </section>
            </div>

            <aside className="space-y-4">
              <section className="rounded-3xl border border-border bg-card p-5 shadow-sm">
                <div className="mb-4 flex items-center gap-2"><Target className="h-5 w-5 text-violet-500" /><h3 className="font-semibold">Investigation health</h3></div>
                <div className="space-y-4">
                  <div><div className="mb-1 flex justify-between text-xs"><span className="text-muted-foreground">Confidence in available evidence</span><span className="font-semibold">{pct(result.confidence)}</span></div><div className="h-2 overflow-hidden rounded-full bg-muted"><div className="h-full rounded-full bg-violet-500" style={{ width: pct(result.confidence) }} /></div></div></div>
                  <div className="grid grid-cols-2 gap-3"><div className="rounded-2xl bg-muted/40 p-3"><div className="text-2xl font-semibold">{result.evidence.verification_notes.length}</div><div className="text-xs text-muted-foreground">verified tool runs</div></div><div className="rounded-2xl bg-muted/40 p-3"><div className="text-2xl font-semibold">{result.evidence.unknowns.length}</div><div className="text-xs text-muted-foreground">known gaps</div></div></div>
                </div>
              </section>

              {result.warnings.length > 0 && <section className="rounded-3xl border border-amber-200 bg-amber-50 p-5 dark:border-amber-900/50 dark:bg-amber-950/20"><div className="mb-3 flex items-center gap-2 text-amber-800 dark:text-amber-200"><AlertTriangle className="h-5 w-5" /><h3 className="font-semibold">Limitations & controls</h3></div><ul className="space-y-2 text-xs leading-5 text-amber-800/90 dark:text-amber-200/90">{result.warnings.map((warning, index) => <li key={`${warning}-${index}`}>• {userFacingWarning(warning)}</li>)}</ul></section>}

              <section className="rounded-3xl border border-emerald-200 bg-emerald-50 p-5 dark:border-emerald-900/50 dark:bg-emerald-950/20"><div className="mb-3 flex items-center gap-2 text-emerald-800 dark:text-emerald-200"><CheckCircle2 className="h-5 w-5" /><h3 className="font-semibold">Agent boundary</h3></div><ul className="space-y-2 text-xs leading-5 text-emerald-800/90 dark:text-emerald-200/90"><li>• Read-only analytical execution</li><li>• Aggregate evidence only</li><li>• Allowlisted tools only</li><li>• Employment-action policy enforced after synthesis</li><li>• LLM optional; deterministic fallback available</li></ul></section>
            </aside>
          </div>
        )}
      </div>
    </div>
  )
}
