'use client'

import { CheckCircle2, ChevronDown, ShieldCheck } from 'lucide-react'
import { StateSummary, StatusBadge, Surface } from '@/components/ui'

export interface EvidenceItem { evidence_id: string; kind: string; claim: string; source_tool: string; value?: unknown; metric?: string | null; confidence: number; metadata?: Record<string, unknown> }
export interface ToolResult { result_id: string; tool_id: string; status: 'success' | 'partial' | 'blocked' | 'failed'; summary: string; evidence: EvidenceItem[]; warnings: string[] }
export interface AgentAnswer { request_id: string; question: string; answer: string; status: string; confidence: number; tools_used: string[]; model?: string | null; evidence: { coverage_score?: number | null; sufficiency?: string; unknowns: string[]; contradictions?: string[]; verification_notes: string[]; tool_results: ToolResult[] }; warnings: string[] }

export function percent(value?: number | null) { return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1 ? `${Math.round(value * 100)}%` : 'Unavailable' }
export function userFacingWarning(warning: string) { return /numpy|dtype|traceback|attributeerror|typeerror|valueerror|exception/i.test(warning) ? 'One analytical capability could not contribute evidence. Review the available evidence and gaps before using this answer.' : warning }

export function evidenceClaim(item: EvidenceItem) {
  const departmentMetrics: Record<string, string> = {
    department_turnover_rate: 'Observed attrition share',
    department_observed_attrition_share: 'Observed attrition share',
    salary_dispersion_consistency_score: 'Salary-dispersion consistency score',
    salary_dispersion_cv: 'Salary coefficient of variation',
    salary_dispersion_gini: 'Salary Gini coefficient',
    pay_equity_score: 'Pay-equity score',
  }
  const metric = item.metric ?? ''
  if (Object.prototype.hasOwnProperty.call(departmentMetrics, metric)) {
    const quoted = JSON.stringify(String(item.metadata?.department ?? 'Unknown')).replace(/[\u007f-\u009f\u2028-\u202e\u2066-\u2069]/g, char => `\\u${char.charCodeAt(0).toString(16).padStart(4, '0')}`)
    const value = item.value === null || item.value === undefined || item.value === '' || typeof item.value === 'boolean' ? NaN : Number(item.value)
    const measurement = Number.isFinite(value) ? (metric.startsWith('department_') ? `${(value * 100).toFixed(1)}%` : value.toFixed(2)) : 'Unavailable'
    return `${departmentMetrics[metric]}: ${measurement} (source department label: ${quoted})`
  }
  if (item.value === null || item.value === undefined || item.value === '' || typeof item.value === 'boolean') return item.claim
  const value = typeof item.value === 'number' ? item.value : Number(item.value)
  if (!Number.isFinite(value)) return item.claim
  const label = item.claim.split(':', 1)[0]
  if (['observed_attrition_share', 'department_observed_attrition_share', 'mean_risk_score'].includes(item.metric ?? '')) return `${label}: ${(value * 100).toFixed(1)}%`
  if (['model_f1', 'model_roc_auc'].includes(item.metric ?? '')) return `${label}: ${value.toFixed(3)}`
  if (item.metric === 'salary_mean') return `${label}: ${Math.round(value).toLocaleString()}`
  if (item.metric === 'tenure_mean') return `${label}: ${value.toFixed(1)} years`
  if (item.metric === 'lastrating_mean') return `${label}: ${value.toFixed(1)}/5`
  if (['headcount', 'record_count', 'active_count', 'department_count'].includes(item.metric ?? '')) return `${label}: ${Math.round(value).toLocaleString()}`
  return item.claim
}

export function InvestigationResult({ result, source }: { result: AgentAnswer; source: string }) {
  const tools = result.evidence.tool_results
  const items = tools.flatMap(tool => tool.evidence)
  const unknowns = result.evidence.unknowns
  const contradictions = result.evidence.contradictions ?? []
  const limitations = Array.from(new Set([...result.warnings, ...tools.flatMap(tool => tool.warnings)]))
  const complete = result.status === 'complete'
  const partial = result.status === 'partial'

  return <div className="space-y-4">
    <Surface padding="lg" className="overflow-hidden">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div><p className="text-xs font-bold uppercase tracking-[0.16em] text-violet-600 dark:text-violet-300">PeopleOS answer</p><h2 className="mt-2 max-w-3xl text-xl font-semibold tracking-tight text-slate-950 dark:text-white">{result.question}</h2></div>
        <StatusBadge tone={complete ? 'success' : partial ? 'warning' : 'neutral'}>{complete ? 'Supported by your data' : partial ? 'Some evidence missing' : 'Not enough evidence'}</StatusBadge>
      </div>

      <div className="mt-6 whitespace-pre-wrap break-words text-[15px] leading-7 text-slate-800 dark:text-slate-200">{result.answer}</div>

      <div className="mt-6 flex flex-wrap items-center gap-x-4 gap-y-2 border-t border-slate-200/70 pt-4 text-xs text-slate-500 dark:border-white/10 dark:text-slate-400">
        <span className="inline-flex items-center gap-1.5"><CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" />Source: {source}</span>
        <span>{result.model ? 'AI organised verified evidence' : 'Deterministic answer'}</span>
        <span>{items.length} supporting evidence item{items.length === 1 ? '' : 's'}</span>
      </div>
    </Surface>

    {(unknowns.length > 0 || contradictions.length > 0) && <StateSummary title={contradictions.length ? 'Some evidence conflicts' : 'There are limits to this answer'} description={(contradictions[0] ?? unknowns[0]) ? userFacingWarning(contradictions[0] ?? unknowns[0]) : 'Review the available evidence before acting.'} tone="warning" />}

    <details className="group rounded-2xl border border-slate-200 bg-white dark:border-white/10 dark:bg-slate-900">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-4 px-5 py-4 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500">
        <div className="flex items-center gap-3"><ShieldCheck className="h-5 w-5 text-emerald-600 dark:text-emerald-400" /><div><div className="font-semibold text-slate-900 dark:text-white">Why you can trust this answer</div><div className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">Population, evidence coverage, missing data and calculation sources</div></div></div>
        <ChevronDown className="h-4 w-4 text-slate-400 transition group-open:rotate-180" />
      </summary>
      <div className="border-t border-slate-200/70 p-5 dark:border-white/10">
        <dl className="grid gap-4 sm:grid-cols-4">
          <TrustMetric label="Evidence quality" value={percent(result.confidence)} />
          <TrustMetric label="Coverage" value={percent(result.evidence.coverage_score)} />
          <TrustMetric label="Tools with evidence" value={`${tools.filter(tool => tool.evidence.length > 0 && ['success', 'partial'].includes(tool.status)).length}/${tools.length}`} />
          <TrustMetric label="Known gaps" value={`${unknowns.length + contradictions.length}`} />
        </dl>
        {limitations.length > 0 && <div className="mt-5"><div className="text-sm font-semibold">Important limitations</div><ul className="mt-2 space-y-1.5 text-sm leading-6 text-slate-600 dark:text-slate-400">{limitations.slice(0, 8).map((item, index) => <li key={index}>• {userFacingWarning(item)}</li>)}</ul></div>}
      </div>
    </details>

    <details className="group rounded-2xl border border-slate-200 bg-white dark:border-white/10 dark:bg-slate-900">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-4 px-5 py-4 font-semibold focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500">Evidence ledger ({items.length} items)<ChevronDown className="h-4 w-4 text-slate-400 transition group-open:rotate-180" /></summary>
      <div className="space-y-3 border-t border-slate-200/70 p-5 dark:border-white/10">{items.map(item => <div key={item.evidence_id} className="rounded-xl border border-border p-4"><p className="text-sm font-medium">{evidenceClaim(item)}</p><EvidencePopulation item={item} /><p className="mt-2 break-words text-xs text-text-muted">Evidence: {item.evidence_id} · Source: {item.source_tool} · {item.kind}</p></div>)}{items.length === 0 && <p className="text-sm text-text-secondary">No supporting evidence was available.</p>}</div>
    </details>

    <details className="group rounded-2xl border border-slate-200 bg-white dark:border-white/10 dark:bg-slate-900">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-4 px-5 py-4 text-sm font-semibold focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500">Technical details<ChevronDown className="h-4 w-4 text-slate-400 transition group-open:rotate-180" /></summary>
      <div className="border-t border-slate-200/70 p-5 text-sm dark:border-white/10"><div className="space-y-3">{tools.map(tool => <div key={tool.result_id} className="flex items-start justify-between gap-4 border-b border-border pb-3 last:border-0"><div><div className="font-medium">{tool.tool_id.replaceAll('_', ' ').replaceAll('.', ' · ')}</div><div className="mt-1 text-xs text-text-muted">{userFacingWarning(tool.summary)} · {tool.evidence.length} evidence items</div></div><StatusBadge tone={tool.status === 'success' ? 'success' : tool.status === 'failed' ? 'danger' : 'warning'}>{tool.status}</StatusBadge></div>)}</div><p className="mt-4 break-all text-xs text-text-muted">Request: {result.request_id}</p></div>
    </details>
  </div>
}

function TrustMetric({ label, value }: { label: string; value: string }) { return <div className="rounded-xl bg-slate-50 p-3 dark:bg-white/[0.03]"><dt className="text-[11px] font-medium text-slate-500 dark:text-slate-400">{label}</dt><dd className="mt-1 text-lg font-semibold text-slate-900 dark:text-white">{value}</dd></div> }

function EvidencePopulation({ item }: { item: EvidenceItem }) {
  const labels: Record<string, string> = { measured_count: 'Measured', eligible_count: 'Eligible', excluded_count: 'Excluded or missing', sample_size: 'Sample size', population: 'Population' }
  const fields = Object.entries(labels).flatMap(([key, label]) => {
    const value = item.metadata?.[key]
    if (typeof value === 'number' && Number.isFinite(value)) return [`${label}: ${value.toLocaleString()}`]
    if (key === 'population' && typeof value === 'string' && value) return [`${label}: ${value.replaceAll('_', ' ')}`]
    return []
  })
  return fields.length > 0 ? <p className="mt-2 text-xs text-text-secondary">{fields.join(' · ')}</p> : null
}
