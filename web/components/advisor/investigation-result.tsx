'use client'

import { SectionHeader, StateSummary, StatusBadge, Surface } from '@/components/ui'

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
    pay_equity_score: 'Pay-equity score',
  }
  const metric = item.metric ?? ''
  if (Object.prototype.hasOwnProperty.call(departmentMetrics, metric)) {
    // Quote source data separately from the typed measurement, retaining raw metadata.
    const quoted = JSON.stringify(String(item.metadata?.department ?? 'Unknown'))
      .replace(/[\u007f-\u009f\u2028-\u202e\u2066-\u2069]/g, char => `\\u${char.charCodeAt(0).toString(16).padStart(4, '0')}`)
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
  return <div className="space-y-5">
    <Surface padding="lg"><div className="flex flex-wrap items-start justify-between gap-3"><div><p className="text-xs font-semibold uppercase tracking-wider text-text-muted">Investigation result</p><h2 className="mt-2 text-xl font-semibold">{result.question}</h2></div><StatusBadge tone={result.status === 'complete' ? 'info' : 'warning'}>{result.status === 'complete' ? 'Investigation completed' : result.status === 'partial' ? 'Partial evidence' : 'Insufficient evidence'}</StatusBadge></div><p className="mt-3 text-xs text-text-muted">Source: {source}</p><div className="mt-5 whitespace-pre-wrap break-words rounded-2xl bg-background-secondary p-5 text-sm leading-7 text-text-primary">{result.answer}</div><p className="mt-3 text-xs text-text-secondary">{result.model ? `Evidence prioritisation: ${result.model}` : 'Deterministic synthesis'} · Review the evidence before making a decision.</p></Surface>
    <div className="grid gap-5 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
      <Surface padding="lg"><SectionHeader title="Evidence and gaps" description="Scores describe support for this investigation; they are not a probability that the answer is true." /><dl className="mt-4 grid grid-cols-2 gap-4"><div><dt className="text-xs text-text-secondary">Heuristic evidence quality</dt><dd className="mt-1 text-xl font-semibold">{percent(result.confidence)}</dd></div><div><dt className="text-xs text-text-secondary">Tool evidence coverage</dt><dd className="mt-1 text-xl font-semibold">{percent(result.evidence.coverage_score)}</dd></div><div><dt className="text-xs text-text-secondary">Successful runs with evidence</dt><dd className="mt-1 font-semibold">{tools.filter(tool => tool.status === 'success' && tool.evidence.length > 0).length} of {tools.length}</dd></div><div><dt className="text-xs text-text-secondary">Known gaps and conflicts</dt><dd className="mt-1 font-semibold">{unknowns.length + contradictions.length}</dd></div></dl>
      {unknowns.length > 0 && <div className="mt-5"><h3 className="font-semibold">Missing evidence</h3><ul className="mt-2 list-disc space-y-2 pl-5 text-sm text-text-secondary">{unknowns.map((item, index) => <li key={index}>{userFacingWarning(item)}</li>)}</ul></div>}
      {contradictions.length > 0 && <div className="mt-5"><h3 className="font-semibold text-warning">Conflicting evidence</h3><ul className="mt-2 list-disc space-y-2 pl-5 text-sm">{contradictions.map((item, index) => <li key={index}>{userFacingWarning(item)}</li>)}</ul></div>}
      {limitations.length > 0 && <details className="mt-5"><summary className="cursor-pointer font-semibold">Limitations and controls ({limitations.length})</summary><ul className="mt-3 list-disc space-y-2 pl-5 text-sm text-text-secondary">{limitations.map((item, index) => <li key={index}>{userFacingWarning(item)}</li>)}</ul></details>}</Surface>
      <Surface padding="lg"><SectionHeader title="Tool execution" description="Actual outcomes returned by the investigation." /><ul className="mt-4 divide-y divide-border">{tools.map(tool => <li key={tool.result_id} className="py-3"><div className="flex flex-wrap items-center justify-between gap-2"><span className="break-words text-sm font-semibold">{tool.tool_id.replaceAll('_', ' ').replaceAll('.', ' · ')}</span><StatusBadge tone={tool.status === 'success' ? 'success' : tool.status === 'failed' ? 'danger' : 'warning'}>{tool.status}</StatusBadge></div><p className="mt-2 text-sm text-text-secondary">{userFacingWarning(tool.summary)}</p><p className="mt-1 text-xs text-text-muted">{tool.evidence.length} evidence items</p></li>)}</ul>{tools.length === 0 && <p className="mt-4 text-sm text-text-secondary">No analytical tools returned results.</p>}</Surface>
    </div>
    <Surface padding="lg"><details><summary className="cursor-pointer font-semibold">Evidence ledger ({items.length} items)</summary><div className="mt-4 space-y-3">{items.map(item => <div key={item.evidence_id} className="rounded-xl border border-border p-4"><p className="text-sm font-medium">{evidenceClaim(item)}</p><EvidencePopulation item={item} /><p className="mt-2 break-words text-xs text-text-muted">Evidence: {item.evidence_id} · Source: {item.source_tool} · {item.kind} · heuristic weight {percent(item.confidence)}</p></div>)}{items.length === 0 && <p className="text-sm text-text-secondary">No supporting evidence was available.</p>}</div></details></Surface>
    <Surface padding="md"><details><summary className="cursor-pointer text-sm font-semibold">Verification and request details</summary><p className="mt-3 break-all text-xs text-text-muted">Request: {result.request_id}</p><ul className="mt-3 list-disc space-y-2 pl-5 text-sm text-text-secondary">{result.evidence.verification_notes.map((note, index) => <li key={index}>{userFacingWarning(note)}</li>)}</ul></details></Surface>
    <StateSummary title="Review before action" description="PeopleOS supports aggregate investigation. It does not make employment decisions or change employee records." tone="info" />
  </div>
}

function EvidencePopulation({ item }: { item: EvidenceItem }) {
  const labels: Record<string, string> = { measured_count: 'Measured', eligible_count: 'Eligible population', excluded_count: 'Excluded or missing', sample_size: 'Sample size', population: 'Population' }
  const fields = Object.entries(labels).flatMap(([key, label]) => {
    const value = item.metadata?.[key]
    if (typeof value === 'number' && Number.isFinite(value)) return [`${label}: ${value.toLocaleString()}`]
    if (key === 'population' && typeof value === 'string' && value) return [`${label}: ${value.replaceAll('_', ' ')}`]
    return []
  })
  return fields.length > 0 ? <p className="mt-2 text-xs text-text-secondary">{fields.join(' · ')}</p> : null
}
