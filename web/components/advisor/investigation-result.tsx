'use client'

import { BarChart3, Bot, CheckCircle2, ChevronDown, Circle, CircleAlert, ShieldCheck } from 'lucide-react'
import { GroupSummaryChart } from '@/components/charts/group-summary-chart'
import { StateSummary, StatusBadge, Surface } from '@/components/ui'

export interface EvidenceItem { evidence_id: string; kind: string; claim: string; source_tool: string; value?: unknown; metric?: string | null; confidence: number; metadata?: Record<string, unknown> }
export interface ToolResult { result_id: string; tool_id: string; status: 'success' | 'partial' | 'blocked' | 'failed'; summary: string; evidence: EvidenceItem[]; warnings: string[] }
export type AgentStepStatus = 'complete' | 'attention' | 'skipped'
export interface AgentStep { id: string; label: string; status: AgentStepStatus; detail: string; tools?: string[] }
export interface AgentNextAction { label: string; question: string; reason: string }
export interface AgentAnswer { request_id: string; question: string; answer: string; status: string; confidence: number; tools_used: string[]; model?: string | null; synthesis_mode?: 'verified_evidence' | 'grounded_llm'; cited_evidence_ids?: string[]; evidence: { coverage_score?: number | null; sufficiency?: string; unknowns: string[]; contradictions?: string[]; verification_notes: string[]; tool_results: ToolResult[] }; warnings: string[]; agent_steps?: AgentStep[]; next_actions?: AgentNextAction[] }

interface AnalysisSpec { operation?: string; population?: string; group_by?: string | null; second_group_by?: string | null; measure?: string | null; second_measure?: string | null; statistic?: string | null }
interface DerivedAnalysis { item: EvidenceItem; output: Record<string, unknown>; spec: AnalysisSpec }
interface GroupSummaryRow { group: string; value: number; measuredCount: number | null; eligibleCount: number | null; excludedCount: number | null }
interface GroupSummaryModel { groupBy: string; measure?: string | null; statistic: string; population: string; rows: GroupSummaryRow[]; populationCount: number | null; suppressedGroups: number }

const COLUMN_LABELS: Record<string, string> = { Dept: 'department', Location: 'location', Gender: 'gender', JobTitle: 'role', JobLevel: 'job level' }
const MEASURE_LABELS: Record<string, string> = { Salary: 'salary', Tenure: 'tenure', Age: 'age', LastRating: 'performance rating', Attrition: 'recorded departure share' }

function asRecord(value: unknown): Record<string, unknown> | null { return typeof value === 'object' && value !== null && !Array.isArray(value) ? value as Record<string, unknown> : null }
function finiteNumber(value: unknown): number | null {
  if (value === null || value === undefined || value === '' || typeof value === 'boolean') return null
  const numeric = typeof value === 'number' ? value : Number(value)
  return Number.isFinite(numeric) ? numeric : null
}
function columnLabel(value: string | null | undefined) { return value ? (COLUMN_LABELS[value] ?? value.replaceAll('_', ' ').toLowerCase()) : 'group' }
function measureLabel(value: string | null | undefined) { return value ? (MEASURE_LABELS[value] ?? value.replaceAll('_', ' ').toLowerCase()) : 'value' }
function titleCase(value: string) { return value.slice(0, 1).toUpperCase() + value.slice(1) }
function pluralLabel(value: string, count: number) { return count === 1 ? value : `${value}s` }

function derivedAnalysis(items: EvidenceItem[]): DerivedAnalysis | null {
  const item = items.find(candidate => candidate.metric === 'derived_analysis')
  const output = asRecord(item?.value)
  if (!item || !output) return null
  const rawSpec = asRecord(item.metadata?.analysis_spec)
  return {
    item,
    output,
    spec: {
      operation: typeof rawSpec?.operation === 'string' ? rawSpec.operation : undefined,
      population: typeof rawSpec?.population === 'string' ? rawSpec.population : undefined,
      group_by: typeof rawSpec?.group_by === 'string' ? rawSpec.group_by : null,
      second_group_by: typeof rawSpec?.second_group_by === 'string' ? rawSpec.second_group_by : null,
      measure: typeof rawSpec?.measure === 'string' ? rawSpec.measure : null,
      second_measure: typeof rawSpec?.second_measure === 'string' ? rawSpec.second_measure : null,
      statistic: typeof rawSpec?.statistic === 'string' ? rawSpec.statistic : null,
    },
  }
}

function readGroupSummary(items: EvidenceItem[]): GroupSummaryModel | null {
  const derived = derivedAnalysis(items)
  if (!derived || derived.spec.operation !== 'group_summary') return null
  const rawGroups = Array.isArray(derived.output.groups) ? derived.output.groups : []
  const rows = rawGroups.flatMap(candidate => {
    const row = asRecord(candidate)
    const group = typeof row?.group === 'string' ? row.group : ''
    const value = finiteNumber(row?.value)
    if (!group || value === null) return []
    return [{ group, value, measuredCount: finiteNumber(row?.measured_count), eligibleCount: finiteNumber(row?.eligible_count), excludedCount: finiteNumber(row?.excluded_count) }]
  }).sort((left, right) => right.value - left.value)
  if (!rows.length) return null
  return {
    groupBy: derived.spec.group_by ?? 'group',
    measure: derived.spec.measure,
    statistic: derived.spec.statistic ?? 'count',
    population: derived.spec.population ?? 'workforce',
    rows,
    populationCount: finiteNumber(derived.item.metadata?.population_count),
    suppressedGroups: Math.max(0, Math.round(finiteNumber(derived.output.suppressed_groups) ?? 0)),
  }
}

function statisticLabel(model: GroupSummaryModel) {
  if (model.statistic === 'count') return 'People'
  if (model.statistic === 'rate') return 'Recorded departure share'
  const statistic = model.statistic === 'mean' ? 'Average' : model.statistic === 'median' ? 'Median' : model.statistic === 'sum' ? 'Total' : titleCase(model.statistic)
  return `${statistic} ${measureLabel(model.measure)}`
}

function formatGroupValue(model: GroupSummaryModel, value: number, reportingCurrency?: string) {
  if (model.statistic === 'rate') return `${(value * 100).toFixed(1)}%`
  if (model.statistic === 'count') return Math.round(value).toLocaleString()
  if (model.measure === 'Salary' || model.statistic === 'sum') return `${Math.round(value).toLocaleString()}${model.measure === 'Salary' && reportingCurrency?.trim() ? ` ${reportingCurrency.trim()}` : ''}`
  if (model.measure === 'Tenure' || model.measure === 'Age') return `${value.toFixed(1)} years`
  if (model.measure === 'LastRating') return `${value.toFixed(1)}/5`
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 })
}

function populationLabel(model: Pick<GroupSummaryModel, 'population'>) { return model.population === 'active' ? 'active workforce' : model.population === 'current' ? 'current workforce' : 'workforce data' }

function friendlyGroupSummary(model: GroupSummaryModel, reportingCurrency?: string) {
  const group = columnLabel(model.groupBy)
  const groupCount = model.rows.length + model.suppressedGroups
  const shownGroups = `${groupCount} ${pluralLabel(group, groupCount)}`
  const top = model.rows[0]
  const topValue = formatGroupValue(model, top.value, reportingCurrency)
  if (model.statistic === 'count') {
    const total = model.populationCount ?? model.rows.reduce((sum, row) => sum + row.value, 0)
    let summary = `There are ${formatGroupValue(model, total, reportingCurrency)} people in the ${populationLabel(model)} across ${shownGroups}. ${top.group} is the largest ${group}, with ${topValue} people.`
    if (model.suppressedGroups) summary += ' Smaller groups are left out of the visual until there is enough data to show them responsibly.'
    return summary
  }
  if (model.statistic === 'rate') {
    let summary = `${top.group} has the highest recorded departure share at ${topValue} among the ${group}s shown. This describes recorded outcomes in the ${populationLabel(model)}; it is not a prediction or a period turnover rate.`
    if (model.suppressedGroups) summary += ' Smaller groups are not shown because there is not enough support for a responsible comparison.'
    return summary
  }
  const lowest = model.rows[model.rows.length - 1]
  let summary = model.rows.length > 1
    ? `${statisticLabel(model)} ranges from ${formatGroupValue(model, lowest.value, reportingCurrency)} to ${topValue} across the ${group}s shown. ${top.group} is highest.`
    : `${top.group} has ${statisticLabel(model).toLowerCase()} of ${topValue}.`
  if (model.suppressedGroups) summary += ' Smaller groups are not shown because there is not enough support for a responsible comparison.'
  return summary
}

function friendlyCorrelation(derived: DerivedAnalysis) {
  const correlation = finiteNumber(derived.output.correlation)
  if (correlation === null) return null
  const absolute = Math.abs(correlation)
  const strength = absolute >= 0.7 ? 'strong' : absolute >= 0.4 ? 'moderate' : absolute >= 0.2 ? 'slight' : 'weak'
  const direction = absolute < 0.05 ? 'no clear' : correlation > 0 ? 'positive' : 'negative'
  const sample = finiteNumber(derived.output.paired_observations)
  const sampleText = sample === null ? '' : ` across ${Math.round(sample).toLocaleString()} people with both values recorded`
  return `The recorded data shows a ${strength} ${direction} relationship between ${measureLabel(derived.spec.measure)} and ${measureLabel(derived.spec.second_measure)}${sampleText}. This is an association in the measured data, not proof that one factor caused the other.`
}

function friendlyDerivedAnswer(items: EvidenceItem[], reportingCurrency?: string) {
  const derived = derivedAnalysis(items)
  if (!derived) return null
  if (derived.spec.operation === 'group_summary') {
    const summary = readGroupSummary(items)
    return summary ? friendlyGroupSummary(summary, reportingCurrency) : null
  }
  if (derived.spec.operation === 'correlation') return friendlyCorrelation(derived)
  if (derived.spec.operation === 'crosstab') {
    const cells = Array.isArray(derived.output.cells) ? derived.output.cells : []
    const supported = cells.filter(cell => { const count = finiteNumber(asRecord(cell)?.count); return count !== null && count > 0 }).length
    const population = populationLabel({ population: derived.spec.population ?? 'workforce' })
    return `PeopleOS found ${supported.toLocaleString()} supported combinations of ${columnLabel(derived.spec.group_by)} and ${columnLabel(derived.spec.second_group_by)} in the ${population}. Smaller combinations are left out when there is not enough support.`
  }
  if (derived.spec.operation === 'compare_groups') {
    const rows = Array.isArray(derived.output.groups) ? derived.output.groups : []
    const readable = rows.slice(0, 2).flatMap(cell => { const record = asRecord(cell); const group = typeof record?.group === 'string' ? record.group : null; const value = finiteNumber(record?.value); return group && value !== null ? [`${group}: ${value.toLocaleString()}`] : [] })
    return readable.length ? `The comparison is ${readable.join(' and ')}. Review the calculation details for the measured support behind each group.` : null
  }
  return null
}

export function percent(value?: number | null) { return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1 ? `${Math.round(value * 100)}%` : 'Unavailable' }
export function userFacingWarning(warning: string) { return /numpy|dtype|traceback|attributeerror|typeerror|valueerror|exception/i.test(warning) ? 'One analytical capability could not contribute evidence. Review the available evidence and gaps before using this answer.' : warning }

const agentStepStatus: Record<AgentStepStatus, { tone: 'success' | 'warning' | 'neutral'; label: string }> = {
  complete: { tone: 'success', label: 'Complete' },
  attention: { tone: 'warning', label: 'Needs attention' },
  skipped: { tone: 'neutral', label: 'Not run' },
}

export function AgentRunPanel({ result }: { result: AgentAnswer }) {
  const steps = result.agent_steps ?? []
  if (!steps.length) return null
  return <Surface padding="md" className="border-violet-200/70 bg-violet-50/35 dark:border-violet-500/20 dark:bg-violet-500/[0.04]">
    <div className="flex items-start gap-3">
      <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-violet-100 text-violet-700 dark:bg-violet-500/15 dark:text-violet-200"><Bot className="h-4 w-4" aria-hidden="true" /></span>
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center justify-between gap-2"><p className="font-semibold text-text-primary">PeopleOS agent</p><StatusBadge tone={result.model ? 'accent' : 'success'}>{result.model ? 'AI assisted' : 'Verified workflow'}</StatusBadge></div>
        <p className="mt-1 text-xs leading-5 text-text-secondary">A governed read-only investigation: approved analytical checks finish first, then local AI explains the completed evidence in People language.</p>
      </div>
    </div>
    <ol aria-label="PeopleOS agent steps" className="mt-4 space-y-3">
      {steps.map(step => {
        const config = agentStepStatus[step.status]
        const Icon = step.status === 'complete' ? CheckCircle2 : step.status === 'attention' ? CircleAlert : Circle
        return <li key={step.id} className="flex items-start gap-2.5">
          <Icon className={`mt-0.5 h-4 w-4 shrink-0 ${step.status === 'complete' ? 'text-emerald-600 dark:text-emerald-400' : step.status === 'attention' ? 'text-amber-600 dark:text-amber-400' : 'text-slate-400'}`} aria-hidden="true" />
          <div className="min-w-0 flex-1"><div className="flex flex-wrap items-center justify-between gap-2"><span className="text-sm font-medium text-text-primary">{step.label}</span><StatusBadge tone={config.tone}>{config.label}</StatusBadge></div><p className="mt-0.5 text-xs leading-5 text-text-secondary">{step.detail}</p></div>
        </li>
      })}
    </ol>
  </Surface>
}

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
  if (item.metric === 'salary_mean') {
    const currency = typeof item.metadata?.reporting_currency === 'string' ? item.metadata.reporting_currency : ''
    return `${label}: ${Math.round(value).toLocaleString()}${currency ? ` ${currency}` : ''}`
  }
  if (item.metric === 'unadjusted_gender_pay_gap_pct') return `${label}: ${value.toFixed(1)}% (unadjusted descriptive comparison; not a legal or causal equity determination)`
  if (item.metric === 'tenure_mean') return `${label}: ${value.toFixed(1)} years`
  if (item.metric === 'age_mean') return `${label}: ${value.toFixed(1)} years`
  if (item.metric === 'lastrating_mean') return `${label}: ${value.toFixed(1)}/5`
  if (['headcount', 'record_count', 'active_count', 'department_count'].includes(item.metric ?? '')) return `${label}: ${Math.round(value).toLocaleString()}`
  return item.claim
}

function numericValue(item?: EvidenceItem) {
  if (!item || item.value === null || item.value === undefined || item.value === '' || typeof item.value === 'boolean') return null
  const value = typeof item.value === 'number' ? item.value : Number(item.value)
  return Number.isFinite(value) ? value : null
}

function requestedMetric(question: string) {
  const q = question.toLowerCase()
  if (/\b(headcount|how many|employee count|workforce size|staff count)\b/.test(q)) return ['active_count', 'headcount']
  if (/\b(attrition|departure)\b/.test(q) && /\b(share|percentage|rate)\b/.test(q)) return ['observed_attrition_share']
  if (/\b(average|mean)\b/.test(q) && /\b(salary|pay|compensation)\b/.test(q)) return ['salary_mean']
  if (/\b(?:gender|sex)\b[^?.;]*\bpay\s+gap\b|\bpay\s+gap\b[^?.;]*\b(?:gender|sex)\b/.test(q)) return ['unadjusted_gender_pay_gap_pct']
  if (/\b(average|mean)\b/.test(q) && /\btenure\b/.test(q)) return ['tenure_mean']
  if (/\b(average|mean)\b/.test(q) && /\bage\b/.test(q)) return ['age_mean']
  if (/\b(average|mean)\b/.test(q) && /\b(rating|performance rating)\b/.test(q)) return ['lastrating_mean']
  if (/\b(how many|number of)\b/.test(q) && /\bdepartments?\b/.test(q)) return ['department_count']
  return []
}

function directAnswer(question: string, items: EvidenceItem[]) {
  const wanted = requestedMetric(question)
  const item = wanted.flatMap(metric => items.filter(candidate => candidate.metric === metric))[0]
  const value = numericValue(item)
  if (!item || value === null) return null

  switch (item.metric) {
    case 'active_count':
    case 'headcount':
      return `Your current active workforce is ${Math.round(value).toLocaleString()} people.`
    case 'observed_attrition_share':
      return `Recorded attrition share is ${(value * 100).toFixed(1)}%. This is the share of known employee outcomes marked as departed, not automatically a period turnover rate.`
    case 'salary_mean': {
      const currency = typeof item.metadata?.reporting_currency === 'string' ? item.metadata.reporting_currency : 'the source reporting currency'
      return `Average active-employee salary is ${Math.round(value).toLocaleString()} ${currency}.`
    }
    case 'unadjusted_gender_pay_gap_pct':
      return `The unadjusted gender pay gap is ${value.toFixed(1)}%. This is a descriptive comparison, not a legal, causal or adjusted equity determination.`
    case 'tenure_mean':
      return `Average active-employee tenure is ${value.toFixed(1)} years.`
    case 'age_mean':
      return `Average active-employee age is ${value.toFixed(1)} years.`
    case 'lastrating_mean':
      return `Average active-employee rating is ${value.toFixed(1)}/5.`
    case 'department_count':
      return `Your current workforce is represented across ${Math.round(value).toLocaleString()} departments.`
    default:
      return null
  }
}

function isStrategicQuestion(question: string) {
  return /\b(?:pay(?:ing)? attention|focus on|watch(?: out)?|priorit(?:y|ise|ize)|what matters|big picture|overall workforce|workforce health|executive summary|how are we doing|where should we look)\b/i.test(question)
}

function firstMetric(items: EvidenceItem[], metric: string) {
  return items.find(item => item.metric === metric && numericValue(item) !== null)
}

function largestOutcomeDisparity(items: EvidenceItem[]) {
  return items
    .filter(item => item.metric === 'attrition_outcome_disparity' && numericValue(item) !== null)
    .sort((left, right) => (numericValue(right) ?? 0) - (numericValue(left) ?? 0))[0]
}

function friendlyStrategicAnswer(question: string, items: EvidenceItem[], reportingCurrency?: string, status?: string) {
  if (!isStrategicQuestion(question)) return null

  const lines: string[] = []
  const active = firstMetric(items, 'active_count') ?? firstMetric(items, 'headcount')
  const departments = firstMetric(items, 'department_count')
  const records = firstMetric(items, 'record_count')
  const departures = firstMetric(items, 'observed_attrition_share')
  const payGap = firstMetric(items, 'unadjusted_gender_pay_gap_pct')
  const experience = firstMetric(items, 'employee_experience_index')
  const disparity = largestOutcomeDisparity(items)
  const spanWarnings = firstMetric(items, 'manager_span_warning_count')
  const stagnation = firstMetric(items, 'critical_stagnation_count')

  if (active) {
    const activeValue = Math.round(numericValue(active) ?? 0).toLocaleString()
    const departmentValue = departments ? Math.round(numericValue(departments) ?? 0).toLocaleString() : null
    lines.push(`Workforce shape: ${activeValue} people are currently recorded as active${departmentValue ? ` across ${departmentValue} departments` : ''}.`)
  }

  if (departures) {
    const share = ((numericValue(departures) ?? 0) * 100).toFixed(1)
    const knownOutcomes = finiteNumber(departures.metadata?.measured_count) ?? finiteNumber(records?.value)
    lines.push(`Recorded departures: ${share}% of ${knownOutcomes === null ? 'the available records' : `${Math.round(knownOutcomes).toLocaleString()} records with a known outcome`} are marked as departed. This is a descriptive snapshot, not a period turnover rate or a prediction.`)
  }

  if (payGap) {
    const gap = (numericValue(payGap) ?? 0).toFixed(1)
    lines.push(`Pay fairness check: the unadjusted gender pay gap is ${gap}%. Use it as a screening signal and compare like-for-like roles before drawing a conclusion.`)
  } else {
    const salary = firstMetric(items, 'salary_mean')
    if (salary) {
      const currency = typeof salary.metadata?.reporting_currency === 'string' && salary.metadata.reporting_currency.trim() ? ` ${salary.metadata.reporting_currency.trim()}` : reportingCurrency?.trim() ? ` ${reportingCurrency.trim()}` : ''
      lines.push(`Pay context: average active-employee salary is ${Math.round(numericValue(salary) ?? 0).toLocaleString()}${currency}.`)
    }
  }

  if (experience) {
    lines.push(`Employee experience: the configured index is ${(numericValue(experience) ?? 0).toFixed(1)}/100. It summarises the fields in this file; it is not a validated engagement score or probability.`)
  }

  if (disparity) {
    const metadata = disparity.metadata ?? {}
    const attribute = typeof metadata.attribute === 'string' && metadata.attribute ? metadata.attribute : 'one workforce group'
    const group = typeof metadata.group === 'string' && metadata.group ? ` (${metadata.group})` : ''
    const subject = group ? `${metadata.group} ${attribute.toLowerCase()}` : attribute
    const groupSize = finiteNumber(metadata.group_size)
    const percentagePoints = ((numericValue(disparity) ?? 0) * 100).toFixed(1)
    lines.push(`Where to look first: ${subject} differs from the overall recorded departure rate by ${percentagePoints} percentage points${groupSize === null ? '' : ` across ${Math.round(groupSize).toLocaleString()} people`}. This is a descriptive follow-up signal, not proof of bias or cause.`)
  }

  if (spanWarnings) {
    lines.push(`Team structure: ${Math.round(numericValue(spanWarnings) ?? 0).toLocaleString()} managers meet the configured span-of-control warning threshold. Review workload and support; this is not a burnout diagnosis.`)
  } else if (stagnation) {
    lines.push(`Career structure: ${Math.round(numericValue(stagnation) ?? 0).toLocaleString()} people meet the configured role-duration threshold. Review progression and role design; this is not proof of stagnation.`)
  }

  if (!lines.length) return null
  const intro = status === 'partial'
    ? 'The current snapshot can support these practical signals:'
    : 'The main practical signals in this snapshot are:'
  return `${intro}\n${lines.map(line => `• ${line}`).join('\n')}\n\nThese signals help decide what to investigate next. They do not explain cause, predict an individual outcome or replace People-team judgement.`
}

function humanAnswer(result: AgentAnswer, items: EvidenceItem[], reportingCurrency?: string) {
  if (result.status === 'insufficient' || result.status === 'unavailable') {
    const reason = result.evidence.unknowns[0] ?? result.warnings[0]
    return reason
      ? `PeopleOS can't answer this reliably from the current data. ${userFacingWarning(reason)}`
      : `PeopleOS can't answer this reliably from the current data.`
  }

  const derived = friendlyDerivedAnswer(items, reportingCurrency)
  if (derived) return derived

  if (result.tools_used.includes('workforce.derived_analysis')) return 'PeopleOS completed a verified workforce calculation. Open the calculation details to inspect the measured support.'

  const strategic = friendlyStrategicAnswer(result.question, items, reportingCurrency, result.status)
  if (strategic) return strategic

  const direct = directAnswer(result.question, items)
  if (direct) return direct

  const useful = items.filter((item, index, all) =>
    numericValue(item) !== null && all.findIndex(candidate => candidate.metric === item.metric && candidate.claim === item.claim) === index
  ).slice(0, 3)

  if (!useful.length) return result.status === 'partial'
    ? 'PeopleOS found some relevant context, but there is not enough measured evidence to give a reliable conclusion.'
    : 'PeopleOS completed the investigation, but there is no concise measured result to surface.'

  const intro = result.status === 'partial'
    ? 'Here is what the available evidence can support so far:'
    : 'Here is what your workforce data shows:'
  return `${intro}\n${useful.map(item => `• ${evidenceClaim(item)}`).join('\n')}`
}

function groundedNarrative(result: AgentAnswer) {
  if (result.synthesis_mode !== 'grounded_llm' || !result.answer.trim()) return null
  return result.answer.replace(/\s*\[(ev_[A-Za-z0-9_-]+)\]/g, '').trim()
}

function GroundedNarrative({ text }: { text: string }) {
  return <div className="rounded-2xl border border-violet-200/80 bg-violet-50/60 p-5 dark:border-violet-400/20 dark:bg-violet-500/[0.08]">
    <div className="flex items-start gap-3">
      <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-violet-600 text-white"><Bot className="h-4 w-4" aria-hidden="true" /></span>
      <div><p className="text-xs font-bold uppercase tracking-[0.16em] text-violet-700 dark:text-violet-200">AI explanation</p><p className="mt-1 text-xs leading-5 text-slate-600 dark:text-slate-300">Composed from the completed analytical checks in this workforce snapshot.</p></div>
    </div>
    <p className="mt-4 whitespace-pre-wrap break-words text-[17px] leading-8 text-slate-800 dark:text-slate-100">{text}</p>
  </div>
}

function GroupSummaryPresentation({ summary, reportingCurrency }: { summary: GroupSummaryModel; reportingCurrency?: string }) {
  const group = columnLabel(summary.groupBy)
  const heading = `${statisticLabel(summary)} by ${group}`
  const formatted = (value: number) => formatGroupValue(summary, value, reportingCurrency)
  const supportHeading = summary.statistic === 'count' ? null : summary.statistic === 'rate' ? 'People with recorded outcome' : 'People with a recorded value'
  return <div className="mt-6 space-y-5">
    <div className="rounded-2xl border border-violet-200/80 bg-violet-50/60 p-5 dark:border-violet-400/20 dark:bg-violet-500/[0.08]">
      <p className="text-xs font-bold uppercase tracking-[0.16em] text-violet-700 dark:text-violet-200">In plain terms</p>
      <p className="mt-2 text-[17px] leading-8 text-slate-800 dark:text-slate-100">{friendlyGroupSummary(summary, reportingCurrency)}</p>
    </div>
    <div className="grid gap-5 lg:grid-cols-[minmax(0,1.35fr)_minmax(260px,0.9fr)]">
      <div className="rounded-2xl border border-slate-200/80 bg-slate-50/60 p-4 dark:border-white/10 dark:bg-white/[0.03]">
        <div className="flex items-start gap-3">
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-white text-violet-700 shadow-sm dark:bg-white/10 dark:text-violet-200"><BarChart3 className="h-4 w-4" /></span>
          <div><h3 className="font-semibold text-slate-900 dark:text-white">{heading}</h3><p className="mt-1 text-xs leading-5 text-slate-500 dark:text-slate-400">Each bar is calculated from the verified workforce snapshot.</p></div>
        </div>
        <figure className="mt-4" aria-labelledby="group-summary-chart-caption">
          <GroupSummaryChart data={summary.rows.map(row => ({ group: row.group, value: row.value }))} valueLabel={statisticLabel(summary)} formatValue={formatted} />
          <figcaption id="group-summary-chart-caption" className="sr-only">{heading}. Hover or focus the bars to read the exact values.</figcaption>
        </figure>
      </div>
      <div className="rounded-2xl border border-slate-200/80 bg-white p-4 dark:border-white/10 dark:bg-slate-950/40">
        <div className="mb-3"><h3 className="font-semibold text-slate-900 dark:text-white">The numbers</h3><p className="mt-1 text-xs leading-5 text-slate-500 dark:text-slate-400">A readable view of every group shown.</p></div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[220px] text-left text-sm">
            <caption className="sr-only">{heading}</caption>
            <thead><tr className="border-b border-slate-200 text-xs text-slate-500 dark:border-white/10 dark:text-slate-400"><th scope="col" className="pb-2 pr-3 font-medium">{titleCase(group)}</th><th scope="col" className="pb-2 text-right font-medium">{statisticLabel(summary)}</th>{supportHeading && <th scope="col" className="pb-2 pl-3 text-right font-medium">{supportHeading}</th>}</tr></thead>
            <tbody>{summary.rows.map(row => <tr key={row.group} className="border-b border-slate-100 last:border-0 dark:border-white/[0.06]"><th scope="row" className="py-2.5 pr-3 font-medium text-slate-800 dark:text-slate-200">{row.group}</th><td className="py-2.5 text-right tabular-nums text-slate-900 dark:text-white">{formatted(row.value)}</td>{supportHeading && <td className="py-2.5 pl-3 text-right tabular-nums text-slate-500 dark:text-slate-400">{row.measuredCount === null ? '—' : Math.round(row.measuredCount).toLocaleString()}</td>}</tr>)}</tbody>
          </table>
        </div>
        {summary.suppressedGroups > 0 && <p className="mt-3 text-xs leading-5 text-slate-500 dark:text-slate-400">{summary.suppressedGroups} smaller {pluralLabel(group, summary.suppressedGroups)} hidden until there is enough support.</p>}
      </div>
    </div>
  </div>
}

export function InvestigationResult({ result, source, reportingCurrency }: { result: AgentAnswer; source: string; reportingCurrency?: string }) {
  const tools = result.evidence.tool_results
  const items = tools.flatMap(tool => tool.evidence)
  const unknowns = result.evidence.unknowns
  const contradictions = result.evidence.contradictions ?? []
  const limitations = Array.from(new Set([...result.warnings, ...tools.flatMap(tool => tool.warnings)]))
  const complete = result.status === 'complete'
  const partial = result.status === 'partial'
  const groupSummary = readGroupSummary(items)
  const narrative = groundedNarrative(result)
  const displayAnswer = humanAnswer(result, items, reportingCurrency)

  return <div className="space-y-4">
    <Surface padding="lg" className="overflow-hidden">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div><p className="text-xs font-bold uppercase tracking-[0.16em] text-violet-600 dark:text-violet-300">PeopleOS answer</p><h2 className="mt-2 max-w-3xl text-xl font-semibold tracking-tight text-slate-950 dark:text-white">{result.question}</h2></div>
        <StatusBadge tone={complete ? 'success' : partial ? 'warning' : 'neutral'}>{complete ? 'Supported by your data' : partial ? 'Some evidence missing' : 'Not enough evidence'}</StatusBadge>
      </div>

      {groupSummary ? <>{narrative && <GroundedNarrative text={narrative} />}<GroupSummaryPresentation summary={groupSummary} reportingCurrency={reportingCurrency} /></> : narrative ? <div className="mt-6"><GroundedNarrative text={narrative} /></div> : <div className="mt-6 whitespace-pre-wrap break-words text-[17px] leading-8 text-slate-800 dark:text-slate-200">{displayAnswer}</div>}

      <div className="mt-6 flex flex-wrap items-center gap-x-4 gap-y-2 border-t border-slate-200/70 pt-4 text-xs text-slate-500 dark:border-white/10 dark:text-slate-400">
        <span className="inline-flex items-center gap-1.5"><CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" />Source: {source}</span>
        <span>{result.synthesis_mode === 'grounded_llm' ? 'AI composed from completed analysis' : 'Calculated from verified data'}</span>
        <span>{items.length} supporting data check{items.length === 1 ? '' : 's'}</span>
      </div>
    </Surface>

    {(unknowns.length > 0 || contradictions.length > 0) && <StateSummary title={contradictions.length ? 'Some evidence conflicts' : 'There are limits to this answer'} description={(contradictions[0] ?? unknowns[0]) ? userFacingWarning(contradictions[0] ?? unknowns[0]) : 'Review the available evidence before acting.'} tone="warning" />}

    <details className="group rounded-2xl border border-slate-200 bg-white dark:border-white/10 dark:bg-slate-900">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-4 px-5 py-4 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500">
        <div className="flex items-center gap-3"><ShieldCheck className="h-5 w-5 text-emerald-600 dark:text-emerald-400" /><div><div className="font-semibold text-slate-900 dark:text-white">Why you can trust this answer</div><div className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">Source, coverage, missing data and calculation checks</div></div></div>
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
      <summary className="flex cursor-pointer list-none items-center justify-between gap-4 px-5 py-4 font-semibold focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500">Calculation details · Evidence ledger ({items.length} checks)<ChevronDown className="h-4 w-4 text-slate-400 transition group-open:rotate-180" /></summary>
      <div className="space-y-3 border-t border-slate-200/70 p-5 dark:border-white/10">{items.map(item => <div key={item.evidence_id} className="rounded-xl border border-border p-4"><p className="text-sm font-medium">{evidenceClaim(item)}</p><EvidencePopulation item={item} /><p className="mt-2 break-words text-xs text-text-muted">Evidence: {item.evidence_id} · Source: {item.source_tool} · {item.kind}</p></div>)}{items.length === 0 && <p className="text-sm text-text-secondary">No supporting evidence was available.</p>}</div>
    </details>

    <details className="group rounded-2xl border border-slate-200 bg-white dark:border-white/10 dark:bg-slate-900">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-4 px-5 py-4 text-sm font-semibold focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500">Technical details<ChevronDown className="h-4 w-4 text-slate-400 transition group-open:rotate-180" /></summary>
      <div className="border-t border-slate-200/70 p-5 text-sm dark:border-white/10">
        <div className="space-y-3">{tools.map(tool => <div key={tool.result_id} className="flex items-start justify-between gap-4 border-b border-border pb-3 last:border-0"><div><div className="font-medium">{tool.tool_id.replaceAll('_', ' ').replaceAll('.', ' · ')}</div><div className="mt-1 text-xs text-text-muted">{userFacingWarning(tool.summary)} · {tool.evidence.length} evidence items</div></div><StatusBadge tone={tool.status === 'success' ? 'success' : tool.status === 'failed' ? 'danger' : 'warning'}>{tool.status}</StatusBadge></div>)}</div>
        <details className="mt-5 rounded-xl border border-border p-4"><summary className="cursor-pointer text-xs font-semibold text-text-secondary">Raw verified response</summary><pre className="mt-3 whitespace-pre-wrap break-words font-sans text-xs leading-5 text-text-muted">{result.answer}</pre></details>
        <p className="mt-4 break-all text-xs text-text-muted">Request: {result.request_id}</p>
      </div>
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
