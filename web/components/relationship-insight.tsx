import Link from 'next/link'
import { ArrowRight, ChevronDown, Lightbulb } from 'lucide-react'
import { StatusBadge } from '@/components/ui'

export type RelationshipContext = 'attrition' | 'experience' | 'hiring'

interface RelationshipInsightProps {
  signal: string
  outcome: string
  correlation: number
  observations?: number | null
  pValue?: number | null
  context?: RelationshipContext
  nextStep?: string
  nextActions?: RelationshipNextAction[]
}

export interface RelationshipNextAction {
  label: string
  question: string
  reason?: string
}

function finiteNumber(value: number | null | undefined) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function strengthFor(value: number) {
  const absolute = Math.abs(value)
  if (absolute >= 0.5) return { label: 'Strong', tone: 'info' as const }
  if (absolute >= 0.3) return { label: 'Moderate', tone: 'info' as const }
  if (absolute >= 0.1) return { label: 'Weak', tone: 'warning' as const }
  return { label: 'Very weak', tone: 'neutral' as const }
}

function relationshipMeaning(signal: string, outcome: string, correlation: number) {
  const absolute = Math.abs(correlation)
  if (absolute < 0.05) {
    const neutralDirection = outcome === 'recorded departures' ? 'more or fewer recorded departures' : `a higher or lower ${outcome}`
    return `${signal} and ${outcome} do not show a clear pattern together in this data. A higher ${signal} does not reliably line up with ${neutralDirection}.`
  }
  const direction = outcome === 'recorded departures'
    ? correlation > 0 ? 'more recorded departures' : 'fewer recorded departures'
    : correlation > 0 ? `a higher ${outcome}` : `a lower ${outcome}`
  const movement = correlation > 0 ? 'in the same direction' : 'in opposite directions'
  return `${signal} and ${outcome} tend to move ${movement}: people with higher ${signal} tended to have ${direction}. The relationship is ${strengthFor(correlation).label.toLowerCase()}, so it is a prompt to investigate rather than a basis for action by itself.`
}

function defaultNextStep(context: RelationshipContext) {
  if (context === 'experience') return 'Check the underlying survey questions and response coverage before treating this as the main lever for experience.'
  if (context === 'hiring') return 'Compare this pattern across roles and future hiring cohorts before changing the interview process.'
  return 'Check this relationship by department, role, location or hiring cohort before treating it as an attrition signal.'
}

function defaultNextActions(context: RelationshipContext): RelationshipNextAction[] {
  if (context === 'experience') return [
    { label: 'Check by department', question: 'Headcount by department', reason: 'Put the experience signal beside the size of each team.' },
    { label: 'Check by location', question: 'Headcount by location', reason: 'See whether the workforce picture differs by location.' },
  ]
  if (context === 'hiring') return [
    { label: 'Check by department', question: 'Headcount by department', reason: 'Put the hiring signal in the context of the workforce groups.' },
    { label: 'Check recorded departures', question: 'Recorded attrition share by department', reason: 'See whether recorded departures differ across the workforce.' },
  ]
  return [
    { label: 'Check by department', question: 'Recorded attrition share by department', reason: 'See where recorded departures are concentrated.' },
    { label: 'Check by location', question: 'Recorded attrition share by location', reason: 'See whether the pattern differs by location.' },
    { label: 'Check by role', question: 'Recorded attrition share by job level', reason: 'Use a role-level cut before drawing a conclusion.' },
  ]
}

function formatPValue(value: number) {
  return value < 0.001 ? '<0.001' : value.toFixed(3)
}

export function RelationshipInsight({ signal, outcome, correlation, observations, pValue, context = 'attrition', nextStep, nextActions }: RelationshipInsightProps) {
  const strength = strengthFor(correlation)
  const sample = finiteNumber(observations)
  const p = finiteNumber(pValue)
  const score = Number.isFinite(correlation) ? correlation.toFixed(2) : 'Unavailable'
  const actions = nextActions ?? defaultNextActions(context)

  return <article aria-label={`${signal} relationship interpretation`} className="flex h-full flex-col rounded-2xl border border-violet-200/80 bg-violet-50/55 p-4 dark:border-violet-400/20 dark:bg-violet-500/[0.07]">
    <div className="flex items-start gap-3">
      <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-white text-violet-700 shadow-sm dark:bg-white/10 dark:text-violet-200"><Lightbulb className="h-4 w-4" aria-hidden="true" /></span>
      <div className="min-w-0 flex-1">
        <p className="text-[11px] font-bold uppercase tracking-[0.14em] text-violet-700 dark:text-violet-300">Relationship signal</p>
        <h3 className="mt-1 text-base font-semibold text-slate-950 dark:text-white">{signal}</h3>
        <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">Compared with {outcome}</p>
      </div>
    </div>
    <div className="mt-4 flex flex-wrap items-center justify-between gap-2">
      <StatusBadge tone={strength.tone}>{strength.label} relationship</StatusBadge>
      <span className="text-xs tabular-nums text-slate-500 dark:text-slate-400">Relationship score {score}</span>
    </div>
    <div className="mt-4 rounded-xl border border-violet-200/70 bg-white/70 px-3 py-3 dark:border-violet-400/15 dark:bg-white/[0.04]">
      <p className="text-xs font-semibold text-slate-700 dark:text-slate-200">What this means in practice</p>
      <p className="mt-1.5 text-sm leading-6 text-slate-700 dark:text-slate-200">{relationshipMeaning(signal, outcome, correlation)}</p>
    </div>
    <p className="mt-3 text-xs leading-5 text-slate-500 dark:text-slate-400">{sample === null ? 'Based on people with both measures recorded.' : `Based on ${Math.round(sample).toLocaleString()} people with both measures recorded.`} This does not tell us that one factor caused the other.</p>
    <div className="mt-auto pt-4">
      <div className="rounded-xl border border-violet-200/70 bg-white/70 px-3 py-2.5 dark:border-violet-400/15 dark:bg-white/[0.04]">
        <p className="text-xs font-semibold text-slate-700 dark:text-slate-200">Useful next check</p>
        <p className="mt-1 text-xs leading-5 text-slate-600 dark:text-slate-400">{nextStep ?? defaultNextStep(context)}</p>
        <div className="mt-3 flex flex-wrap gap-2">
          {actions.map(action => <Link key={`${action.label}-${action.question}`} href={`/advisor?q=${encodeURIComponent(action.question)}`} title={action.reason} className="inline-flex items-center gap-1.5 rounded-lg border border-violet-200 bg-white px-2.5 py-1.5 text-xs font-semibold text-violet-700 transition hover:border-violet-400 hover:bg-violet-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500 dark:border-violet-400/30 dark:bg-slate-950/40 dark:text-violet-200 dark:hover:border-violet-300/60 dark:hover:bg-violet-500/10">{action.label}<ArrowRight className="h-3.5 w-3.5" aria-hidden="true" /></Link>)}
        </div>
        <p className="mt-2 text-[11px] leading-5 text-slate-500 dark:text-slate-400">Each check opens Ask PeopleOS with the question ready to run.</p>
      </div>
      <details className="group mt-3">
        <summary className="flex cursor-pointer list-none items-center gap-1.5 text-xs font-medium text-slate-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500 dark:text-slate-400"><span>Calculation detail</span><ChevronDown className="h-3.5 w-3.5 transition group-open:rotate-180" /></summary>
        <p className="mt-2 text-xs leading-5 text-slate-500 dark:text-slate-400">Technical check: relationship score r={score}{p === null ? '' : ` · sample test p=${formatPValue(p)}`}. A p-value tests whether this pattern is distinguishable from zero in this sample; it is not a probability that the finding is true, a measure of business importance, or proof of cause and effect.</p>
      </details>
    </div>
  </article>
}
