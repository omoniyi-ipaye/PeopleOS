import { CheckCircle2, ShieldCheck, Target } from 'lucide-react'
import { StatusBadge } from '@/components/ui'

export interface ScenarioDrilldownBriefData {
  status: 'complete' | 'fallback'
  answer: string
  focus_label?: string
  headline?: string
  people_takeaway?: string
  use_for?: string[]
  validate_next?: string[]
  decision_boundary?: string
  selected_evidence: string[]
  warnings: string[]
}

const evidenceLabels: Record<string, string> = {
  impact: 'Modeled financial impact',
  outcome: 'Modeled people outcome',
  scope: 'People in scope',
  uncertainty: 'Model uncertainty',
  assumptions: 'Scenario assumptions',
}

export function ScenarioDrilldownBrief({ drilldown }: { drilldown: ScenarioDrilldownBriefData }) {
  const useFor = (drilldown.use_for ?? []).filter(Boolean)
  const validateNext = (drilldown.validate_next ?? []).filter(Boolean)
  const evidence = (drilldown.selected_evidence ?? []).filter(Boolean)

  return <div className="mt-4 rounded-2xl border border-violet-200 bg-white/90 p-4 shadow-sm dark:border-violet-400/20 dark:bg-slate-950/40">
    <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
      <div className="min-w-0">
        <p className="text-[11px] font-bold uppercase tracking-[0.15em] text-violet-700 dark:text-violet-300">People takeaway</p>
        <h3 className="mt-1 text-lg font-semibold leading-7 text-slate-950 dark:text-white">{drilldown.headline || 'Review the difference between these two situations.'}</h3>
      </div>
      <StatusBadge tone={drilldown.status === 'complete' ? 'success' : 'warning'}>{drilldown.focus_label || 'Comparison focus'}</StatusBadge>
    </div>

    <div className="mt-4 grid gap-3 lg:grid-cols-2">
      <div className="rounded-xl border border-violet-200/80 bg-violet-50/60 p-4 dark:border-violet-400/15 dark:bg-violet-500/[0.08]">
        <div className="flex items-center gap-2 text-sm font-semibold text-slate-900 dark:text-white"><Target className="h-4 w-4 text-violet-600 dark:text-violet-300" aria-hidden="true" />What this means for People</div>
        <p className="mt-2 text-sm leading-6 text-slate-700 dark:text-slate-200">{drilldown.people_takeaway || 'Use the comparison to understand the trade-off, not to predict what employees will do.'}</p>
      </div>
      <div className="rounded-xl border border-slate-200 bg-slate-50/80 p-4 dark:border-white/10 dark:bg-white/[0.04]">
        <div className="flex items-center gap-2 text-sm font-semibold text-slate-900 dark:text-white"><CheckCircle2 className="h-4 w-4 text-emerald-600 dark:text-emerald-300" aria-hidden="true" />How to use this</div>
        {useFor.length ? <ul className="mt-2 space-y-2 text-sm leading-6 text-slate-700 dark:text-slate-200">{useFor.map(item => <li key={item} className="flex items-start gap-2"><span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-emerald-500" aria-hidden="true" />{item}</li>)}</ul> : <p className="mt-2 text-sm leading-6 text-slate-700 dark:text-slate-200">Use this comparison to frame the trade-off and decide what needs validation.</p>}
      </div>
    </div>

    <div className="mt-3 rounded-xl border border-amber-200 bg-amber-50/80 p-4 dark:border-amber-400/20 dark:bg-amber-500/[0.08]">
      <div className="flex items-center gap-2 text-sm font-semibold text-amber-950 dark:text-amber-100"><ShieldCheck className="h-4 w-4 text-amber-700 dark:text-amber-300" aria-hidden="true" />What to validate next</div>
      {validateNext.length ? <ol className="mt-2 space-y-2 text-sm leading-6 text-amber-950/80 dark:text-amber-100/80">{validateNext.map((item, index) => <li key={item} className="flex items-start gap-2"><span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-amber-200 text-[11px] font-bold text-amber-900 dark:bg-amber-400/20 dark:text-amber-100">{index + 1}</span>{item}</li>)}</ol> : <p className="mt-2 text-sm leading-6 text-amber-950/80 dark:text-amber-100/80">Confirm the assumptions with the accountable People and Finance owners.</p>}
    </div>

    <div className="mt-3 rounded-xl border border-slate-200 bg-slate-50/80 p-3 dark:border-white/10 dark:bg-white/[0.04]">
      <div className="flex items-center gap-2 text-xs font-semibold text-slate-700 dark:text-slate-200"><ShieldCheck className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-300" aria-hidden="true" />Decision boundary</div>
      <p className="mt-1 text-xs leading-5 text-slate-600 dark:text-slate-400">{drilldown.decision_boundary || 'This is exploratory planning evidence, not a forecast or a recommendation to take workforce action.'}</p>
    </div>

    {evidence.length > 0 && <div className="mt-3 flex flex-wrap items-center gap-2"><span className="text-xs font-medium text-slate-500 dark:text-slate-400">Evidence used:</span>{evidence.map(item => <StatusBadge key={item} tone="neutral">{evidenceLabels[item] ?? item}</StatusBadge>)}</div>}

    <details className="group mt-4 border-t border-slate-200 pt-3 dark:border-white/10">
      <summary className="cursor-pointer list-none text-xs font-semibold text-slate-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500 dark:text-slate-300">Show evidence explanation <span className="ml-1 inline-block transition group-open:rotate-180">⌄</span></summary>
      <div className="mt-3 whitespace-pre-line text-sm leading-6 text-slate-600 dark:text-slate-400">{drilldown.answer}</div>
    </details>

    {drilldown.warnings.map(warning => <div key={warning} className="mt-3 rounded-lg bg-amber-50 p-3 text-xs leading-5 text-amber-900 dark:bg-amber-400/10 dark:text-amber-100">{warning}</div>)}
  </div>
}
