'use client'

import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api-client'
import { Button, Input, MetricCard, Page, PageHeader, SectionHeader, StateSummary, StatusBadge, Surface, Textarea, TrustDisclosure } from '@/components/ui'
import { ScenarioDrilldownBrief, type ScenarioDrilldownBriefData } from '@/components/scenario-drilldown-brief'
import { ArrowRight, BarChart3, Check, Clock3, DollarSign, GitBranch, Play, Save, Sparkles, Target, Users } from 'lucide-react'

type ScenarioType = 'compensation' | 'headcount'

interface ScenarioProvenance {
  workspace_id?: string
  generation: string
  dataset_id?: string
  current_fingerprint?: string
  dataset_version?: number
  source_name?: string
  reporting_currency?: string
}

interface ScenarioResult {
  available: boolean
  scenario_id: string
  scenario_name: string
  scenario_type: string
  input_parameters: Record<string, unknown>
  provenance?: ScenarioProvenance
  affected_employees: number
  affected_departments: string[]
  baseline_turnover_rate: number
  projected_turnover_rate: number
  turnover_change: number
  turnover_change_pct: number
  simulation: { n_iterations: number; outcome_mean: number; outcome_std: number; roi_mean: number; roi_positive_probability: number }
  cost_impact: { salary_change: number; total_cost: number; total_benefit: number; net_impact: number }
  roi_estimate: number | null
  payback_months: number | null
  confidence_level: string
  confidence_score: number
  recommendation: string
  risks: string[]
  assumptions: string[]
  alternative_actions: string[]
  data_sources: string[]
  engines_used: string[]
  computed_at: string
  cost_semantics?: { summary?: string[]; payback_available?: boolean; decision_boundary?: string }
}

interface ScenarioSummary {
  scenario_id: string
  scenario_name: string
  scenario_type: string
  computed_at: string
  roi_estimate: number | null
  evidence_strength: string
  provenance?: ScenarioProvenance
}

interface ScenarioHistory {
  available: boolean
  count: number
  scenarios: ScenarioSummary[]
}

interface ComparisonItem {
  scenario_id: string
  scenario_name: string
  affected_employees: number
  turnover_change_pct: number
  roi_estimate: number | null
  net_impact: number
  confidence_level: string
  roi_positive_probability: number
}

interface ScenarioComparison {
  available: boolean
  scenarios: ComparisonItem[]
  recommended_scenario: string
  reasoning: string
}

interface ScenarioDrilldown extends ScenarioDrilldownBriefData {
  available: boolean
  focus: string
  model: string | null
  comparison: ScenarioComparison
}

function money(value: number, currency?: string) {
  if (!Number.isFinite(value)) return 'Unavailable'
  const abs = Math.abs(value)
  const amount = abs >= 1_000_000 ? `${(value / 1_000_000).toFixed(1)}M` : abs >= 1_000 ? `${(value / 1_000).toFixed(0)}K` : `${Math.round(value).toLocaleString()}`
  return currency ? `${currency} ${amount}` : amount
}

function scenarioType(value: string) {
  return value === 'headcount' ? 'Add people' : 'Change pay'
}

function when(value: string) {
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
}

function comparisonTakeaway(items: ComparisonItem[], currency?: string) {
  if (items.length !== 2) return 'Review both situations side by side before interpreting the difference.'

  const [first, second] = items
  const impactDelta = second.net_impact - first.net_impact
  const impact = Math.abs(impactDelta) < 0.5
    ? 'The modeled financial result is effectively the same in both situations.'
    : `In plain terms, the model puts ${second.scenario_name} at ${money(Math.abs(impactDelta), currency)} ${impactDelta < 0 ? 'less' : 'more'} net value than ${first.scenario_name}.`
  const outcomeDelta = second.turnover_change_pct - first.turnover_change_pct
  const outcome = Math.abs(outcomeDelta) < 0.05
    ? `The modeled change in turnover is unchanged in both situations (${first.turnover_change_pct.toFixed(1)}% in each).`
    : `The model shows a ${Math.abs(outcomeDelta).toFixed(1)} percentage-point difference in the change in turnover between the two situations.`
  const scope = first.affected_employees === second.affected_employees
    ? `Both situations cover ${first.affected_employees.toLocaleString()} people.`
    : `The situations cover different groups (${first.affected_employees.toLocaleString()} versus ${second.affected_employees.toLocaleString()} people), so treat the comparison carefully.`

  return `${impact} ${outcome} ${scope} Use the AI brief below to see what this means for People and what to validate next.`
}

export default function ScenarioPlannerPage() {
  const queryClient = useQueryClient()
  const [type, setType] = useState<ScenarioType>('compensation')
  const [storedResult, setResult] = useState<ScenarioResult | null>(null)
  const [scenarioName, setScenarioName] = useState('')
  const [savedMessage, setSavedMessage] = useState('')
  const [adjustmentValue, setAdjustmentValue] = useState(5)
  const [targetScope, setTargetScope] = useState<'all' | 'department'>('all')
  const [targetDept, setTargetDept] = useState('')
  const [changeCount, setChangeCount] = useState(10)
  const [compareSelection, setCompareSelection] = useState<string[]>([])
  const [comparison, setComparison] = useState<ScenarioComparison | null>(null)
  const [drilldownQuestion, setDrilldownQuestion] = useState('What is the difference between these scenarios, and what should we validate next?')
  const [drilldown, setDrilldown] = useState<ScenarioDrilldown | null>(null)

  const { data: departmentsData } = useQuery({ queryKey: ['analytics', 'departments'], queryFn: () => api.analytics.getDepartments() })
  const departments: string[] = (departmentsData as { departments?: Array<{ dept: string }> })?.departments?.map(item => item.dept) ?? []
  const { data: status } = useQuery({ queryKey: ['platform', 'status'], queryFn: () => api.getStatus() as Promise<{ integrity?: { snapshot?: { generation?: string; dataset_id?: string; current_fingerprint?: string; reporting_currency?: string } } }> })
  const historyQuery = useQuery({
    queryKey: ['scenario', 'history'],
    queryFn: () => api.scenario.getRecentScenarios(20) as Promise<ScenarioHistory>,
  })

  const snapshot = status?.integrity?.snapshot
  const resultMatchesSnapshot = Boolean(
    storedResult && (!snapshot?.generation || (
      storedResult.provenance?.generation === snapshot.generation &&
      storedResult.provenance.dataset_id === snapshot.dataset_id &&
      storedResult.provenance.current_fingerprint === snapshot.current_fingerprint
    )),
  )
  const result = resultMatchesSnapshot ? storedResult : null
  const staleResult = Boolean(storedResult && !resultMatchesSnapshot)
  const reportingCurrency = snapshot?.reporting_currency ?? result?.provenance?.reporting_currency
  const inputValid = type === 'compensation'
    ? Number.isFinite(adjustmentValue) && adjustmentValue >= 0 && adjustmentValue <= 100
    : Number.isInteger(changeCount) && changeCount >= 1 && changeCount <= 100000

  const handleScenarioResult = (data: ScenarioResult) => {
    setResult(data)
    setScenarioName(data.scenario_name)
    setSavedMessage('')
    setComparison(null)
    setDrilldown(null)
    setCompareSelection(current => current.includes(data.scenario_id) ? current : [...current.slice(-1), data.scenario_id])
    void queryClient.invalidateQueries({ queryKey: ['scenario', 'history'] })
  }

  const compensation = useMutation<ScenarioResult, Error>({
    mutationFn: async () => api.scenario.simulateCompensation({
      adjustment_type: 'percentage',
      target: { scope: targetScope, department: targetScope === 'department' ? targetDept : undefined },
      adjustment_value: adjustmentValue,
      time_horizon_months: 12,
    }) as Promise<ScenarioResult>,
    onSuccess: handleScenarioResult,
  })
  const headcount = useMutation<ScenarioResult, Error>({
    mutationFn: async () => api.scenario.simulateHeadcount({
      change_type: 'expansion',
      target: { scope: targetScope, department: targetScope === 'department' ? targetDept : undefined },
      change_count: changeCount,
      selection_criteria: 'performance',
    }) as Promise<ScenarioResult>,
    onSuccess: handleScenarioResult,
  })
  const saveMutation = useMutation<ScenarioResult, Error, { scenario_id: string; scenario_name: string }>({
    mutationFn: async request => api.scenario.saveScenario(request) as Promise<ScenarioResult>,
    onSuccess: data => {
      setResult(data)
      setScenarioName(data.scenario_name)
      setSavedMessage('Name updated in your local library')
      void queryClient.invalidateQueries({ queryKey: ['scenario', 'history'] })
    },
  })
  const openMutation = useMutation<ScenarioResult, Error, string>({
    mutationFn: async scenarioId => api.scenario.getScenario(scenarioId) as Promise<ScenarioResult>,
    onSuccess: data => {
      setResult(data)
      setScenarioName(data.scenario_name)
      setSavedMessage('')
      setComparison(null)
      setDrilldown(null)
    },
  })
  const compareMutation = useMutation<ScenarioComparison, Error>({
    mutationFn: async () => api.scenario.compareScenarios(compareSelection) as Promise<ScenarioComparison>,
    onSuccess: data => {
      setComparison(data)
      setDrilldown(null)
    },
  })
  const drilldownMutation = useMutation<ScenarioDrilldown, Error>({
    mutationFn: async () => api.scenario.drilldownScenarios({ scenario_ids: compareSelection, question: drilldownQuestion }) as Promise<ScenarioDrilldown>,
    onSuccess: setDrilldown,
  })

  const pending = compensation.isPending || headcount.isPending || saveMutation.isPending || openMutation.isPending || compareMutation.isPending || drilldownMutation.isPending
  const failure = compensation.error || headcount.error || saveMutation.error || openMutation.error || compareMutation.error || drilldownMutation.error || historyQuery.error

  const history = historyQuery.data?.scenarios ?? []

  const run = () => {
    compensation.reset()
    headcount.reset()
    setResult(null)
    setSavedMessage('')
    setComparison(null)
    setDrilldown(null)
    if (type === 'compensation') compensation.mutate()
    else headcount.mutate()
  }

  const toggleCompare = (scenarioId: string) => {
    setCompareSelection(current => {
      if (current.includes(scenarioId)) return current.filter(item => item !== scenarioId)
      return current.length >= 2 ? [current[1], scenarioId] : [...current, scenarioId]
    })
    setComparison(null)
    setDrilldown(null)
  }

  return <Page>
    <PageHeader
      eyebrow="Plan"
      title="What if we changed something?"
      description="Explore a pay or hiring assumption, save the situation, and compare it with another before taking it into a real decision."
    />

    {failure && <StateSummary title="This planning action could not be completed" description={failure instanceof Error ? failure.message : 'Review the inputs and try again.'} tone="warning" />}
    {staleResult && <StateSummary title="Your data changed" description="The previous scenario has been hidden. Run it again using the current workforce data." tone="warning" />}

    <div className="grid gap-6 xl:grid-cols-[340px_minmax(0,1fr)]">
      <div className="space-y-6">
        <Surface padding="lg">
          <fieldset disabled={pending}>
            <SectionHeader title="Build a scenario" description="Choose what you want to explore." />
            <div className="mt-5 grid grid-cols-2 gap-2">
              <Button size="sm" variant={type === 'compensation' ? 'primary' : 'secondary'} onClick={() => { setResult(null); setType('compensation') }}><DollarSign className="h-4 w-4" />Change pay</Button>
              <Button size="sm" variant={type === 'headcount' ? 'primary' : 'secondary'} onClick={() => { setResult(null); setType('headcount') }}><Users className="h-4 w-4" />Add people</Button>
            </div>

            <div className="mt-6 space-y-5">
              {type === 'compensation' && <Input label="Pay change (%)" type="number" min={0} max={100} step="0.1" value={adjustmentValue} onChange={event => { setResult(null); setAdjustmentValue(Number(event.target.value)) }} error={Number.isFinite(adjustmentValue) && adjustmentValue >= 0 && adjustmentValue <= 100 ? undefined : 'Enter a value from 0% to 100%.'} helperText="PeopleOS models pay increases here; pay reductions and employee actions are outside this exploratory planner." />}
              {type === 'headcount' && <Input label="Additional positions" type="number" min={1} max={100000} step="1" value={changeCount} onChange={event => { setResult(null); setChangeCount(Number(event.target.value)) }} error={Number.isInteger(changeCount) && changeCount >= 1 && changeCount <= 100000 ? undefined : 'Enter a whole number from 1 to 100,000.'} helperText="PeopleOS models aggregate expansion only; person-level selection for reductions is outside this planner." />}

              <div><label htmlFor="scenario-scope" className="mb-2 block text-sm font-medium">Who does this apply to?</label><select id="scenario-scope" value={targetScope} onChange={event => { setResult(null); setTargetScope(event.target.value as 'all' | 'department') }} className="h-10 w-full rounded-xl border border-border bg-surface px-3 text-sm focus:outline-none focus:ring-2 focus:ring-accent/30"><option value="all">Whole workforce</option><option value="department">One department</option></select></div>
              {targetScope === 'department' && <div><label htmlFor="scenario-department" className="mb-2 block text-sm font-medium">Department</label><select id="scenario-department" value={targetDept} onChange={event => { setResult(null); setTargetDept(event.target.value) }} className="h-10 w-full rounded-xl border border-border bg-surface px-3 text-sm focus:outline-none focus:ring-2 focus:ring-accent/30"><option value="">Choose a department</option>{departments.map(department => <option key={department} value={department}>{department}</option>)}</select></div>}
              <Button onClick={run} isLoading={compensation.isPending || headcount.isPending} disabled={!inputValid || (targetScope === 'department' && !targetDept)} className="w-full"><Play className="h-4 w-4" />Explore scenario</Button>
            </div>
          </fieldset>
        </Surface>

        <Surface padding="lg">
          <SectionHeader title="Scenario library" description="Calculations stay local and tied to the dataset they used. Select two to compare." />
          {historyQuery.isLoading && <p className="text-sm text-text-secondary">Loading recent scenarios…</p>}
          {!historyQuery.isLoading && history.length === 0 && <p className="text-sm leading-6 text-text-secondary">Run your first scenario and it will appear here. Name it when you want to keep it easy to find.</p>}
          <div className="space-y-2">
            {history.map(item => {
              const selected = compareSelection.includes(item.scenario_id)
              return <div key={item.scenario_id} className={`rounded-xl border p-3 transition-colors ${selected ? 'border-violet-300 bg-violet-50/70 dark:border-violet-400/40 dark:bg-violet-500/10' : 'border-border bg-surface'}`}>
                <div className="flex items-start gap-3">
                  <input type="checkbox" aria-label={`Select ${item.scenario_name} for comparison`} checked={selected} onChange={() => toggleCompare(item.scenario_id)} className="mt-1 h-4 w-4 accent-violet-600" />
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-sm font-semibold">{item.scenario_name}</div>
                    <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-text-muted"><span>{scenarioType(item.scenario_type)}</span><span>·</span><span className="inline-flex items-center gap-1"><Clock3 className="h-3 w-3" />{when(item.computed_at)}</span></div>
                  </div>
                  <Button size="sm" variant="ghost" onClick={() => openMutation.mutate(item.scenario_id)} disabled={pending}>Open<ArrowRight className="h-3.5 w-3.5" /></Button>
                </div>
              </div>
            })}
          </div>
          <div className="mt-4 flex items-center justify-between gap-3 border-t border-border pt-4"><span className="text-xs text-text-muted">{compareSelection.length} of 2 selected</span><Button size="sm" onClick={() => compareMutation.mutate()} disabled={compareSelection.length !== 2 || pending}><GitBranch className="h-4 w-4" />Compare selected</Button></div>
          <p className="mt-3 text-xs leading-5 text-text-muted">A comparison never picks a winner automatically. It keeps both situations visible so People and Finance can review the trade-off.</p>
        </Surface>
      </div>

      <div className="space-y-6">
        {!result || !result.available ? (!comparison && <Surface padding="lg" className="grid min-h-[360px] place-items-center"><div className="max-w-lg text-center"><div className="mx-auto grid h-12 w-12 place-items-center rounded-2xl bg-accent/10 text-accent"><GitBranch className="h-6 w-6" /></div><h2 className="mt-5 text-xl font-semibold">Choose a scenario to explore</h2><p className="mt-2 text-sm leading-6 text-text-secondary">PeopleOS will show the modeled cost and workforce effect, then keep the assumptions available for review and comparison.</p></div></Surface>) : <>
          <Surface padding="lg" className="border-violet-200/70 dark:border-violet-400/20">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between"><div><SectionHeader title="Name this situation" description="Each calculation is saved locally as soon as it runs. Give it a name you will recognize when comparing it later." /><div className="mt-1 flex items-center gap-2 text-xs text-text-muted"><Check className="h-3.5 w-3.5 text-emerald-600" />Already saved in your local scenario library · naming it is optional.</div></div>{savedMessage && <StatusBadge tone="success">{savedMessage}</StatusBadge>}</div>
            <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-end"><div className="min-w-0 flex-1"><Input label="Scenario name" value={scenarioName} onChange={event => { setScenarioName(event.target.value); setSavedMessage('') }} placeholder="e.g. 5% pay review — whole workforce" /></div><Button onClick={() => result && saveMutation.mutate({ scenario_id: result.scenario_id, scenario_name: scenarioName })} disabled={!scenarioName.trim() || pending}><Save className="h-4 w-4" />Name and save</Button></div>
          </Surface>

          <p className="text-sm text-text-secondary">Using {result.provenance?.source_name ?? 'your active workforce data'}{result.provenance?.dataset_version ? ` · version ${result.provenance.dataset_version}` : ''}{reportingCurrency ? ` · monetary amounts in ${reportingCurrency}` : ''}</p>
          <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <MetricCard label="People in scope" value={result.affected_employees.toLocaleString()} detail={result.affected_departments.length ? result.affected_departments.join(', ') : 'Configured scope'} icon={Users} />
            <MetricCard label="Recorded starting rate" value={`${result.baseline_turnover_rate.toFixed(1)}%`} detail="Reference workforce outcome used by the model" icon={Target} />
            <MetricCard label="Estimated workforce outcome" value={`${result.projected_turnover_rate.toFixed(1)}%`} detail="Modeled result under this assumption; not a forecast" icon={BarChart3} />
            <MetricCard label={reportingCurrency ? `Estimated net financial effect (${reportingCurrency})` : 'Estimated net financial effect'} value={money(result.cost_impact.net_impact, reportingCurrency)} valueClassName="whitespace-normal break-words text-xl leading-tight" detail={result.roi_estimate == null ? 'Return estimate unavailable' : `Estimated return: ${result.roi_estimate.toFixed(1)}%`} icon={DollarSign} tone={result.cost_impact.net_impact >= 0 ? 'success' : 'warning'} />
          </section>

          <div className="rounded-2xl border border-violet-200/80 bg-violet-50/55 p-4 dark:border-violet-400/20 dark:bg-violet-500/[0.07]"><div className="text-sm font-semibold text-slate-950 dark:text-white">What this means for a People or Finance conversation</div><p className="mt-1 text-sm leading-6 text-slate-700 dark:text-slate-200">This is a way to frame the cost and workforce trade-off under stated assumptions. Use it to decide what to validate with Finance and People owners; it does not predict employee behavior or approve a change.</p></div>

          <div className="grid gap-6 lg:grid-cols-[minmax(0,1.1fr)_minmax(300px,0.9fr)]">
            <Surface padding="lg"><SectionHeader title="What the scenario says" description="A concise interpretation of the modeled result." /><div className="mt-5"><StatusBadge tone="info">Exploratory</StatusBadge><p className="mt-4 text-sm leading-7 text-text-secondary">{result.recommendation}</p><div className="mt-5 grid grid-cols-2 gap-3"><div className="rounded-xl border border-border p-4"><div className="text-xl font-semibold">{money(result.cost_impact.total_cost, reportingCurrency)}</div><div className="text-xs text-text-muted">estimated implementation cost</div></div><div className="rounded-xl border border-border p-4"><div className="text-xl font-semibold">{money(result.cost_impact.total_benefit, reportingCurrency)}</div><div className="text-xs text-text-muted">estimated financial benefit</div></div><div className="rounded-xl border border-border p-4"><div className="text-xl font-semibold">{result.payback_months == null ? 'Unavailable' : `${result.payback_months} mo`}</div><div className="text-xs text-text-muted">simple payback estimate</div></div></div></div></Surface>
            <Surface padding="lg"><SectionHeader title="Things to pressure-test" description="What a People or Finance owner should validate before using this scenario." /><div className="mt-5 space-y-3">{result.risks.slice(0, 4).map(item => <div key={item} className="rounded-xl border border-border p-3 text-sm leading-6 text-text-secondary">{item}</div>)}</div></Surface>
          </div>

          <TrustDisclosure title="How this scenario was modeled" summary={`${result.simulation.n_iterations.toLocaleString()} assumption draws`}>
            <p>Scenario results are deterministic calculations over configured assumptions and your current workforce data. They are not causal forecasts or guarantees.</p><div className="mt-3"><div className="font-semibold text-slate-700 dark:text-slate-200">Cost timing</div><ul className="mt-1 space-y-1">{(result.cost_semantics?.summary ?? []).map(item => <li key={item}>• {item}</li>)}</ul></div><div className="mt-3"><div className="font-semibold text-slate-700 dark:text-slate-200">Assumptions used</div><ul className="mt-1 space-y-1">{result.assumptions.slice(0, 8).map(item => <li key={item}>• {item}</li>)}</ul></div><p className="mt-3">{(result.simulation.roi_positive_probability * 100).toFixed(0)}% of configured simulation draws produced positive modeled ROI. This is a configured draw share, not an empirical probability that the investment will succeed.</p>
          </TrustDisclosure>
        </>}

        {comparison && <Surface padding="lg" className="border-violet-200/70 dark:border-violet-400/20"><SectionHeader title="Side-by-side comparison" description="The second selected situation is shown against the first. Differences are descriptive, not a verdict." action={<StatusBadge tone="info">No automatic winner</StatusBadge>} />
          <div className="grid gap-3 md:grid-cols-2">{comparison.scenarios.map((item, index) => <div key={item.scenario_id} className="rounded-2xl border border-border bg-surface p-4"><div className="flex items-start justify-between gap-3"><div><div className="text-xs font-bold uppercase tracking-[0.14em] text-violet-600">{index === 0 ? 'First situation' : 'Second situation'}</div><div className="mt-1 text-sm font-semibold">{item.scenario_name}</div></div><StatusBadge tone="neutral">{item.confidence_level}</StatusBadge></div><div className="mt-4 grid grid-cols-2 gap-3"><div><div className="text-lg font-semibold">{money(item.net_impact, reportingCurrency)}</div><div className="text-xs text-text-muted">estimated net financial effect</div></div><div><div className="text-lg font-semibold">{item.affected_employees.toLocaleString()}</div><div className="text-xs text-text-muted">people in scope</div></div><div><div className="text-lg font-semibold">{item.turnover_change_pct.toFixed(1)}%</div><div className="text-xs text-text-muted">change in modeled workforce outcome</div></div><div><div className="text-lg font-semibold">{(item.roi_positive_probability * 100).toFixed(0)}%</div><div className="text-xs text-text-muted">share of assumption runs with positive result</div></div></div></div>)}</div>
          <div className="mt-4 rounded-xl bg-slate-50 p-4 text-sm leading-6 text-text-secondary dark:bg-white/[0.04]"><div className="font-semibold text-text-primary">What changed between them?</div><p className="mt-1">{comparisonTakeaway(comparison.scenarios, reportingCurrency)}</p></div>

          <div className="mt-6 rounded-2xl border border-violet-200 bg-violet-50/60 p-4 dark:border-violet-400/20 dark:bg-violet-500/10"><div className="flex items-start gap-3"><div className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-violet-600 text-white"><Sparkles className="h-4 w-4" /></div><div className="min-w-0 flex-1"><div className="flex flex-wrap items-center gap-2"><div className="font-semibold">AI-guided drill-down</div>{drilldown && <StatusBadge tone={drilldown.status === 'complete' ? 'success' : 'warning'}>{drilldown.status === 'complete' ? `Local AI · ${drilldown.model ?? 'ready'}` : 'Verified fallback'}</StatusBadge>}</div><p className="mt-1 text-sm leading-6 text-text-secondary">Ask in normal People language. The local AI prioritizes what to inspect; PeopleOS keeps the numbers tied to verified scenario evidence.</p></div></div><div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-end"><div className="min-w-0 flex-1"><Textarea label="What would you like to understand?" rows={2} value={drilldownQuestion} onChange={event => setDrilldownQuestion(event.target.value)} /></div><Button onClick={() => drilldownMutation.mutate()} disabled={!drilldownQuestion.trim() || compareSelection.length !== 2 || pending} isLoading={drilldownMutation.isPending}><Sparkles className="h-4 w-4" />Explain this comparison</Button></div>{drilldown && <ScenarioDrilldownBrief drilldown={drilldown} />}</div>
          <TrustDisclosure title="Decision boundary" summary="Why PeopleOS keeps the final choice with people"><p>{comparison.reasoning}</p></TrustDisclosure>
        </Surface>}
      </div>
    </div>
  </Page>
}
