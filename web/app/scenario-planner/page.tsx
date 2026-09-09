'use client'

import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api-client'
import { Button, Input, MetricCard, Page, PageHeader, SectionHeader, StateSummary, StatusBadge, Surface, TrustDisclosure } from '@/components/ui'
import { BarChart3, DollarSign, GitBranch, Play, Target, Users } from 'lucide-react'

type ScenarioType = 'compensation' | 'headcount'
interface ScenarioResult {
  provenance?: {generation: string; dataset_version?: number; source_name?: string}
  available: boolean
  scenario_name: string
  scenario_type: string
  affected_employees: number
  affected_departments: string[]
  baseline_turnover_rate: number
  projected_turnover_rate: number
  turnover_change: number
  simulation: { n_iterations: number; outcome_mean: number; outcome_std: number; roi_mean: number; roi_positive_probability: number }
  cost_impact: { salary_change: number; total_cost: number; total_benefit: number; net_impact: number }
  roi_estimate: number | null
  payback_months: number | null
  confidence_level: string
  recommendation: string
  risks: string[]
  assumptions: string[]
  alternative_actions: string[]
  data_sources: string[]
  engines_used: string[]
}

function money(value: number) {
  if (!Number.isFinite(value)) return 'Unavailable'
  const abs = Math.abs(value)
  if (abs >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`
  if (abs >= 1_000) return `${(value / 1_000).toFixed(0)}K`
  return `${Math.round(value).toLocaleString()}`
}

export default function ScenarioPlannerPage() {
  const [type, setType] = useState<ScenarioType>('compensation')
  const [storedResult, setResult] = useState<ScenarioResult | null>(null)
  const [adjustmentValue, setAdjustmentValue] = useState(5)
  const [targetScope, setTargetScope] = useState<'all' | 'department'>('all')
  const [targetDept, setTargetDept] = useState('')
  const [changeCount, setChangeCount] = useState(10)

  const { data: departmentsData } = useQuery({ queryKey: ['analytics', 'departments'], queryFn: () => api.analytics.getDepartments() })
  const departments: string[] = (departmentsData as { departments?: Array<{ dept: string }> })?.departments?.map(item => item.dept) ?? []
  const compensation = useMutation({ mutationFn: () => api.scenario.simulateCompensation({ adjustment_type: 'percentage', target: { scope: targetScope, department: targetScope === 'department' ? targetDept : undefined }, adjustment_value: adjustmentValue, time_horizon_months: 12 }), onSuccess: data => setResult(data as ScenarioResult) })
  const headcount = useMutation({ mutationFn: () => api.scenario.simulateHeadcount({ change_type: 'expansion', target: { scope: targetScope, department: targetScope === 'department' ? targetDept : undefined }, change_count: changeCount, selection_criteria: 'performance' }), onSuccess: data => setResult(data as ScenarioResult) })
  const { data: status } = useQuery({ queryKey: ['platform', 'status'], queryFn: () => api.getStatus() as Promise<{integrity?: {snapshot?: {generation?: string}}}> })

  const result = storedResult?.provenance?.generation === status?.integrity?.snapshot?.generation ? storedResult : null
  const staleResult = Boolean(storedResult && storedResult.provenance?.generation !== status?.integrity?.snapshot?.generation)
  const loading = compensation.isPending || headcount.isPending
  const failure = compensation.error || headcount.error
  const inputValid = type === 'compensation' ? Number.isFinite(adjustmentValue) && adjustmentValue >= -100 && adjustmentValue <= 100 : Number.isInteger(changeCount) && changeCount >= 1 && changeCount <= 100000
  const run = () => { compensation.reset(); headcount.reset(); setResult(null); if (type === 'compensation') compensation.mutate(); else headcount.mutate() }

  return <Page>
    <PageHeader eyebrow="Plan" title="What if we changed something?" description="Explore the financial and workforce implications of a pay or hiring assumption before taking it into a real decision." />

    {failure && <StateSummary title="This scenario could not be calculated" description={failure instanceof Error ? failure.message : 'Review the inputs and try again.'} tone="warning" />}
    {staleResult && <StateSummary title="Your data changed" description="The previous scenario has been hidden. Run it again using the current workforce data." tone="warning" />}

    <div className="grid gap-6 xl:grid-cols-[340px_minmax(0,1fr)]">
      <Surface padding="lg">
        <fieldset disabled={loading}>
          <SectionHeader title="Build a scenario" description="Choose what you want to explore." />
          <div className="mt-5 grid grid-cols-2 gap-2">
            <Button size="sm" variant={type === 'compensation' ? 'primary' : 'secondary'} onClick={() => { setResult(null); setType('compensation') }}><DollarSign className="h-4 w-4" />Change pay</Button>
            <Button size="sm" variant={type === 'headcount' ? 'primary' : 'secondary'} onClick={() => { setResult(null); setType('headcount') }}><Users className="h-4 w-4" />Add people</Button>
          </div>

          <div className="mt-6 space-y-5">
            {type === 'compensation' && <Input label="Pay change (%)" type="number" min={-100} max={100} step="0.1" value={adjustmentValue} onChange={event => { setResult(null); setAdjustmentValue(Number(event.target.value)) }} error={Number.isFinite(adjustmentValue) && adjustmentValue >= -100 && adjustmentValue <= 100 ? undefined : 'Enter a value from -100% to 100%.'} helperText="Use a positive value for an increase and a negative value for a decrease." />}
            {type === 'headcount' && <Input label="Additional positions" type="number" min={1} max={100000} step="1" value={changeCount} onChange={event => { setResult(null); setChangeCount(Number(event.target.value)) }} error={Number.isInteger(changeCount) && changeCount >= 1 && changeCount <= 100000 ? undefined : 'Enter a whole number from 1 to 100,000.'} helperText="PeopleOS models aggregate expansion only; it does not choose individuals for reductions." />}

            <div><label htmlFor="scenario-scope" className="mb-2 block text-sm font-medium">Who does this apply to?</label><select id="scenario-scope" value={targetScope} onChange={event => { setResult(null); setTargetScope(event.target.value as 'all' | 'department') }} className="h-10 w-full rounded-xl border border-border bg-surface px-3 text-sm focus:outline-none focus:ring-2 focus:ring-accent/30"><option value="all">Whole workforce</option><option value="department">One department</option></select></div>
            {targetScope === 'department' && <div><label htmlFor="scenario-department" className="mb-2 block text-sm font-medium">Department</label><select id="scenario-department" value={targetDept} onChange={event => { setResult(null); setTargetDept(event.target.value) }} className="h-10 w-full rounded-xl border border-border bg-surface px-3 text-sm focus:outline-none focus:ring-2 focus:ring-accent/30"><option value="">Choose a department</option>{departments.map(department => <option key={department} value={department}>{department}</option>)}</select></div>}
            <Button onClick={run} isLoading={loading} disabled={!inputValid || (targetScope === 'department' && !targetDept)} className="w-full"><Play className="h-4 w-4" />Explore scenario</Button>
          </div>
        </fieldset>
      </Surface>

      <div className="space-y-6">
        {!result || !result.available ? <Surface padding="lg" className="grid min-h-[360px] place-items-center"><div className="max-w-lg text-center"><div className="mx-auto grid h-12 w-12 place-items-center rounded-2xl bg-accent/10 text-accent"><GitBranch className="h-6 w-6" /></div><h2 className="mt-5 text-xl font-semibold">Choose a scenario to explore</h2><p className="mt-2 text-sm leading-6 text-text-secondary">PeopleOS will show the modeled cost and workforce effect, then keep the assumptions available for review.</p></div></Surface> : <>
          <p className="text-sm text-text-secondary">Using {result.provenance?.source_name ?? 'your active workforce data'}{result.provenance?.dataset_version ? ` · version ${result.provenance.dataset_version}` : ''}</p>
          <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <MetricCard label="People in scope" value={result.affected_employees.toLocaleString()} detail={result.affected_departments.length ? result.affected_departments.join(', ') : 'Configured scope'} icon={Users} />
            <MetricCard label="Starting assumption" value={`${result.baseline_turnover_rate.toFixed(1)}%`} detail="Baseline used in the scenario" icon={Target} />
            <MetricCard label="Modeled outcome" value={`${result.projected_turnover_rate.toFixed(1)}%`} detail="Result under the current assumptions" icon={BarChart3} />
            <MetricCard label="Modeled net impact" value={money(result.cost_impact.net_impact)} detail={result.roi_estimate == null ? 'ROI unavailable' : `Modeled ROI ${result.roi_estimate.toFixed(1)}%`} icon={DollarSign} tone={result.cost_impact.net_impact >= 0 ? 'success' : 'warning'} />
          </section>

          <div className="grid gap-6 lg:grid-cols-[minmax(0,1.1fr)_minmax(300px,0.9fr)]">
            <Surface padding="lg"><SectionHeader title="What the scenario says" description="A concise interpretation of the modeled result." /><div className="mt-5"><StatusBadge tone="info">Exploratory</StatusBadge><p className="mt-4 text-sm leading-7 text-text-secondary">{result.recommendation}</p><div className="mt-5 grid grid-cols-2 gap-3"><div className="rounded-xl border border-border p-4"><div className="text-xl font-semibold">{money(result.cost_impact.total_cost)}</div><div className="text-xs text-text-muted">modeled cost</div></div><div className="rounded-xl border border-border p-4"><div className="text-xl font-semibold">{money(result.cost_impact.total_benefit)}</div><div className="text-xs text-text-muted">modeled benefit</div></div></div></div></Surface>
            <Surface padding="lg"><SectionHeader title="Things to pressure-test" description="What a People or Finance owner should validate before using this scenario." /><div className="mt-5 space-y-3">{result.risks.slice(0, 4).map(item => <div key={item} className="rounded-xl border border-border p-3 text-sm leading-6 text-text-secondary">{item}</div>)}</div></Surface>
          </div>

          <TrustDisclosure title="How this scenario was modeled" summary={`${result.simulation.n_iterations.toLocaleString()} assumption draws`}>
            <p>Scenario results are deterministic calculations over configured assumptions and your current workforce data. They are not causal forecasts or guarantees.</p><div className="mt-3"><div className="font-semibold text-slate-700 dark:text-slate-200">Assumptions used</div><ul className="mt-1 space-y-1">{result.assumptions.slice(0, 8).map(item => <li key={item}>• {item}</li>)}</ul></div><p className="mt-3">{(result.simulation.roi_positive_probability * 100).toFixed(0)}% of configured simulation draws produced positive modeled ROI. This is not an empirical probability that the investment will succeed.</p>
          </TrustDisclosure>
        </>}
      </div>
    </div>
  </Page>
}
