'use client'

import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api-client'
import { Button, Input, MetricCard, Page, PageHeader, SectionHeader, StateSummary, StatusBadge, Surface } from '@/components/ui'
import { BarChart3, DollarSign, GitBranch, Play, Target, Users } from 'lucide-react'

type ScenarioType = 'compensation' | 'headcount'

interface ScenarioResult {
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
  const abs = Math.abs(value)
  if (abs >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`
  if (abs >= 1_000) return `${(value / 1_000).toFixed(0)}K`
  return `${Math.round(value).toLocaleString()}`
}

export default function ScenarioPlannerPage() {
  const [type, setType] = useState<ScenarioType>('compensation')
  const [result, setResult] = useState<ScenarioResult | null>(null)
  const [adjustmentValue, setAdjustmentValue] = useState(5)
  const [targetScope, setTargetScope] = useState<'all' | 'department'>('all')
  const [targetDept, setTargetDept] = useState('')
  const [changeCount, setChangeCount] = useState(10)

  const { data: departmentsData } = useQuery({ queryKey: ['analytics', 'departments'], queryFn: () => api.analytics.getDepartments() })
  const departments: string[] = (departmentsData as { departments?: Array<{ dept: string }> })?.departments?.map((item) => item.dept) ?? []

  const compensation = useMutation({
    mutationFn: () => api.scenario.simulateCompensation({
      adjustment_type: 'percentage',
      target: { scope: targetScope, department: targetScope === 'department' ? targetDept : undefined },
      adjustment_value: adjustmentValue,
      time_horizon_months: 12,
    }),
    onSuccess: (data) => setResult(data as ScenarioResult),
  })

  const headcount = useMutation({
    mutationFn: () => api.scenario.simulateHeadcount({
      change_type: 'expansion',
      target: { scope: targetScope, department: targetScope === 'department' ? targetDept : undefined },
      change_count: changeCount,
      selection_criteria: 'performance',
    }),
    onSuccess: (data) => setResult(data as ScenarioResult),
  })

  const loading = compensation.isPending || headcount.isPending
  const failure = compensation.error || headcount.error
  const run = () => {
    compensation.reset()
    headcount.reset()
    setResult(null)
    if (type === 'compensation') compensation.mutate()
    else headcount.mutate()
  }

  return (
    <Page>
      <PageHeader eyebrow="Plan · Scenario Planner" title="Explore assumptions before making workforce decisions" description="Compare aggregate what-if cases with explicit assumptions and costs. Scenario outputs are exploratory sensitivity models, not causal forecasts or authorization to act." />
      <StateSummary title="Assumption sensitivity, not prediction certainty" description="Simulation frequencies describe the configured model, not the empirical probability that an outcome will happen." tone="info" />

      {failure && <StateSummary title="Scenario unavailable" description={failure instanceof Error ? failure.message : 'The scenario could not be calculated.'} tone="warning" />}
      <p className="text-sm text-text-muted">Financial values use source salary units; annual salary is required.</p>
      <div className="grid gap-6 xl:grid-cols-[360px_minmax(0,1fr)]">
        <Surface padding="lg">
          <SectionHeader title="Configure scenario" description="Choose a governed aggregate decision class." />
          <div className="mt-5 grid grid-cols-2 gap-2">
            <Button size="sm" variant={type === 'compensation' ? 'primary' : 'secondary'} onClick={() => setType('compensation')}><DollarSign className="h-4 w-4" />Pay assumption</Button>
            <Button size="sm" variant={type === 'headcount' ? 'primary' : 'secondary'} onClick={() => setType('headcount')}><Users className="h-4 w-4" />Expansion</Button>
          </div>

          <div className="mt-6 space-y-5">
            {type === 'compensation' && <Input label="Compensation adjustment (%)" type="number" value={adjustmentValue} onChange={(event) => setAdjustmentValue(Number(event.target.value))} helperText="Explores a pay-change assumption; it does not estimate a causal retention effect." />}
            {type === 'headcount' && <Input label="Additional positions" type="number" value={changeCount} onChange={(event) => setChangeCount(Number(event.target.value))} helperText="Aggregate expansion only. PeopleOS does not rank employees for reduction decisions." />}

            <div><label className="mb-2 block text-sm font-medium">Scope</label><select value={targetScope} onChange={(event) => setTargetScope(event.target.value as 'all' | 'department')} className="h-10 w-full rounded-xl border border-border bg-surface px-3 text-sm focus:outline-none focus:ring-2 focus:ring-accent/30"><option value="all">Whole workforce</option><option value="department">One department</option></select></div>
            {targetScope === 'department' && <div><label className="mb-2 block text-sm font-medium">Department</label><select value={targetDept} onChange={(event) => setTargetDept(event.target.value)} className="h-10 w-full rounded-xl border border-border bg-surface px-3 text-sm focus:outline-none focus:ring-2 focus:ring-accent/30"><option value="">Select department</option>{departments.map((department) => <option key={department} value={department}>{department}</option>)}</select></div>}

            <Button onClick={run} isLoading={loading} disabled={targetScope === 'department' && !targetDept} className="w-full"><Play className="h-4 w-4" />Run exploratory scenario</Button>
          </div>
        </Surface>

        <div className="space-y-6">
          {!result ? <Surface padding="lg" className="min-h-[360px] grid place-items-center"><div className="max-w-lg text-center"><div className="mx-auto grid h-12 w-12 place-items-center rounded-2xl bg-accent/10 text-accent"><GitBranch className="h-6 w-6" /></div><h2 className="mt-5 text-xl font-semibold">Configure a scenario to inspect sensitivity</h2><p className="mt-2 text-sm leading-6 text-text-secondary">Model outcomes and costs while keeping the assumptions available for review.</p></div></Surface> : <>
            <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
              <MetricCard label="People in scope" value={result.affected_employees.toLocaleString()} detail={result.affected_departments.length ? result.affected_departments.join(', ') : 'Configured scope'} icon={Users} />
              <MetricCard label="Assumed baseline rate" value={`${result.baseline_turnover_rate.toFixed(1)}%`} detail="Baseline used by the scenario" icon={Target} />
              <MetricCard label="Modeled scenario rate" value={`${result.projected_turnover_rate.toFixed(1)}%`} detail="Assumption-based output" icon={BarChart3} />
              <MetricCard label="Modeled net impact" value={money(result.cost_impact.net_impact)} detail={result.roi_estimate == null ? 'ROI assumption unavailable' : `Modeled ROI ${result.roi_estimate.toFixed(1)}%`} icon={DollarSign} tone={result.cost_impact.net_impact >= 0 ? 'success' : 'warning'} />
            </section>

            <div className="grid gap-6 lg:grid-cols-2">
              <Surface padding="lg"><SectionHeader title="Scenario interpretation" description="A concise read of the modeled result." /><div className="mt-5"><StatusBadge tone="info">{result.confidence_level} evidence</StatusBadge><p className="mt-4 text-sm leading-7 text-text-secondary">{result.recommendation}</p><div className="mt-5 grid grid-cols-2 gap-3"><div className="rounded-xl border border-border p-4"><div className="text-xl font-semibold">{result.simulation.n_iterations.toLocaleString()}</div><div className="text-xs text-text-muted">assumption draws</div></div><div className="rounded-xl border border-border p-4"><div className="text-xl font-semibold">{(result.simulation.roi_positive_probability * 100).toFixed(0)}%</div><div className="text-xs text-text-muted">draws with positive modeled ROI</div></div></div></div></Surface>
              <Surface padding="lg"><SectionHeader title="Assumptions & risks" description="Review these when you need to interrogate the model." /><div className="mt-5 space-y-4"><div><div className="text-xs font-semibold uppercase tracking-wider text-text-muted">Assumptions</div><ul className="mt-2 space-y-2 text-sm leading-6 text-text-secondary">{result.assumptions.slice(0, 8).map((item) => <li key={item}>• {item}</li>)}</ul></div><div className="border-t border-border pt-4"><div className="text-xs font-semibold uppercase tracking-wider text-text-muted">Risks</div><ul className="mt-2 space-y-2 text-sm leading-6 text-text-secondary">{result.risks.slice(0, 5).map((item) => <li key={item}>• {item}</li>)}</ul></div></div></Surface>
            </div>
          </>}
        </div>
      </div>

      <StateSummary title="Use boundary" description="Scenario output supports planning discussions only; PeopleOS does not select employees for reductions or target individuals for retention interventions." tone="info" />
    </Page>
  )
}
