'use client'

import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { Card } from '../../components/ui/card'
import { Badge } from '../../components/ui/badge'
import { api } from '../../lib/api-client'
import { cn } from '../../lib/utils'
import {
  GitBranch,
  DollarSign,
  Users,
  Heart,
  Play,
  BarChart3,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  RefreshCw,
  ChevronRight,
} from 'lucide-react'
import { ExplanationBox } from '../../components/ui/metric-explainer'

type ScenarioType = 'compensation' | 'headcount' | 'intervention'

interface ScenarioResult {
  available: boolean
  scenario_id: string
  scenario_name: string
  scenario_type: string
  affected_employees: number
  affected_departments: string[]
  baseline_turnover_rate: number
  projected_turnover_rate: number
  turnover_change: number
  turnover_change_pct: number
  simulation: {
    n_iterations: number
    outcome_mean: number
    outcome_std: number
    roi_mean: number
    roi_positive_probability: number
  }
  cost_impact: {
    salary_change: number
    total_cost: number
    total_benefit: number
    net_impact: number
  }
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

export default function ScenarioPlannerPage() {
  const [scenarioType, setScenarioType] = useState<ScenarioType>('compensation')
  const [result, setResult] = useState<ScenarioResult | null>(null)

  // Compensation form state
  const [adjustmentType, setAdjustmentType] = useState<'percentage' | 'absolute'>('percentage')
  const [adjustmentValue, setAdjustmentValue] = useState(5)
  const [targetScope, setTargetScope] = useState<'all' | 'department'>('all')
  const [targetDept, setTargetDept] = useState('')

  // Headcount form state
  const [changeType, setChangeType] = useState<'reduction' | 'expansion'>('reduction')
  const [changeCount, setChangeCount] = useState(10)
  const [selectionCriteria, setSelectionCriteria] = useState<'performance' | 'tenure' | 'cost'>('performance')

  // Intervention form state
  const [interventionType, setInterventionType] = useState<'retention_bonus' | 'career_path' | 'manager_change'>('retention_bonus')
  const [targetEmployees, setTargetEmployees] = useState<'high_risk' | 'high_risk_high_performer'>('high_risk')
  const [bonusPercentage, setBonusPercentage] = useState(10)

  // Fetch departments for dropdown
  const { data: departmentsData } = useQuery({
    queryKey: ['analytics', 'departments'],
    queryFn: () => api.analytics.getDepartments(),
  })
  const departments: string[] = (departmentsData as { departments?: Array<{ dept: string }> })?.departments?.map(d => d.dept) || []

  const compensationMutation = useMutation({
    mutationFn: () => api.scenario.simulateCompensation({
      adjustment_type: adjustmentType,
      target: {
        scope: targetScope,
        department: targetScope === 'department' ? targetDept : undefined,
      },
      adjustment_value: adjustmentValue,
      time_horizon_months: 12,
    }),
    onSuccess: (data) => setResult(data as ScenarioResult),
  })

  const headcountMutation = useMutation({
    mutationFn: () => api.scenario.simulateHeadcount({
      change_type: changeType,
      target: { scope: targetScope, department: targetScope === 'department' ? targetDept : undefined },
      change_count: changeCount,
      selection_criteria: selectionCriteria,
    }),
    onSuccess: (data) => setResult(data as ScenarioResult),
  })

  const interventionMutation = useMutation({
    mutationFn: () => api.scenario.simulateIntervention({
      intervention_type: interventionType,
      target_employees: targetEmployees,
      intervention_params: interventionType === 'retention_bonus' ? { bonus_percentage: bonusPercentage } : {},
    }),
    onSuccess: (data) => setResult(data as ScenarioResult),
  })

  const isLoading = compensationMutation.isPending || headcountMutation.isPending || interventionMutation.isPending

  const runScenario = () => {
    setResult(null)
    if (scenarioType === 'compensation') {
      compensationMutation.mutate()
    } else if (scenarioType === 'headcount') {
      headcountMutation.mutate()
    } else {
      interventionMutation.mutate()
    }
  }

  const formatCurrency = (value: number) => {
    const abs = Math.abs(value)
    if (abs >= 1000000) return `$${(value / 1000000).toFixed(1)}M`
    if (abs >= 1000) return `$${(value / 1000).toFixed(0)}K`
    return `$${value.toFixed(0)}`
  }

  return (
    <div className="space-y-6">
      {/* Page Title */}
      <div>
        <h1 className="text-2xl font-bold text-text-primary dark:text-text-dark-primary">Scenario Planner</h1>
        <p className="text-text-secondary dark:text-text-dark-secondary mt-1">
          Model what-if scenarios and predict workforce outcomes
        </p>
      </div>

      {/* Monte Carlo Explanation */}
      <div className="bg-gradient-to-r from-accent/5 to-accent/10 border border-accent/20 rounded-xl p-6">
        <div className="flex items-start gap-4">
          <div className="p-3 bg-accent/10 rounded-xl flex-shrink-0">
            <BarChart3 className="w-6 h-6 text-accent" />
          </div>
          <div>
            <h3 className="font-bold text-lg mb-2">What is Monte Carlo Simulation?</h3>
            <p className="text-sm text-text-secondary mb-3">
              Monte Carlo simulation is a mathematical technique that predicts possible outcomes by running thousands of "what-if" scenarios with random variations. Instead of giving you a single prediction, it shows you the <strong>range of likely outcomes</strong> and their probabilities.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div className="bg-surface/50 rounded-lg p-3">
                <p className="font-semibold text-accent mb-1">1,000 Simulations</p>
                <p className="text-text-muted text-xs">Each scenario runs 1,000 iterations with random variations to capture uncertainty</p>
              </div>
              <div className="bg-surface/50 rounded-lg p-3">
                <p className="font-semibold text-accent mb-1">Probability Distribution</p>
                <p className="text-text-muted text-xs">See the full range of possible outcomes, not just best/worst case</p>
              </div>
              <div className="bg-surface/50 rounded-lg p-3">
                <p className="font-semibold text-accent mb-1">Confidence Levels</p>
                <p className="text-text-muted text-xs">Know how confident you can be in the predicted ROI</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Scenario Builder */}
        <div className="lg:col-span-1 space-y-4">
          <Card title="Build Scenario" subtitle="Configure your what-if scenario">
            {/* Scenario Type Selector */}
            <div className="p-4 border-b border-border">
              <label className="text-xs font-medium text-text-muted uppercase tracking-wide mb-2 block">
                Scenario Type
              </label>
              <div className="grid grid-cols-3 gap-2">
                <button
                  onClick={() => setScenarioType('compensation')}
                  className={cn(
                    "p-3 rounded-lg border text-center transition-all",
                    scenarioType === 'compensation'
                      ? 'border-accent bg-accent/10 text-accent'
                      : 'border-border hover:border-accent/50'
                  )}
                >
                  <DollarSign className="w-5 h-5 mx-auto mb-1" />
                  <span className="text-xs font-medium">Compensation</span>
                </button>
                <button
                  onClick={() => setScenarioType('headcount')}
                  className={cn(
                    "p-3 rounded-lg border text-center transition-all",
                    scenarioType === 'headcount'
                      ? 'border-accent bg-accent/10 text-accent'
                      : 'border-border hover:border-accent/50'
                  )}
                >
                  <Users className="w-5 h-5 mx-auto mb-1" />
                  <span className="text-xs font-medium">Headcount</span>
                </button>
                <button
                  onClick={() => setScenarioType('intervention')}
                  className={cn(
                    "p-3 rounded-lg border text-center transition-all",
                    scenarioType === 'intervention'
                      ? 'border-accent bg-accent/10 text-accent'
                      : 'border-border hover:border-accent/50'
                  )}
                >
                  <Heart className="w-5 h-5 mx-auto mb-1" />
                  <span className="text-xs font-medium">Retention</span>
                </button>
              </div>
            </div>

            {/* Scenario Parameters */}
            <div className="p-4 space-y-4">
              {scenarioType === 'compensation' && (
                <>
                  <div>
                    <label className="text-xs font-medium text-text-muted mb-1 block">Adjustment Type</label>
                    <select
                      value={adjustmentType}
                      onChange={(e) => setAdjustmentType(e.target.value as 'percentage' | 'absolute')}
                      className="w-full p-2 rounded-lg border border-border bg-surface text-sm"
                    >
                      <option value="percentage">Percentage Raise</option>
                      <option value="absolute">Absolute Amount</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-xs font-medium text-text-muted mb-1 block">
                      {adjustmentType === 'percentage' ? 'Raise Percentage (%)' : 'Amount ($)'}
                    </label>
                    <input
                      type="number"
                      value={adjustmentValue}
                      onChange={(e) => setAdjustmentValue(Number(e.target.value))}
                      className="w-full p-2 rounded-lg border border-border bg-surface text-sm"
                      min={0}
                      max={adjustmentType === 'percentage' ? 50 : 50000}
                    />
                  </div>
                  <div>
                    <label className="text-xs font-medium text-text-muted mb-1 block">Target</label>
                    <select
                      value={targetScope}
                      onChange={(e) => setTargetScope(e.target.value as 'all' | 'department')}
                      className="w-full p-2 rounded-lg border border-border bg-surface text-sm"
                    >
                      <option value="all">All Employees</option>
                      <option value="department">Specific Department</option>
                    </select>
                  </div>
                  {targetScope === 'department' && (
                    <div>
                      <label className="text-xs font-medium text-text-muted mb-1 block">Department</label>
                      <select
                        value={targetDept}
                        onChange={(e) => setTargetDept(e.target.value)}
                        className="w-full p-2 rounded-lg border border-border bg-surface text-sm"
                      >
                        <option value="">Select a department...</option>
                        {departments.map((dept) => (
                          <option key={dept} value={dept}>{dept}</option>
                        ))}
                      </select>
                    </div>
                  )}
                </>
              )}

              {scenarioType === 'headcount' && (
                <>
                  <div>
                    <label className="text-xs font-medium text-text-muted mb-1 block">Change Type</label>
                    <select
                      value={changeType}
                      onChange={(e) => setChangeType(e.target.value as 'reduction' | 'expansion')}
                      className="w-full p-2 rounded-lg border border-border bg-surface text-sm"
                    >
                      <option value="reduction">Workforce Reduction</option>
                      <option value="expansion">Workforce Expansion</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-xs font-medium text-text-muted mb-1 block">Number of Positions</label>
                    <input
                      type="number"
                      value={changeCount}
                      onChange={(e) => setChangeCount(Number(e.target.value))}
                      className="w-full p-2 rounded-lg border border-border bg-surface text-sm"
                      min={1}
                      max={100}
                    />
                  </div>
                  <div>
                    <label className="text-xs font-medium text-text-muted mb-1 block">Selection Criteria</label>
                    <select
                      value={selectionCriteria}
                      onChange={(e) => setSelectionCriteria(e.target.value as 'performance' | 'tenure' | 'cost')}
                      className="w-full p-2 rounded-lg border border-border bg-surface text-sm"
                    >
                      <option value="performance">By Performance</option>
                      <option value="tenure">By Tenure</option>
                      <option value="cost">By Cost (Salary)</option>
                    </select>
                  </div>
                </>
              )}

              {scenarioType === 'intervention' && (
                <>
                  <div>
                    <label className="text-xs font-medium text-text-muted mb-1 block">Intervention Type</label>
                    <select
                      value={interventionType}
                      onChange={(e) => setInterventionType(e.target.value as 'retention_bonus' | 'career_path' | 'manager_change')}
                      className="w-full p-2 rounded-lg border border-border bg-surface text-sm"
                    >
                      <option value="retention_bonus">Retention Bonus</option>
                      <option value="career_path">Career Development Program</option>
                      <option value="manager_change">Manager Reorg</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-xs font-medium text-text-muted mb-1 block">Target Employees</label>
                    <select
                      value={targetEmployees}
                      onChange={(e) => setTargetEmployees(e.target.value as 'high_risk' | 'high_risk_high_performer')}
                      className="w-full p-2 rounded-lg border border-border bg-surface text-sm"
                    >
                      <option value="high_risk">High Risk Employees</option>
                      <option value="high_risk_high_performer">High Risk + High Performers</option>
                    </select>
                  </div>
                  {interventionType === 'retention_bonus' && (
                    <div>
                      <label className="text-xs font-medium text-text-muted mb-1 block">Bonus Percentage (%)</label>
                      <input
                        type="number"
                        value={bonusPercentage}
                        onChange={(e) => setBonusPercentage(Number(e.target.value))}
                        className="w-full p-2 rounded-lg border border-border bg-surface text-sm"
                        min={1}
                        max={50}
                      />
                    </div>
                  )}
                </>
              )}

              <button
                onClick={runScenario}
                disabled={isLoading}
                className="w-full flex items-center justify-center gap-2 p-3 bg-accent text-white rounded-lg hover:bg-accent/90 transition-colors disabled:opacity-50"
              >
                {isLoading ? (
                  <RefreshCw className="w-4 h-4 animate-spin" />
                ) : (
                  <Play className="w-4 h-4" />
                )}
                {isLoading ? 'Running Simulation...' : 'Run Scenario'}
              </button>
            </div>
          </Card>
        </div>

        {/* Results Panel */}
        <div className="lg:col-span-2 space-y-4">
          {!result && !isLoading && (
            <Card className="p-12 text-center">
              <GitBranch className="w-12 h-12 mx-auto text-text-muted mb-4" />
              <h3 className="text-lg font-semibold mb-2">No Scenario Results Yet</h3>
              <p className="text-text-muted">
                Configure a scenario on the left and click "Run Scenario" to see predictions.
              </p>
            </Card>
          )}

          {isLoading && (
            <Card className="p-12 text-center">
              <RefreshCw className="w-12 h-12 mx-auto text-accent mb-4 animate-spin" />
              <h3 className="text-lg font-semibold mb-2">Running Monte Carlo Simulation</h3>
              <p className="text-text-muted">
                Simulating 1,000 possible outcomes...
              </p>
            </Card>
          )}

          {result && (
            <>
              {/* Result Summary */}
              <Card title={result.scenario_name} subtitle={`${result.affected_employees} employees affected`}>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-4">
                  <div className="p-4 rounded-lg bg-surface-secondary text-center">
                    <p className="text-2xl font-bold">{result.baseline_turnover_rate}%</p>
                    <p className="text-xs text-text-muted">Baseline Turnover</p>
                  </div>
                  <div className="p-4 rounded-lg bg-surface-secondary text-center">
                    <p className={cn(
                      "text-2xl font-bold",
                      result.projected_turnover_rate < result.baseline_turnover_rate ? 'text-success' : 'text-danger'
                    )}>
                      {result.projected_turnover_rate}%
                    </p>
                    <p className="text-xs text-text-muted">Projected Turnover</p>
                  </div>
                  <div className="p-4 rounded-lg bg-surface-secondary text-center">
                    <p className={cn(
                      "text-2xl font-bold flex items-center justify-center gap-1",
                      result.turnover_change > 0 ? 'text-success' : 'text-danger'
                    )}>
                      {result.turnover_change > 0 ? <TrendingDown className="w-5 h-5" /> : <TrendingUp className="w-5 h-5" />}
                      {Math.abs(result.turnover_change)}pp
                    </p>
                    <p className="text-xs text-text-muted">Turnover Change</p>
                  </div>
                  <div className="p-4 rounded-lg bg-surface-secondary text-center">
                    <Badge variant={
                      result.confidence_level === 'High' ? 'success' :
                      result.confidence_level === 'Medium' ? 'warning' : 'outline'
                    }>
                      {result.confidence_level} Confidence
                    </Badge>
                  </div>
                </div>
              </Card>

              {/* Financial Impact */}
              <Card title="Financial Impact" subtitle="Cost-benefit analysis">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-4">
                  <div className="p-4 rounded-lg border border-border">
                    <p className="text-xs text-text-muted mb-1">Total Cost</p>
                    <p className="text-xl font-bold text-danger">
                      {formatCurrency(result.cost_impact.total_cost)}
                    </p>
                  </div>
                  <div className="p-4 rounded-lg border border-border">
                    <p className="text-xs text-text-muted mb-1">Total Benefit</p>
                    <p className="text-xl font-bold text-success">
                      {formatCurrency(result.cost_impact.total_benefit)}
                    </p>
                  </div>
                  <div className="p-4 rounded-lg border border-border">
                    <p className="text-xs text-text-muted mb-1">Net Impact</p>
                    <p className={cn(
                      "text-xl font-bold",
                      result.cost_impact.net_impact >= 0 ? 'text-success' : 'text-danger'
                    )}>
                      {formatCurrency(result.cost_impact.net_impact)}
                    </p>
                  </div>
                  <div className="p-4 rounded-lg border border-border">
                    <p className="text-xs text-text-muted mb-1">Expected ROI</p>
                    <p className={cn(
                      "text-xl font-bold",
                      (result.roi_estimate || 0) >= 0 ? 'text-success' : 'text-danger'
                    )}>
                      {result.roi_estimate !== null ? `${result.roi_estimate}%` : 'N/A'}
                    </p>
                  </div>
                </div>
              </Card>

              {/* Monte Carlo Results */}
              <Card title="Simulation Results" subtitle={`Based on ${result.simulation.n_iterations} Monte Carlo iterations`}>
                <div className="p-4 space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-3 rounded-lg bg-surface-secondary">
                      <p className="text-xs text-text-muted mb-1">Mean Outcome</p>
                      <p className="font-bold">{(result.simulation.outcome_mean * 100).toFixed(1)}%</p>
                    </div>
                    <div className="p-3 rounded-lg bg-surface-secondary">
                      <p className="text-xs text-text-muted mb-1">Std Deviation</p>
                      <p className="font-bold">{(result.simulation.outcome_std * 100).toFixed(1)}%</p>
                    </div>
                    <div className="p-3 rounded-lg bg-surface-secondary">
                      <p className="text-xs text-text-muted mb-1">Mean ROI</p>
                      <p className="font-bold">{result.simulation.roi_mean.toFixed(1)}%</p>
                    </div>
                    <div className="p-3 rounded-lg bg-surface-secondary">
                      <p className="text-xs text-text-muted mb-1">Probability of Positive ROI</p>
                      <p className={cn(
                        "font-bold",
                        result.simulation.roi_positive_probability >= 0.7 ? 'text-success' :
                        result.simulation.roi_positive_probability >= 0.5 ? 'text-warning' : 'text-danger'
                      )}>
                        {(result.simulation.roi_positive_probability * 100).toFixed(0)}%
                      </p>
                    </div>
                  </div>
                </div>
              </Card>

              {/* Recommendation */}
              <Card>
                <div className="p-4">
                  <div className="flex items-start gap-3 mb-4">
                    {(result.roi_estimate || 0) > 0 ? (
                      <CheckCircle className="w-6 h-6 text-success flex-shrink-0" />
                    ) : (
                      <AlertTriangle className="w-6 h-6 text-warning flex-shrink-0" />
                    )}
                    <div>
                      <p className="font-bold">Recommendation</p>
                      <p className="text-text-secondary">{result.recommendation}</p>
                    </div>
                  </div>

                  {result.risks.length > 0 && (
                    <div className="mb-4">
                      <p className="text-xs font-medium text-text-muted uppercase tracking-wide mb-2">Risks</p>
                      <ul className="space-y-1">
                        {result.risks.map((risk, i) => (
                          <li key={i} className="flex items-start gap-2 text-sm text-text-secondary">
                            <AlertTriangle className="w-4 h-4 text-warning mt-0.5 flex-shrink-0" />
                            {risk}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {result.alternative_actions.length > 0 && (
                    <div>
                      <p className="text-xs font-medium text-text-muted uppercase tracking-wide mb-2">Alternatives to Consider</p>
                      <ul className="space-y-1">
                        {result.alternative_actions.map((action, i) => (
                          <li key={i} className="flex items-start gap-2 text-sm text-text-secondary">
                            <ChevronRight className="w-4 h-4 text-accent mt-0.5 flex-shrink-0" />
                            {action}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </Card>

              {/* Data Sources - Important for users to know what predictions are based on */}
              {result.data_sources && result.data_sources.length > 0 && (
                <Card title="Data Sources" subtitle="What this prediction is based on">
                  <div className="p-4">
                    <div className={cn(
                      "p-4 rounded-lg border mb-4",
                      result.data_sources.some(s => s.includes('ML model') || s.includes('Historical attrition') || s.includes('Survival'))
                        ? "border-success/30 bg-success/5"
                        : "border-warning/30 bg-warning/5"
                    )}>
                      <div className="flex items-start gap-3">
                        {result.data_sources.some(s => s.includes('ML model') || s.includes('Historical attrition') || s.includes('Survival')) ? (
                          <CheckCircle className="w-5 h-5 text-success flex-shrink-0 mt-0.5" />
                        ) : (
                          <AlertTriangle className="w-5 h-5 text-warning flex-shrink-0 mt-0.5" />
                        )}
                        <div>
                          <p className="font-semibold text-sm mb-1">
                            {result.data_sources.some(s => s.includes('ML model') || s.includes('Historical attrition') || s.includes('Survival'))
                              ? "Data-Driven Prediction"
                              : "Using Industry Estimates"}
                          </p>
                          <p className="text-xs text-text-muted">
                            {result.data_sources.some(s => s.includes('ML model') || s.includes('Historical attrition') || s.includes('Survival'))
                              ? "This prediction is based on patterns learned from your actual employee data."
                              : "For more accurate predictions, upload data with an Attrition column to enable data-driven analysis."}
                          </p>
                        </div>
                      </div>
                    </div>
                    <ul className="space-y-2">
                      {result.data_sources.map((source, i) => (
                        <li key={i} className="flex items-center gap-2 text-sm text-text-secondary">
                          <div className="w-2 h-2 rounded-full bg-accent flex-shrink-0" />
                          {source}
                        </li>
                      ))}
                    </ul>
                  </div>
                </Card>
              )}

              {/* Assumptions */}
              {result.assumptions.length > 0 && (
                <Card title="Assumptions" subtitle="Key assumptions used in this simulation">
                  <ul className="divide-y divide-border">
                    {result.assumptions.map((assumption, i) => (
                      <li key={i} className="px-4 py-2 text-sm text-text-secondary">
                        {assumption}
                      </li>
                    ))}
                  </ul>
                </Card>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
