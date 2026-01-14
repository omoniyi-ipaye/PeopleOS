'use client'

import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { GlassCard } from '@/components/ui/glass-card'
import { Badge } from '@/components/ui/badge'
import { api } from '@/lib/api-client'
import { cn } from '@/lib/utils'
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
  Calculator,
  Save,
  History,
  Info,
  Target
} from 'lucide-react'

// Types
type ScenarioType = 'compensation' | 'headcount' | 'intervention'
type TabType = 'simulator' | 'saved'

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
  const [activeTab, setActiveTab] = useState<TabType>('simulator')
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
    <div className="space-y-6 h-[calc(100vh-100px)] flex flex-col animate-in fade-in duration-700 slide-in-from-bottom-4">
      {/* Header */}
      <div className="flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between flex-shrink-0">
        <div>
          <h1 className="text-4xl font-display font-bold text-gradient bg-clip-text text-transparent bg-gradient-to-r from-gray-900 to-gray-600 dark:from-white dark:to-gray-400">
            Scenario Planner
          </h1>
          <p className="text-text-secondary dark:text-text-dark-secondary mt-2 text-lg font-light flex items-center gap-2">
            Monte Carlo Simulation Engine
          </p>
        </div>

        {/* Tab Controls */}
        <div className="glass p-1.5 rounded-2xl flex gap-1">
          <button
            onClick={() => setActiveTab('simulator')}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-medium transition-all duration-300 ${activeTab === 'simulator'
              ? 'bg-white dark:bg-slate-800 shadow-lg text-text-primary dark:text-white scale-105'
              : 'text-text-secondary dark:text-slate-400 hover:text-text-primary dark:hover:text-white hover:bg-white/10'
              }`}
          >
            <Calculator className="w-4 h-4" />
            Simulator
          </button>
          <button
            onClick={() => setActiveTab('saved')}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-medium transition-all duration-300 ${activeTab === 'saved'
              ? 'bg-white dark:bg-slate-800 shadow-lg text-text-primary dark:text-white scale-105'
              : 'text-text-secondary dark:text-slate-400 hover:text-text-primary dark:hover:text-white hover:bg-white/10'
              }`}
          >
            <History className="w-4 h-4" />
            Saved Scenarios
          </button>
        </div>
      </div>

      {activeTab === 'simulator' ? (
        <div className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-12 gap-6 overflow-hidden">
          {/* Controls Panel (Left - Scrollable) */}
          <div className="lg:col-span-4 overflow-y-auto pr-2 space-y-6 pb-6">

            {/* Methodology Note - RESTORED PROMINTENTLY */}
            <div className="bg-gradient-to-r from-accent/5 to-accent/10 border border-accent/20 rounded-xl p-6">
              <div className="flex items-start gap-4">
                <div className="p-3 bg-accent/10 rounded-xl flex-shrink-0">
                  <BarChart3 className="w-6 h-6 text-accent" />
                </div>
                <div>
                  <h3 className="font-bold text-lg mb-2">What is Monte Carlo Simulation?</h3>
                  <p className="text-sm text-text-secondary mb-3">
                    Monte Carlo simulation is a mathematical technique that predicts possible outcomes by running thousands of "what-if" scenarios with random variations.
                  </p>
                </div>
              </div>
            </div>

            <GlassCard className="p-0 overflow-hidden relative group">
              {/* Background Glow */}
              <div className="absolute top-0 right-0 w-64 h-64 bg-accent/5 rounded-full blur-3xl -z-10 group-hover:bg-accent/10 transition-colors"></div>

              <div className="p-6 border-b border-white/5">
                <h2 className="text-lg font-bold flex items-center gap-2">
                  <GitBranch className="w-5 h-5 text-accent" />
                  Configure Parameters
                </h2>
                <p className="text-xs text-text-muted mt-1">Define the variables for your simulation.</p>
              </div>

              <div className="p-6 space-y-6">
                {/* Type Selector */}
                <div className="grid grid-cols-3 gap-2">
                  {[
                    { id: 'compensation', icon: DollarSign, label: 'Comp' },
                    { id: 'headcount', icon: Users, label: 'Headcount' },
                    { id: 'intervention', icon: Heart, label: 'Retention' }
                  ].map((type) => (
                    <button
                      key={type.id}
                      onClick={() => setScenarioType(type.id as ScenarioType)}
                      className={cn(
                        "flex flex-col items-center justify-center p-3 rounded-xl border transition-all duration-300",
                        scenarioType === type.id
                          ? "border-accent bg-accent/10 text-accent shadow-sm"
                          : "border-white/5 bg-white/5 text-text-secondary hover:bg-white/10"
                      )}
                    >
                      <type.icon className="w-5 h-5 mb-1" />
                      <span className="text-xs font-medium">{type.label}</span>
                    </button>
                  ))}
                </div>

                {/* Dynamic Inputs */}
                <div className="space-y-6 animate-in fade-in slide-in-from-left-2 duration-300">
                  {scenarioType === 'compensation' && (
                    <>
                      {/* Adjustment Type Section */}
                      <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-white/50 dark:bg-slate-900/50 space-y-3">
                        <label className="text-xs font-bold text-text-muted uppercase tracking-wider flex items-center gap-2">
                          <GitBranch className="w-3 h-3" /> Adjustment Type
                        </label>
                        <div className="flex bg-slate-100 dark:bg-slate-800 p-1 rounded-lg border border-slate-200 dark:border-slate-700">
                          <button
                            onClick={() => setAdjustmentType('percentage')}
                            className={cn(
                              "flex-1 py-2 text-xs font-bold rounded-md transition-all",
                              adjustmentType === 'percentage'
                                ? 'bg-white dark:bg-slate-700 text-accent shadow-sm border border-slate-200 dark:border-slate-600'
                                : 'text-text-secondary hover:text-text-primary hover:bg-black/5 dark:hover:bg-white/5'
                            )}
                          >
                            Percentage
                          </button>
                          <button
                            onClick={() => setAdjustmentType('absolute')}
                            className={cn(
                              "flex-1 py-2 text-xs font-bold rounded-md transition-all",
                              adjustmentType === 'absolute'
                                ? 'bg-white dark:bg-slate-700 text-accent shadow-sm border border-slate-200 dark:border-slate-600'
                                : 'text-text-secondary hover:text-text-primary hover:bg-black/5 dark:hover:bg-white/5'
                            )}
                          >
                            Absolute Amount
                          </button>
                        </div>
                      </div>

                      {/* Value Input Section */}
                      <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-white/50 dark:bg-slate-900/50 space-y-4">
                        <label className="text-xs font-bold text-text-muted uppercase tracking-wider flex items-center gap-2">
                          <DollarSign className="w-3 h-3" />
                          {adjustmentType === 'percentage' ? 'Raise Percentage (%)' : 'Amount ($)'}
                        </label>
                        <div className="flex items-center gap-4">
                          <input
                            type="number"
                            value={adjustmentValue}
                            onChange={(e) => setAdjustmentValue(Number(e.target.value))}
                            className="w-24 bg-white dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-lg px-3 py-2 text-sm font-bold text-center focus:outline-none focus:border-accent focus:ring-1 focus:ring-accent transition-all shadow-sm text-text-primary dark:text-white"
                          />
                          <input
                            type="range"
                            min="0"
                            max={adjustmentType === 'percentage' ? 50 : 50000}
                            value={adjustmentValue}
                            onChange={(e) => setAdjustmentValue(Number(e.target.value))}
                            className="flex-1 accent-accent h-1.5 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer"
                          />
                        </div>
                      </div>
                    </>
                  )}

                  {scenarioType === 'headcount' && (
                    <>
                      {/* Action Section */}
                      <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-white/50 dark:bg-slate-900/50 space-y-3">
                        <label className="text-xs font-bold text-text-muted uppercase tracking-wider flex items-center gap-2">
                          <GitBranch className="w-3 h-3" /> Action
                        </label>
                        <div className="flex bg-slate-100 dark:bg-slate-800 p-1 rounded-lg border border-slate-200 dark:border-slate-700">
                          <button
                            onClick={() => setChangeType('expansion')}
                            className={cn(
                              "flex-1 py-2 text-xs font-bold rounded-md transition-all flex items-center justify-center gap-2",
                              changeType === 'expansion'
                                ? 'bg-emerald-500 text-white shadow-sm ring-1 ring-emerald-600'
                                : 'text-text-secondary hover:text-text-primary hover:bg-black/5 dark:hover:bg-white/5'
                            )}
                          >
                            Hire
                          </button>
                          <button
                            onClick={() => setChangeType('reduction')}
                            className={cn(
                              "flex-1 py-2 text-xs font-bold rounded-md transition-all flex items-center justify-center gap-2",
                              changeType === 'reduction'
                                ? 'bg-red-500 text-white shadow-sm ring-1 ring-red-600'
                                : 'text-text-secondary hover:text-text-primary hover:bg-black/5 dark:hover:bg-white/5'
                            )}
                          >
                            Reduce
                          </button>
                        </div>
                      </div>

                      {/* Count Section */}
                      <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-white/50 dark:bg-slate-900/50 space-y-4">
                        <label className="text-xs font-bold text-text-muted uppercase tracking-wider flex items-center gap-2">
                          <Users className="w-3 h-3" /> Count
                        </label>
                        <div className="flex items-center gap-4">
                          <div className="w-20 h-10 flex items-center justify-center bg-white dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-lg shadow-sm">
                            <span className="text-xl font-bold font-display">{changeCount}</span>
                          </div>
                          <input
                            type="range"
                            min="1"
                            max="100"
                            value={changeCount}
                            onChange={(e) => setChangeCount(Number(e.target.value))}
                            className="flex-1 accent-accent h-1.5 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer"
                          />
                        </div>
                      </div>

                      {/* Criteria Section */}
                      <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-white/50 dark:bg-slate-900/50 space-y-3">
                        <label className="text-xs font-bold text-text-muted uppercase tracking-wider flex items-center gap-2">
                          <BarChart3 className="w-3 h-3" /> Selection Criteria
                        </label>
                        <select
                          value={selectionCriteria}
                          onChange={(e) => setSelectionCriteria(e.target.value as any)}
                          className="w-full bg-white dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-lg px-3 py-2.5 text-sm font-medium focus:outline-none focus:border-accent focus:ring-1 focus:ring-accent transition-all shadow-sm text-text-primary dark:text-white"
                        >
                          <option value="performance">Performance Based</option>
                          <option value="tenure">Tenure Based</option>
                          <option value="cost">Cost Based</option>
                        </select>
                      </div>
                    </>
                  )}

                  {scenarioType === 'intervention' && (
                    <>
                      {/* Strategy Section */}
                      <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-white/50 dark:bg-slate-900/50 space-y-3">
                        <label className="text-xs font-bold text-text-muted uppercase tracking-wider flex items-center gap-2">
                          <Heart className="w-3 h-3" /> Strategy
                        </label>
                        <select
                          value={interventionType}
                          onChange={(e) => setInterventionType(e.target.value as any)}
                          className="w-full bg-white dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-lg px-3 py-2.5 text-sm font-medium focus:outline-none focus:border-accent focus:ring-1 focus:ring-accent transition-all shadow-sm text-text-primary dark:text-white"
                        >
                          <option value="retention_bonus">Retention Bonus</option>
                          <option value="career_path">Career Pathing</option>
                          <option value="manager_change">Internal Transfer</option>
                        </select>
                      </div>

                      {interventionType === 'retention_bonus' && (
                        <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-white/50 dark:bg-slate-900/50 space-y-4">
                          <label className="text-xs font-bold text-text-muted uppercase tracking-wider flex items-center gap-2">
                            <DollarSign className="w-3 h-3" /> Bonus Percentage
                          </label>
                          <div className="flex items-center gap-4">
                            <div className="w-20 h-10 flex items-center justify-center bg-white dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-lg shadow-sm">
                              <span className="text-xl font-bold font-display">{bonusPercentage}%</span>
                            </div>
                            <input
                              type="range"
                              min="1"
                              max="50"
                              value={bonusPercentage}
                              onChange={(e) => setBonusPercentage(Number(e.target.value))}
                              className="flex-1 accent-accent h-1.5 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer"
                            />
                          </div>
                        </div>
                      )}
                    </>
                  )}

                  {/* Common Scope Selector */}
                  <div className="pt-4 border-t border-dashed border-slate-200 dark:border-slate-800 space-y-4">
                    <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-white/50 dark:bg-slate-900/50 space-y-3">
                      <label className="text-xs font-bold text-text-muted uppercase tracking-wider flex items-center gap-2">
                        <Target className="w-3 h-3" /> Target Scope
                      </label>
                      <select
                        value={targetScope}
                        onChange={(e) => setTargetScope(e.target.value as any)}
                        className="w-full bg-white dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-lg px-3 py-2.5 text-sm font-medium focus:outline-none focus:border-accent focus:ring-1 focus:ring-accent transition-all shadow-sm text-text-primary dark:text-white mb-2"
                      >
                        <option value="all">Entire Organization</option>
                        <option value="department">Specific Department</option>
                      </select>
                      {targetScope === 'department' && (
                        <select
                          value={targetDept}
                          onChange={(e) => setTargetDept(e.target.value)}
                          className="w-full bg-white dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-lg px-3 py-2.5 text-sm font-medium focus:outline-none focus:border-accent focus:ring-1 focus:ring-accent transition-all shadow-sm text-text-primary dark:text-white animate-in fade-in slide-in-from-top-1"
                        >
                          <option value="">Select Department...</option>
                          {departments.map(d => <option key={d} value={d}>{d}</option>)}
                        </select>
                      )}
                    </div>
                  </div>
                </div>

                <button
                  onClick={runScenario}
                  disabled={isLoading}
                  className="w-full group mt-4 py-4 bg-gradient-to-r from-accent to-accent-600 hover:from-accent-600 hover:to-accent text-white rounded-xl font-bold shadow-glow hover:shadow-glow-lg transition-all active:scale-[0.98] flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isLoading ? <RefreshCw className="w-5 h-5 animate-spin" /> : <Play className="w-5 h-5 fill-current" />}
                  {isLoading ? 'Simulating...' : 'Run Simulation'}
                </button>
              </div>
            </GlassCard>

            {/* Methodology Note */}
            <div className="px-4">
              <div className="flex items-start gap-3 opacity-60 hover:opacity-100 transition-opacity">
                <Info className="w-4 h-4 mt-0.5 flex-shrink-0" />
                <p className="text-xs leading-relaxed">
                  Uses Monte Carlo simulation (1,000 iterations) to predict outcomes. Results include 95% confidence intervals based on your historical data patterns.
                </p>
              </div>
            </div>
          </div>

          {/* Results Panel (Right - Scrollable) */}
          <div className="lg:col-span-8 overflow-y-auto pr-2 pb-6">
            {/* Empty State */}
            {!result && !isLoading && (
              <div className="h-full flex flex-col items-center justify-center text-center p-12 opacity-50">
                <div className="w-24 h-24 bg-white/5 rounded-full flex items-center justify-center mb-6">
                  <Calculator className="w-10 h-10 text-text-muted" />
                </div>
                <h3 className="text-xl font-bold text-text-primary dark:text-white">Ready to Simulate</h3>
                <p className="max-w-md text-text-secondary mt-2">Adjust the parameters on the left and run a simulation to see the projected impact on your workforce.</p>
              </div>
            )}

            {/* Loading State */}
            {isLoading && (
              <div className="h-full flex flex-col items-center justify-center text-center p-12">
                <div className="w-24 h-24 bg-accent/10 rounded-full flex items-center justify-center mb-6 relative">
                  <div className="absolute inset-0 border-4 border-accent/20 rounded-full border-t-accent animate-spin" />
                  <RefreshCw className="w-10 h-10 text-accent animate-pulse" />
                </div>
                <h3 className="text-xl font-bold text-text-primary dark:text-white">Crunching numbers...</h3>
                <p className="max-w-md text-text-secondary mt-2">Running 1,000 Monte Carlo iterations across your dataset...</p>
              </div>
            )}

            {/* Results Dashboard */}
            {result && (
              <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
                {/* Top Summary Cards */}
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                  <GlassCard className="p-4 text-center">
                    <p className="text-xs font-bold text-text-muted uppercase tracking-wider mb-1">Impact</p>
                    <p className="text-2xl font-bold font-display">{result.affected_employees}</p>
                    <p className="text-[10px] text-text-secondary">Employees</p>
                  </GlassCard>
                  <GlassCard className="p-4 text-center">
                    <p className="text-xs font-bold text-text-muted uppercase tracking-wider mb-1">Confidence</p>
                    <p className={cn("text-2xl font-bold font-display", result.confidence_level === 'High' ? 'text-success' : 'text-warning')}>{result.confidence_level}</p>
                    <p className="text-[10px] text-text-secondary">Statistical Certainty</p>
                  </GlassCard>
                  <GlassCard className="p-4 text-center">
                    <p className="text-xs font-bold text-text-muted uppercase tracking-wider mb-1">ROI Prob</p>
                    <p className={cn("text-2xl font-bold font-display", result.simulation.roi_positive_probability > 0.6 ? 'text-success' : 'text-warning')}>{(result.simulation.roi_positive_probability * 100).toFixed(0)}%</p>
                    <p className="text-[10px] text-text-secondary">Chance of Positive ROI</p>
                  </GlassCard>
                  <GlassCard className="p-4 text-center">
                    <p className="text-xs font-bold text-text-muted uppercase tracking-wider mb-1">Net Impact</p>
                    <p className={cn("text-2xl font-bold font-display", result.cost_impact.net_impact >= 0 ? 'text-success' : 'text-danger')}>{formatCurrency(result.cost_impact.net_impact)}</p>
                    <p className="text-[10px] text-text-secondary">Projected Value</p>
                  </GlassCard>
                </div>

                {/* Main Analysis */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <GlassCard className="flex flex-col">
                    <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                      <BarChart3 className="w-5 h-5 text-accent" />
                      Turnover Projection
                    </h3>
                    <div className="flex-1 flex items-center justify-center p-4">
                      {/* Simple Visual for Turnover Change */}
                      <div className="flex gap-8 items-end">
                        <div className="flex flex-col items-center gap-2">
                          <span className="text-3xl font-bold text-text-muted">{result.baseline_turnover_rate}%</span>
                          <div className="w-16 h-32 bg-slate-200 dark:bg-slate-700 rounded-t-xl relative overflow-hidden">
                            <div className="absolute bottom-0 w-full bg-slate-400 dark:bg-slate-500 transition-all duration-1000" style={{ height: `${result.baseline_turnover_rate * 2}%` }} />
                          </div>
                          <span className="text-xs font-bold text-text-muted uppercase">Baseline</span>
                        </div>
                        <ArrowRight className="w-6 h-6 text-text-muted mb-12" />
                        <div className="flex flex-col items-center gap-2">
                          <span className={cn("text-3xl font-bold", result.projected_turnover_rate < result.baseline_turnover_rate ? 'text-success' : 'text-danger')}>{result.projected_turnover_rate}%</span>
                          <div className="w-16 h-32 bg-slate-200 dark:bg-slate-700 rounded-t-xl relative overflow-hidden">
                            <div className={cn("absolute bottom-0 w-full transition-all duration-1000", result.projected_turnover_rate < result.baseline_turnover_rate ? 'bg-success' : 'bg-danger')} style={{ height: `${result.projected_turnover_rate * 2}%` }} />
                          </div>
                          <span className="text-xs font-bold text-text-muted uppercase">Projected</span>
                        </div>
                      </div>
                    </div>
                  </GlassCard>

                  <GlassCard>
                    <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                      <DollarSign className="w-5 h-5 text-accent" />
                      Financial Breakdown
                    </h3>
                    <div className="space-y-4">
                      <div className="flex justify-between items-center p-3 rounded-lg bg-surface-secondary">
                        <span className="text-sm font-medium">Cost Investment</span>
                        <span className="font-bold text-danger">{formatCurrency(result.cost_impact.total_cost)}</span>
                      </div>
                      <div className="flex justify-between items-center p-3 rounded-lg bg-surface-secondary">
                        <span className="text-sm font-medium">Retention Savings</span>
                        <span className="font-bold text-success">{formatCurrency(result.cost_impact.total_benefit)}</span>
                      </div>
                      <div className="h-px bg-white/10 my-2" />
                      <div className="flex justify-between items-center p-3 rounded-lg bg-surface-secondary border border-white/5">
                        <span className="text-sm font-bold">Net Financial Impact</span>
                        <span className={cn("font-bold", result.cost_impact.net_impact >= 0 ? 'text-success' : 'text-danger')}>
                          {formatCurrency(result.cost_impact.net_impact)}
                        </span>
                      </div>
                    </div>
                  </GlassCard>
                </div>

                {/* Recommendation Engine */}
                <GlassCard className="border-l-4 border-l-accent relative overflow-hidden">
                  <div className="absolute top-0 right-0 p-8 opacity-10 text-accent">
                    <CheckCircle className="w-32 h-32" />
                  </div>
                  <div className="relative z-10">
                    <h3 className="text-lg font-bold mb-2 flex items-center gap-2">
                      <CheckCircle className="w-5 h-5 text-accent" />
                      Strategic Recommendation
                    </h3>
                    <p className="text-lg font-medium text-text-primary dark:text-white mb-4 leading-relaxed max-w-2xl">
                      {result.recommendation}
                    </p>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
                      <div>
                        <p className="text-xs font-bold text-text-muted uppercase tracking-wider mb-2">Primary Risks</p>
                        <ul className="space-y-2">
                          {result.risks.slice(0, 3).map((risk, i) => (
                            <li key={i} className="flex items-start gap-2 text-sm text-text-secondary">
                              <AlertTriangle className="w-4 h-4 text-warning mt-0.5 flex-shrink-0" />
                              {risk}
                            </li>
                          ))}
                        </ul>
                      </div>
                      <div>
                        <p className="text-xs font-bold text-text-muted uppercase tracking-wider mb-2">Alternatives</p>
                        <ul className="space-y-2">
                          {result.alternative_actions.slice(0, 3).map((alt, i) => (
                            <li key={i} className="flex items-start gap-2 text-sm text-text-secondary">
                              <ChevronRight className="w-4 h-4 text-accent mt-0.5 flex-shrink-0" />
                              {alt}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </div>
                </GlassCard>
              </div>
            )}
          </div>
        </div>
      ) : (
        <SavedScenariosList activeTab={activeTab} setActiveTab={setActiveTab} />
      )}
    </div>
  )
}


function SavedScenariosList({ activeTab, setActiveTab }: { activeTab: TabType, setActiveTab: (t: TabType) => void }) {
  const { data, isLoading } = useQuery({
    queryKey: ['scenario', 'recent'],
    queryFn: () => api.scenario.getRecentScenarios(10).catch(() => ({ scenarios: [] }))
  })

  // Mock data for visual verification if API is empty
  const hasData = (data as any)?.scenarios?.length > 0;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 p-1">
      <div className="lg:col-span-12">
        <div className="bg-gradient-to-r from-accent/5 to-accent/10 border border-accent/20 rounded-xl p-6 mb-6">
          <h2 className="text-xl font-bold mb-2">Saved Scenarios</h2>
          <p className="text-text-secondary">Review past simulations and compare their outcomes.</p>
        </div>

        {isLoading ? (
          <div className="flex justify-center p-12"><RefreshCw className="w-8 h-8 animate-spin text-accent" /></div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {(data as any)?.scenarios?.map((s: any) => (
              <GlassCard key={s.scenario_id} className="hover:border-accent/50 transition-colors cursor-pointer group relative">
                <div className="flex justify-between items-start mb-4">
                  <Badge variant="outline" className="capitalize">{s.scenario_type}</Badge>
                  <span className="text-xs text-text-muted">{new Date(s.computed_at).toLocaleDateString()}</span>
                </div>
                <h3 className="font-bold text-lg mb-2 group-hover:text-accent transition-colors">{s.scenario_name || 'Untitled Scenario'}</h3>

                <div className="space-y-2 mt-4">
                  <div className="flex justify-between text-sm">
                    <span className="text-text-secondary">ROI Estimate</span>
                    <span className={cn("font-bold", (s.roi_estimate || 0) > 0 ? 'text-success' : 'text-danger')}>{(s.roi_estimate || 0).toFixed(0)}%</span>
                  </div>
                </div>
              </GlassCard>
            ))}
            {!hasData && (
              <div className="col-span-full text-center py-12 text-text-secondary">
                No saved scenarios found. Run a simulation to save it automatically.
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function ArrowRight({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}><path d="M5 12h14" /><path d="m12 5 7 7-7 7" /></svg>
  )
}
