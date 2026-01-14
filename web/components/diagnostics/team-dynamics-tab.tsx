'use client'

import { useState, useMemo, useCallback } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api-client'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'
import {
  Users,
  UserCheck,
  Clock,
  Star,
  TrendingUp,
  Building2,
  MapPin,
  Filter,
  X,
  ChevronDown,
  Briefcase,
  Heart,
  AlertTriangle,
  RefreshCw,
  Activity,
  PieChart,
  BarChart3,
  Smile,
  Table,
} from 'lucide-react'

// Types
interface TeamHealth {
  dept: string
  health_score: number
  avg_tenure: number | null
  avg_rating: number | null
  headcount: number
  attrition_rate: number | null
  status: string
}

interface FilterOptions {
  departments: string[]
  locations: string[]
  countries: string[]
  genders: string[]
  age_groups: string[]
  job_levels: number[]
  tenure_ranges: string[]
}

interface Filters {
  departments: string[]
  locations: string[]
  countries: string[]
  genders: string[]
  age_groups: string[]
  job_levels: number[]
  tenure_ranges: string[]
}

interface GenderBreakdown {
  male: number
  female: number
  other: number
  male_pct: number
  female_pct: number
  other_pct: number
}

interface AgeDistribution {
  group: string
  count: number
  percentage: number
}

interface SatisfactionMetrics {
  avg_enps: number | null
  avg_pulse: number | null
  avg_manager_satisfaction: number | null
  avg_work_life_balance: number | null
  avg_career_growth: number | null
}

interface PerformanceDistribution {
  rating_range: string
  count: number
  percentage: number
}

interface ComprehensiveMetrics {
  total_employees: number
  filters_applied: Record<string, string[]>
  gender_breakdown: GenderBreakdown
  age_distribution: AgeDistribution[]
  location_distribution: { location: string; count: number; percentage: number }[]
  composition: {
    by_tenure: { group: string; count: number; percentage: number }[]
    by_job_level: { level: number; name: string; count: number; percentage: number }[]
    avg_span_of_control: number | null
  }
  avg_rating: number | null
  performance_distribution: PerformanceDistribution[]
  top_performers_count: number
  top_performers_pct: number
  satisfaction: SatisfactionMetrics
  avg_tenure: number | null
  attrition_rate: number | null
  avg_manager_changes: number | null
  new_hires_count: number
  veterans_count: number
  by_department: {
    department: string
    headcount: number
    percentage: number
    male_pct?: number
    female_pct?: number
    avg_rating?: number
    avg_tenure?: number
    avg_enps?: number
    avg_manager_satisfaction?: number
  }[]
}

// Sub-tab configuration
type SubTab = 'health' | 'demographics' | 'composition' | 'satisfaction' | 'departments'

const SUB_TABS: { id: SubTab; label: string; icon: React.ElementType }[] = [
  { id: 'health', label: 'Health & Overview', icon: Activity },
  { id: 'demographics', label: 'Demographics', icon: PieChart },
  { id: 'composition', label: 'Composition', icon: BarChart3 },
  { id: 'satisfaction', label: 'Satisfaction', icon: Smile },
  { id: 'departments', label: 'By Department', icon: Table },
]

// Filter Chip Component
function FilterChip({ label, onRemove }: { label: string; onRemove: () => void }) {
  return (
    <span className="inline-flex items-center gap-1 px-2 py-1 bg-accent/10 text-accent text-xs font-medium rounded-full">
      {label}
      <button onClick={onRemove} className="hover:bg-accent/20 rounded-full p-0.5 transition-colors">
        <X className="w-3 h-3" />
      </button>
    </span>
  )
}

// Multi-Select Dropdown Component
function MultiSelectDropdown({
  label,
  icon: Icon,
  options,
  selected,
  onChange,
}: {
  label: string
  icon: React.ElementType
  options: string[]
  selected: string[]
  onChange: (selected: string[]) => void
}) {
  const [isOpen, setIsOpen] = useState(false)

  const toggleOption = useCallback((option: string) => {
    if (selected.includes(option)) {
      onChange(selected.filter((s) => s !== option))
    } else {
      onChange([...selected, option])
    }
  }, [selected, onChange])

  if (options.length === 0) return null

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          "flex items-center gap-2 px-3 py-1.5 text-xs border rounded-lg transition-all",
          selected.length > 0
            ? "border-accent bg-accent/5 text-accent"
            : "border-border hover:border-accent/50 text-text-secondary"
        )}
      >
        <Icon className="w-3.5 h-3.5" />
        <span>{label}</span>
        {selected.length > 0 && (
          <span className="ml-1 px-1.5 py-0.5 bg-accent text-white text-[10px] rounded-full">
            {selected.length}
          </span>
        )}
        <ChevronDown className={cn("w-3.5 h-3.5 transition-transform", isOpen && "rotate-180")} />
      </button>

      {isOpen && (
        <>
          <div className="fixed inset-0 z-10" onClick={() => setIsOpen(false)} />
          <div className="absolute top-full left-0 mt-1 w-52 max-h-60 overflow-y-auto bg-white dark:bg-slate-800 border border-border rounded-lg shadow-lg z-20">
            <div className="p-1.5 space-y-0.5">
              {options.map((option) => (
                <button
                  key={option}
                  onClick={() => toggleOption(option)}
                  className={cn(
                    "w-full flex items-center gap-2 px-2.5 py-1.5 text-xs rounded-md transition-colors text-left",
                    selected.includes(option)
                      ? "bg-accent/10 text-accent"
                      : "hover:bg-surface-hover text-text-primary"
                  )}
                >
                  <div
                    className={cn(
                      "w-3.5 h-3.5 border rounded flex items-center justify-center flex-shrink-0",
                      selected.includes(option) ? "border-accent bg-accent" : "border-border"
                    )}
                  >
                    {selected.includes(option) && (
                      <svg className="w-2.5 h-2.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                      </svg>
                    )}
                  </div>
                  <span className="truncate">{option}</span>
                </button>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  )
}

// Stat Card Component
function StatCard({
  title,
  value,
  subtitle,
  icon: Icon,
  variant = 'default',
}: {
  title: string
  value: string | number
  subtitle?: string
  icon: React.ElementType
  variant?: 'default' | 'success' | 'warning' | 'danger'
}) {
  const variantColors = {
    default: 'text-accent bg-accent/10',
    success: 'text-success bg-success/10',
    warning: 'text-warning bg-warning/10',
    danger: 'text-danger bg-danger/10',
  }

  return (
    <div className="bg-surface dark:bg-surface-dark border border-border dark:border-border-dark rounded-xl p-4">
      <div className="flex items-start justify-between mb-2">
        <div className={cn("p-2 rounded-lg", variantColors[variant])}>
          <Icon className="w-4 h-4" />
        </div>
      </div>
      <div className="text-2xl font-bold text-text-primary dark:text-text-dark-primary">{value}</div>
      <div className="text-sm text-text-secondary dark:text-text-dark-secondary">{title}</div>
      {subtitle && <div className="text-xs text-text-muted mt-1">{subtitle}</div>}
    </div>
  )
}

export function TeamDynamicsTab() {
  const [activeSubTab, setActiveSubTab] = useState<SubTab>('health')
  const [filters, setFilters] = useState<Filters>({
    departments: [],
    locations: [],
    countries: [],
    genders: [],
    age_groups: [],
    job_levels: [],
    tenure_ranges: [],
  })

  // Fetch filter options
  const { data: filterOptions } = useQuery<FilterOptions>({
    queryKey: ['team', 'filters'],
    queryFn: () => api.team.getFilters() as Promise<FilterOptions>,
  })

  // Fetch comprehensive metrics with filters
  const { data: metrics, isLoading } = useQuery<ComprehensiveMetrics>({
    queryKey: ['team', 'comprehensive', filters],
    queryFn: () => api.team.getComprehensive(filters) as Promise<ComprehensiveMetrics>,
    staleTime: 0,
  })

  // Fetch team health scores
  const { data: teamHealth } = useQuery<TeamHealth[]>({
    queryKey: ['team', 'health'],
    queryFn: () => api.team.getHealth() as Promise<TeamHealth[]>,
  })

  const activeFilterCount = useMemo(() => {
    return Object.values(filters).reduce((sum, arr) => sum + arr.length, 0)
  }, [filters])

  const clearAllFilters = useCallback(() => {
    setFilters({
      departments: [],
      locations: [],
      countries: [],
      genders: [],
      age_groups: [],
      job_levels: [],
      tenure_ranges: [],
    })
  }, [])

  const removeFilter = useCallback((type: keyof Filters, value: string | number) => {
    setFilters((prev) => ({
      ...prev,
      [type]: prev[type].filter((v) => v !== value),
    }))
  }, [])

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="flex items-center gap-3">
          <RefreshCw className="w-5 h-5 animate-spin text-accent" />
          <span className="text-text-secondary">Loading team dynamics...</span>
        </div>
      </div>
    )
  }

  if (!metrics) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-4">
        <AlertTriangle className="w-12 h-12 text-warning" />
        <p className="text-text-secondary">No team data available</p>
      </div>
    )
  }

  return (
    <div className="flex gap-6 min-h-[600px]">
      {/* Left Sidebar - Sub Navigation */}
      <div className="w-56 flex-shrink-0">
        <div className="sticky top-4 space-y-2">
          {/* Sub-tabs */}
          <div className="bg-surface dark:bg-surface-dark border border-border dark:border-border-dark rounded-xl p-2">
            {SUB_TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveSubTab(tab.id)}
                className={cn(
                  "w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all text-left",
                  activeSubTab === tab.id
                    ? "bg-accent text-white shadow-md"
                    : "text-text-secondary hover:bg-surface-hover dark:hover:bg-surface-dark-hover hover:text-text-primary"
                )}
              >
                <tab.icon className="w-4 h-4" />
                {tab.label}
              </button>
            ))}
          </div>

          {/* Filters Section */}
          <div className="bg-surface dark:bg-surface-dark border border-border dark:border-border-dark rounded-xl p-3">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Filter className="w-4 h-4 text-accent" />
                <span className="font-semibold text-sm">Filters</span>
              </div>
              {activeFilterCount > 0 && (
                <button
                  onClick={clearAllFilters}
                  className="text-xs text-text-muted hover:text-danger transition-colors"
                >
                  Clear
                </button>
              )}
            </div>

            <div className="space-y-2">
              <MultiSelectDropdown
                label="Department"
                icon={Building2}
                options={filterOptions?.departments || []}
                selected={filters.departments}
                onChange={(selected) => setFilters((prev) => ({ ...prev, departments: selected }))}
              />
              <MultiSelectDropdown
                label="Location"
                icon={MapPin}
                options={filterOptions?.locations || []}
                selected={filters.locations}
                onChange={(selected) => setFilters((prev) => ({ ...prev, locations: selected }))}
              />
              <MultiSelectDropdown
                label="Gender"
                icon={Users}
                options={filterOptions?.genders || []}
                selected={filters.genders}
                onChange={(selected) => setFilters((prev) => ({ ...prev, genders: selected }))}
              />
              <MultiSelectDropdown
                label="Generation"
                icon={UserCheck}
                options={filterOptions?.age_groups || []}
                selected={filters.age_groups}
                onChange={(selected) => setFilters((prev) => ({ ...prev, age_groups: selected }))}
              />
              <MultiSelectDropdown
                label="Tenure"
                icon={Clock}
                options={filterOptions?.tenure_ranges || []}
                selected={filters.tenure_ranges}
                onChange={(selected) => setFilters((prev) => ({ ...prev, tenure_ranges: selected }))}
              />
            </div>

            {/* Active Filters */}
            {activeFilterCount > 0 && (
              <div className="flex flex-wrap gap-1.5 mt-3 pt-3 border-t border-border dark:border-border-dark">
                {filters.departments.map((d) => (
                  <FilterChip key={d} label={d} onRemove={() => removeFilter('departments', d)} />
                ))}
                {filters.locations.map((l) => (
                  <FilterChip key={l} label={l} onRemove={() => removeFilter('locations', l)} />
                ))}
                {filters.genders.map((g) => (
                  <FilterChip key={g} label={g} onRemove={() => removeFilter('genders', g)} />
                ))}
                {filters.age_groups.map((a) => (
                  <FilterChip key={a} label={a} onRemove={() => removeFilter('age_groups', a)} />
                ))}
                {filters.tenure_ranges.map((t) => (
                  <FilterChip key={t} label={t} onRemove={() => removeFilter('tenure_ranges', t)} />
                ))}
              </div>
            )}
          </div>

          {/* Quick Stats */}
          <div className="bg-gradient-to-br from-accent/10 to-accent/5 border border-accent/20 rounded-xl p-3">
            <div className="text-xs font-semibold text-accent uppercase tracking-wider mb-2">Selected</div>
            <div className="text-3xl font-bold text-text-primary">{metrics.total_employees.toLocaleString()}</div>
            <div className="text-xs text-text-muted">employees</div>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 space-y-6">
        {/* Health & Overview Tab */}
        {activeSubTab === 'health' && (
          <>
            {/* Team Health Scores */}
            {teamHealth && teamHealth.length > 0 && (
              <Card title="Team Health Scores" subtitle="Wellness indicators from performance feedback">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {teamHealth.map((team) => (
                    <div
                      key={team.dept}
                      className="p-4 rounded-xl bg-surface dark:bg-surface-dark border border-border dark:border-border-dark hover:shadow-md transition-all"
                    >
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2">
                          <div className="w-8 h-8 rounded-lg bg-accent/10 flex items-center justify-center">
                            <Building2 className="w-4 h-4 text-accent" />
                          </div>
                          <span className="font-semibold text-text-primary text-sm">{team.dept}</span>
                        </div>
                        <Badge
                          variant={
                            team.status === 'Thriving' ? 'success'
                              : team.status === 'Healthy' ? 'info'
                              : team.status === 'At Risk' ? 'warning' : 'danger'
                          }
                        >
                          {team.status}
                        </Badge>
                      </div>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-text-secondary">Wellness Score</span>
                          <span className="font-mono font-bold">{(team.health_score * 100).toFixed(0)}%</span>
                        </div>
                        <div className="h-2 bg-surface-secondary dark:bg-background-dark rounded-full overflow-hidden">
                          <div
                            className={cn(
                              "h-full rounded-full transition-all duration-700",
                              team.health_score >= 0.75 ? 'bg-success'
                                : team.health_score >= 0.6 ? 'bg-accent'
                                : team.health_score >= 0.45 ? 'bg-warning' : 'bg-danger'
                            )}
                            style={{ width: `${team.health_score * 100}%` }}
                          />
                        </div>
                        <div className="flex justify-between text-xs text-text-muted pt-1">
                          <span>{team.headcount} employees</span>
                          {team.avg_rating && <span>Rating: {team.avg_rating.toFixed(1)}</span>}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </Card>
            )}

            {/* Summary Stats */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <StatCard title="Avg Tenure" value={metrics.avg_tenure ? `${metrics.avg_tenure.toFixed(1)} yrs` : 'N/A'} icon={Clock} />
              <StatCard title="Avg Rating" value={metrics.avg_rating?.toFixed(2) || 'N/A'} icon={Star} variant={metrics.avg_rating && metrics.avg_rating >= 4 ? 'success' : 'default'} />
              <StatCard title="Top Performers" value={`${metrics.top_performers_pct}%`} subtitle={`${metrics.top_performers_count} employees`} icon={TrendingUp} variant="success" />
              <StatCard title="Attrition Rate" value={metrics.attrition_rate !== null && metrics.attrition_rate !== undefined ? `${metrics.attrition_rate}%` : 'N/A'} icon={AlertTriangle} variant={metrics.attrition_rate !== null && metrics.attrition_rate !== undefined && metrics.attrition_rate > 15 ? 'danger' : 'default'} />
            </div>

            {/* New Hires vs Veterans */}
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/10 border border-green-200 dark:border-green-800 rounded-xl p-5">
                <div className="flex items-center gap-3 mb-2">
                  <UserCheck className="w-5 h-5 text-green-600" />
                  <span className="font-semibold text-green-800 dark:text-green-300">New Hires</span>
                </div>
                <div className="text-3xl font-bold text-green-700 dark:text-green-400">{metrics.new_hires_count}</div>
                <div className="text-sm text-green-600 dark:text-green-500">Less than 1 year tenure</div>
              </div>
              <div className="bg-gradient-to-br from-amber-50 to-amber-100 dark:from-amber-900/20 dark:to-amber-800/10 border border-amber-200 dark:border-amber-800 rounded-xl p-5">
                <div className="flex items-center gap-3 mb-2">
                  <Briefcase className="w-5 h-5 text-amber-600" />
                  <span className="font-semibold text-amber-800 dark:text-amber-300">Veterans</span>
                </div>
                <div className="text-3xl font-bold text-amber-700 dark:text-amber-400">{metrics.veterans_count}</div>
                <div className="text-sm text-amber-600 dark:text-amber-500">5+ years tenure</div>
              </div>
            </div>
          </>
        )}

        {/* Demographics Tab */}
        {activeSubTab === 'demographics' && (
          <>
            {/* Gender Distribution */}
            <Card title="Gender Distribution" subtitle="Workforce composition by gender">
              <div className="space-y-4">
                <div className="flex items-center gap-6">
                  <div className="flex-1 h-10 rounded-full overflow-hidden flex shadow-inner">
                    <div className="bg-blue-500 transition-all duration-500 flex items-center justify-center text-white text-xs font-bold" style={{ width: `${metrics.gender_breakdown.male_pct}%` }}>
                      {metrics.gender_breakdown.male_pct > 10 && `${metrics.gender_breakdown.male_pct}%`}
                    </div>
                    <div className="bg-pink-500 transition-all duration-500 flex items-center justify-center text-white text-xs font-bold" style={{ width: `${metrics.gender_breakdown.female_pct}%` }}>
                      {metrics.gender_breakdown.female_pct > 10 && `${metrics.gender_breakdown.female_pct}%`}
                    </div>
                    {metrics.gender_breakdown.other_pct > 0 && (
                      <div className="bg-purple-500 transition-all duration-500" style={{ width: `${metrics.gender_breakdown.other_pct}%` }} />
                    )}
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-4 text-center">
                  <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-xl">
                    <div className="text-3xl font-bold text-blue-600">{metrics.gender_breakdown.male}</div>
                    <div className="text-sm text-text-secondary">Male ({metrics.gender_breakdown.male_pct}%)</div>
                  </div>
                  <div className="p-4 bg-pink-50 dark:bg-pink-900/20 rounded-xl">
                    <div className="text-3xl font-bold text-pink-600">{metrics.gender_breakdown.female}</div>
                    <div className="text-sm text-text-secondary">Female ({metrics.gender_breakdown.female_pct}%)</div>
                  </div>
                  {metrics.gender_breakdown.other > 0 && (
                    <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-xl">
                      <div className="text-3xl font-bold text-purple-600">{metrics.gender_breakdown.other}</div>
                      <div className="text-sm text-text-secondary">Other ({metrics.gender_breakdown.other_pct}%)</div>
                    </div>
                  )}
                </div>
              </div>
            </Card>

            {/* Age Distribution */}
            <Card title="Generational Mix" subtitle="Workforce composition by age group">
              <div className="space-y-4">
                {metrics.age_distribution.map((age) => {
                  const colors: Record<string, string> = {
                    'Gen Z (< 25)': '#10b981',
                    'Millennial (25-39)': '#3b82f6',
                    'Gen X (40-54)': '#8b5cf6',
                    'Boomer (55+)': '#f59e0b',
                  }
                  return (
                    <div key={age.group} className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="font-medium text-text-primary">{age.group}</span>
                        <span className="font-bold">{age.count} ({age.percentage}%)</span>
                      </div>
                      <div className="h-4 bg-surface-secondary dark:bg-background-dark rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-500"
                          style={{ width: `${age.percentage}%`, backgroundColor: colors[age.group] || '#6b7280' }}
                        />
                      </div>
                    </div>
                  )
                })}
              </div>
            </Card>

            {/* Top Locations */}
            <Card title="Top Locations" subtitle="Employee distribution by location">
              <div className="space-y-2">
                {metrics.location_distribution.slice(0, 10).map((loc, idx) => (
                  <div key={loc.location} className="flex items-center justify-between p-3 hover:bg-surface-hover rounded-lg transition-colors">
                    <div className="flex items-center gap-3">
                      <span className="w-7 h-7 flex items-center justify-center bg-accent/10 text-accent text-xs font-bold rounded-full">{idx + 1}</span>
                      <span className="font-medium">{loc.location}</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="w-24 h-2 bg-surface-secondary rounded-full overflow-hidden">
                        <div className="h-full bg-accent rounded-full" style={{ width: `${loc.percentage}%` }} />
                      </div>
                      <span className="font-bold w-12 text-right">{loc.count}</span>
                      <span className="text-xs text-text-muted w-12">({loc.percentage}%)</span>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          </>
        )}

        {/* Composition Tab */}
        {activeSubTab === 'composition' && (
          <>
            {/* Tenure Distribution */}
            <Card title="Tenure Distribution" subtitle="Employee experience levels">
              <div className="space-y-4">
                {metrics.composition.by_tenure.map((tenure) => {
                  const colors: Record<string, string> = {
                    'New (< 1 yr)': '#10b981',
                    'Growing (1-3 yrs)': '#3b82f6',
                    'Established (3-5 yrs)': '#8b5cf6',
                    'Veteran (5+ yrs)': '#f59e0b',
                  }
                  return (
                    <div key={tenure.group} className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="font-medium">{tenure.group}</span>
                        <span className="font-bold">{tenure.count} ({tenure.percentage}%)</span>
                      </div>
                      <div className="h-4 bg-surface-secondary dark:bg-background-dark rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-500"
                          style={{ width: `${tenure.percentage}%`, backgroundColor: colors[tenure.group] || '#6b7280' }}
                        />
                      </div>
                    </div>
                  )
                })}
              </div>
            </Card>

            {/* Performance Distribution */}
            <Card title="Performance Distribution" subtitle="Rating breakdown across workforce">
              <div className="space-y-4">
                {metrics.performance_distribution.map((perf) => {
                  const colors: Record<string, string> = {
                    'Outstanding (4.5+)': '#10b981',
                    'Exceeds (4.0-4.4)': '#3b82f6',
                    'Meets (3.0-3.9)': '#6b7280',
                    'Below (2.0-2.9)': '#f59e0b',
                    'Needs Improvement (< 2.0)': '#ef4444',
                  }
                  return (
                    <div key={perf.rating_range} className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="font-medium">{perf.rating_range}</span>
                        <span className="font-bold">{perf.count} ({perf.percentage}%)</span>
                      </div>
                      <div className="h-4 bg-surface-secondary dark:bg-background-dark rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-500"
                          style={{ width: `${perf.percentage}%`, backgroundColor: colors[perf.rating_range] || '#6b7280' }}
                        />
                      </div>
                    </div>
                  )
                })}
              </div>
            </Card>

            {/* Job Level Distribution */}
            {metrics.composition.by_job_level.length > 0 && (
              <Card title="Job Level Distribution" subtitle="Seniority breakdown">
                <div className="space-y-4">
                  {metrics.composition.by_job_level.map((level) => (
                    <div key={level.level} className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="font-medium">Level {level.level} - {level.name}</span>
                        <span className="font-bold">{level.count} ({level.percentage}%)</span>
                      </div>
                      <div className="h-4 bg-surface-secondary dark:bg-background-dark rounded-full overflow-hidden">
                        <div className="h-full bg-accent rounded-full transition-all duration-500" style={{ width: `${level.percentage}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </Card>
            )}

            {/* Span of Control */}
            {metrics.composition.avg_span_of_control && (
              <div className="bg-gradient-to-br from-indigo-50 to-indigo-100 dark:from-indigo-900/20 dark:to-indigo-800/10 border border-indigo-200 dark:border-indigo-800 rounded-xl p-6">
                <div className="flex items-center gap-3 mb-2">
                  <Users className="w-5 h-5 text-indigo-600" />
                  <span className="font-semibold text-indigo-800 dark:text-indigo-300">Avg Span of Control</span>
                </div>
                <div className="text-4xl font-bold text-indigo-700 dark:text-indigo-400">{metrics.composition.avg_span_of_control}</div>
                <div className="text-sm text-indigo-600 dark:text-indigo-500">direct reports per manager</div>
              </div>
            )}
          </>
        )}

        {/* Satisfaction Tab */}
        {activeSubTab === 'satisfaction' && (
          <>
            <div className="grid grid-cols-2 gap-6">
              <div className="bg-gradient-to-br from-green-50 to-emerald-100 dark:from-green-900/20 dark:to-emerald-800/10 border border-green-200 dark:border-green-800 rounded-xl p-6">
                <div className="text-sm font-semibold text-green-700 dark:text-green-400 mb-2">eNPS Score</div>
                <div className="text-5xl font-bold text-green-800 dark:text-green-300">
                  {metrics.satisfaction.avg_enps !== null ? metrics.satisfaction.avg_enps.toFixed(1) : 'N/A'}
                </div>
                <div className="text-sm text-green-600 dark:text-green-500 mt-2">Employee Net Promoter Score</div>
              </div>
              <div className="bg-gradient-to-br from-blue-50 to-sky-100 dark:from-blue-900/20 dark:to-sky-800/10 border border-blue-200 dark:border-blue-800 rounded-xl p-6">
                <div className="text-sm font-semibold text-blue-700 dark:text-blue-400 mb-2">Pulse Score</div>
                <div className="text-5xl font-bold text-blue-800 dark:text-blue-300">
                  {metrics.satisfaction.avg_pulse !== null ? metrics.satisfaction.avg_pulse.toFixed(2) : 'N/A'}
                </div>
                <div className="text-sm text-blue-600 dark:text-blue-500 mt-2">Latest pulse survey average</div>
              </div>
            </div>

            <Card title="Satisfaction Breakdown" subtitle="Key satisfaction indicators">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="text-center p-6 bg-surface-secondary dark:bg-background-dark rounded-xl">
                  <Heart className="w-8 h-8 text-pink-500 mx-auto mb-3" />
                  <div className="text-3xl font-bold text-text-primary">
                    {metrics.satisfaction.avg_manager_satisfaction !== null ? metrics.satisfaction.avg_manager_satisfaction.toFixed(2) : 'N/A'}
                  </div>
                  <div className="text-sm text-text-secondary mt-1">Manager Satisfaction</div>
                </div>
                <div className="text-center p-6 bg-surface-secondary dark:bg-background-dark rounded-xl">
                  <Clock className="w-8 h-8 text-blue-500 mx-auto mb-3" />
                  <div className="text-3xl font-bold text-text-primary">
                    {metrics.satisfaction.avg_work_life_balance !== null ? metrics.satisfaction.avg_work_life_balance.toFixed(2) : 'N/A'}
                  </div>
                  <div className="text-sm text-text-secondary mt-1">Work-Life Balance</div>
                </div>
                <div className="text-center p-6 bg-surface-secondary dark:bg-background-dark rounded-xl">
                  <TrendingUp className="w-8 h-8 text-purple-500 mx-auto mb-3" />
                  <div className="text-3xl font-bold text-text-primary">
                    {metrics.satisfaction.avg_career_growth !== null ? metrics.satisfaction.avg_career_growth.toFixed(2) : 'N/A'}
                  </div>
                  <div className="text-sm text-text-secondary mt-1">Career Growth</div>
                </div>
              </div>
            </Card>
          </>
        )}

        {/* Departments Tab */}
        {activeSubTab === 'departments' && (
          <Card title="Department Breakdown" subtitle="Comprehensive metrics by department">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border dark:border-border-dark">
                    <th className="text-left py-3 px-4 text-xs font-semibold text-text-muted uppercase tracking-wider">Department</th>
                    <th className="text-right py-3 px-4 text-xs font-semibold text-text-muted uppercase tracking-wider">Headcount</th>
                    <th className="text-center py-3 px-4 text-xs font-semibold text-text-muted uppercase tracking-wider">Gender Ratio</th>
                    <th className="text-right py-3 px-4 text-xs font-semibold text-text-muted uppercase tracking-wider">Avg Rating</th>
                    <th className="text-right py-3 px-4 text-xs font-semibold text-text-muted uppercase tracking-wider">Avg Tenure</th>
                    <th className="text-right py-3 px-4 text-xs font-semibold text-text-muted uppercase tracking-wider">eNPS</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics.by_department.map((dept) => (
                    <tr key={dept.department} className="border-b border-border/50 hover:bg-surface-hover transition-colors">
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-2">
                          <Building2 className="w-4 h-4 text-accent" />
                          <span className="font-semibold">{dept.department}</span>
                        </div>
                      </td>
                      <td className="py-3 px-4 text-right">
                        <span className="font-semibold">{dept.headcount}</span>
                        <span className="text-xs text-text-muted ml-1">({dept.percentage}%)</span>
                      </td>
                      <td className="py-3 px-4">
                        {dept.male_pct !== undefined && dept.female_pct !== undefined ? (
                          <div className="flex items-center gap-2 justify-center">
                            <div className="w-20 h-2 rounded-full overflow-hidden flex bg-surface-secondary">
                              <div className="bg-blue-500" style={{ width: `${dept.male_pct}%` }} />
                              <div className="bg-pink-500" style={{ width: `${dept.female_pct}%` }} />
                            </div>
                            <span className="text-xs text-text-muted">{dept.male_pct.toFixed(0)}%/{dept.female_pct.toFixed(0)}%</span>
                          </div>
                        ) : '-'}
                      </td>
                      <td className="py-3 px-4 text-right">
                        {dept.avg_rating !== undefined ? (
                          <span className={cn("font-semibold", dept.avg_rating >= 4 ? 'text-success' : dept.avg_rating >= 3 ? '' : 'text-warning')}>
                            {dept.avg_rating.toFixed(2)}
                          </span>
                        ) : '-'}
                      </td>
                      <td className="py-3 px-4 text-right text-text-secondary">
                        {dept.avg_tenure !== undefined ? `${dept.avg_tenure.toFixed(1)} yrs` : '-'}
                      </td>
                      <td className="py-3 px-4 text-right">
                        {dept.avg_enps !== undefined ? (
                          <span className={cn("font-semibold", dept.avg_enps >= 7 ? 'text-success' : dept.avg_enps >= 5 ? 'text-warning' : 'text-danger')}>
                            {dept.avg_enps.toFixed(1)}
                          </span>
                        ) : '-'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        )}
      </div>
    </div>
  )
}
