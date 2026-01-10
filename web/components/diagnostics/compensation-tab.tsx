'use client'

import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api-client'
import { Card } from '@/components/ui/card'
import { KPICard } from '@/components/dashboard/kpi-card'
import { DollarSign, TrendingDown, Users, AlertTriangle, Scale } from 'lucide-react'
import { DepartmentBarChart } from '@/components/charts/department-bar-chart'
import { Badge } from '@/components/ui/badge'

export function CompensationTab() {
    const { data, isLoading } = useQuery<any>({
        queryKey: ['compensation', 'analysis'],
        queryFn: api.compensation.getAnalysis as any,
    })

    if (isLoading) return <div className="animate-pulse space-y-4">
        <div className="h-24 bg-surface dark:bg-surface-dark rounded-xl" />
        <div className="grid grid-cols-2 gap-4">
            <div className="h-64 bg-surface dark:bg-surface-dark rounded-xl" />
            <div className="h-64 bg-surface dark:bg-surface-dark rounded-xl" />
        </div>
    </div>

    const { summary, equity_scores, outliers } = data || { summary: {}, equity_scores: [], outliers: [] }

    return (
        <div className="space-y-6">
            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <KPICard
                    title="Avg Salary"
                    value={`$${summary.avg_salary?.toLocaleString() || '0'}`}
                    icon={DollarSign}
                    subtitle="Across organization"
                    insight="The average annual salary across the entire workforce."
                />
                <KPICard
                    title="Total Payroll"
                    value={`$${(summary.total_payroll / 1e6).toFixed(1)}M`}
                    icon={TrendingDown}
                    subtitle="Annual spend"
                    variant="success"
                    insight="The total annual investment in base compensation."
                />
                <KPICard
                    title="Pay Equity"
                    value="84"
                    icon={Scale}
                    subtitle="Overall fairness"
                    variant="warning"
                    insight="A measure of how fairly pay is distributed across similar roles and backgrounds."
                />
                <KPICard
                    title="Pay Deviations"
                    value={outliers.length}
                    icon={AlertTriangle}
                    subtitle="Significant gaps"
                    variant={outliers.length > 0 ? 'danger' : 'default'}
                    insight="Number of employees whose pay significantly differs from their department average."
                />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Equity by Department */}
                <Card title="Pay Equity by Department" subtitle="Fairness index across different units (0-100)">
                    <DepartmentBarChart
                        data={equity_scores.map((s: any) => ({
                            dept: s.dept,
                            score: s.equity_score * 100
                        }))}
                        dataKey={"score" as any}
                    />
                </Card>

                {/* Salary Outliers Table */}
                <Card title="Pay Deviations" subtitle="Employees with significant compensation gaps">
                    <div className="space-y-3">
                        {outliers.slice(0, 5).map((outlier: any) => (
                            <div key={outlier.employee_id} className="flex items-center justify-between p-3 rounded-lg bg-surface-hover dark:bg-surface-dark-hover border border-border dark:border-border-dark">
                                <div>
                                    <div className="font-medium text-sm text-text-primary dark:text-text-dark-primary">{outlier.employee_id}</div>
                                    <div className="text-[10px] text-text-secondary dark:text-text-dark-secondary">{outlier.dept}</div>
                                </div>
                                <div className="text-right">
                                    <div className="font-bold text-sm text-danger">${outlier.salary.toLocaleString()}</div>
                                    <div className="text-[10px] text-danger/80">+{outlier.deviation_pct.toFixed(1)}% dev.</div>
                                </div>
                            </div>
                        ))}
                        {outliers.length === 0 && (
                            <div className="h-32 flex items-center justify-center text-text-muted text-sm">
                                No significant outliers detected
                            </div>
                        )}
                    </div>
                </Card>
            </div>
        </div>
    )
}
