'use client'

import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api-client'
import { Card } from '@/components/ui/card'
import { KPICard } from '@/components/dashboard/kpi-card'
import { DollarSign, Users } from 'lucide-react'

interface CompensationAnalysis {
    summary: { avg_salary: number | null; total_payroll: number | null; headcount: number }
    equity_scores: { dept: string; equity_score: number; headcount: number }[]
    warnings: string[]
}

export function CompensationTab() {
    const { data, isLoading, isError } = useQuery<CompensationAnalysis>({
        queryKey: ['compensation', 'analysis'],
        queryFn: () => api.compensation.getAnalysis() as Promise<CompensationAnalysis>,
    })
    if (isLoading) return <p role="status">Loading compensation measurements…</p>
    if (isError || !data) return <p role="status">Compensation measurements are unavailable.</p>
    const { summary, equity_scores, warnings } = data
    const format = (value: number | null) => value != null && Number.isFinite(value) ? value.toLocaleString() : 'Unavailable'
    return <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <KPICard title="Mean recorded salary" value={format(summary.avg_salary)} icon={DollarSign} subtitle="Source salary units" insight="Mean of valid positive salaries for current active employees." />
            <KPICard title="Recorded salary total" value={format(summary.total_payroll)} icon={DollarSign} subtitle="Current employees with valid salaries" insight="Sum of recorded salaries; currency and pay period follow the source contract." />
            <KPICard title="Salary observations" value={summary.headcount} icon={Users} subtitle="Valid positive salary records" insight="Records included in the displayed salary aggregates." />
        </div>
        <Card title="Department salary dispersion" subtitle="Descriptive consistency index (0–100); adjusted pay equity has not been estimated.">
            <table className="w-full text-sm"><thead><tr><th scope="col" className="text-left">Department</th><th scope="col">Observations</th><th scope="col">Consistency index</th></tr></thead>
                <tbody>{equity_scores.map(row => <tr key={row.dept}><th scope="row" className="text-left py-2">{row.dept}</th><td className="text-center">{row.headcount}</td><td className="text-center">{format(Number.isFinite(row.equity_score) ? row.equity_score * 100 : null)}</td></tr>)}</tbody>
            </table>
            {!equity_scores.length && <p>No department dispersion measurements are available.</p>}
        </Card>
        {warnings.map((warning, index) => <p key={index} className="text-sm text-text-muted">{warning}</p>)}
    </div>
}
