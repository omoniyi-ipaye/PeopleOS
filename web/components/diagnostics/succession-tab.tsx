'use client'

import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api-client'
import { Card } from '@/components/ui/card'
import { KPICard } from '@/components/dashboard/kpi-card'
import { Target, Users, ClipboardCheck, AlertCircle } from 'lucide-react'
import { NineBoxGrid } from '@/components/charts/nine-box-grid'

interface BenchRow {
    dept: string
    ready_now: number
    ready_soon: number
    developing: number
    assessed: number
    unassessed: number
    total: number
}
interface NineBoxRow { category: string; count: number; percentage: number }
interface GapRow { dept: string; ready_now: number; ready_soon: number; recommendation: string }

export function SuccessionTab() {
    const nineBox = useQuery({
        queryKey: ['succession', '9box', 'summary'],
        queryFn: () => api.succession.get9BoxSummary() as Promise<NineBoxRow[]>,
    })
    const bench = useQuery({
        queryKey: ['succession', 'bench-strength'],
        queryFn: () => api.succession.getBenchStrength() as Promise<BenchRow[]>,
    })
    const gaps = useQuery({
        queryKey: ['succession', 'gaps'],
        queryFn: () => api.succession.getGaps() as Promise<GapRow[]>,
    })

    if (nineBox.isLoading || bench.isLoading || gaps.isLoading) {
        return <p role="status">Loading recorded assessments…</p>
    }
    if (nineBox.isError || bench.isError || gaps.isError) {
        return <p role="alert">Succession assessments are unavailable. Retry when the data is available.</p>
    }

    const rows = bench.data ?? []
    const assessed = rows.reduce((sum, row) => sum + row.assessed, 0)
    const total = rows.reduce((sum, row) => sum + row.total, 0)
    const readyNow = rows.reduce((sum, row) => sum + row.ready_now, 0)
    // Combine assessment counts directly; do not average rounded department scores.
    const points = rows.reduce((sum, row) => sum + row.ready_now + .7 * row.ready_soon +
        .3 * row.developing + .1 * (row.assessed - row.ready_now - row.ready_soon - row.developing), 0)

    return (
        <div className="space-y-6">
            <p className="text-sm text-text-secondary">
                These summaries describe recorded talent-review assessments. Readiness requires a
                SuccessionReadiness assessment; the matrix requires separate performance and potential
                ratings on a 1–5 scale. Missing assessments remain unassessed. Role coverage and future
                performance have not been established by these summaries.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <KPICard title="Recorded Ready Now" value={assessed ? readyNow : 'Unassessed'} icon={Target}
                    subtitle="Among assessed employees" insight="Count of recorded Ready Now assessments." />
                <KPICard title="Assessment Index" value={assessed ? `${(points / assessed).toFixed(3)} / 1` : 'Unavailable'} icon={Users}
                    subtitle="Weighted assessed records" insight="Average of Ready Now = 1, Ready 1–2 Years = .7, Developing = .3 and Early Career = .1. This is a descriptive index." />
                <KPICard title="Assessment Coverage" value={total ? `${(assessed / total * 100).toFixed(1)}%` : 'Unavailable'} icon={ClipboardCheck}
                    subtitle={`${assessed} assessed; ${total - assessed} unassessed`}
                    insight="Share of active employees with a recognized recorded readiness assessment." />
                <KPICard title="Departments for Review" value={assessed ? (gaps.data?.length ?? 0) : 'Unassessed'} icon={AlertCircle}
                    subtitle="Based on available assessments" insight="Departments with assessed records but no Ready Now assessment, or an assessment index below .3. Missing assessments limit interpretation." />
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card title="Recorded Talent Assessments" subtitle="Performance and separately assessed potential">
                    <NineBoxGrid data={nineBox.data ?? []} />
                </Card>
                <Card title="Department Assessment Review" subtitle="Review incomplete coverage before drawing conclusions">
                    <div className="space-y-4">
                        {gaps.data?.map(gap => (
                            <div key={gap.dept} className="p-4 rounded-xl border border-border dark:border-border-dark">
                                <p className="font-bold">{gap.dept}</p>
                                <p className="text-sm">Recorded: {gap.ready_now} Ready Now; {gap.ready_soon} Ready 1–2 Years</p>
                                <p className="mt-2 text-sm text-text-secondary">{gap.recommendation}</p>
                            </div>
                        ))}
                        {!gaps.data?.length && <p className="text-sm text-text-muted">
                            {assessed ? 'No departments meet the review rule among assessed records.' : 'No recorded readiness assessments are available.'}
                        </p>}
                    </div>
                </Card>
            </div>
        </div>
    )
}
