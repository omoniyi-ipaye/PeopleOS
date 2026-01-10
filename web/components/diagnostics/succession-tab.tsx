'use client'

import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api-client'
import { Card } from '@/components/ui/card'
import { KPICard } from '@/components/dashboard/kpi-card'
import { Target, Users, TrendingUp, AlertCircle, Info } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { NineBoxGrid } from '@/components/charts/nine-box-grid'

export function SuccessionTab() {
    const { data: nineBoxData, isLoading: nineBoxLoading } = useQuery<any>({
        queryKey: ['succession', '9box', 'summary'],
        queryFn: api.succession.get9BoxSummary as any,
    })

    const { data: benchStrength, isLoading: benchLoading } = useQuery<any>({
        queryKey: ['succession', 'bench-strength'],
        queryFn: api.succession.getBenchStrength as any,
    })

    const { data: gaps } = useQuery<any>({
        queryKey: ['succession', 'gaps'],
        queryFn: api.succession.getGaps as any,
    })

    const { data: highPotentials } = useQuery<any>({
        queryKey: ['succession', 'high-potentials'],
        queryFn: api.succession.getHighPotentials as any,
    })

    if (nineBoxLoading || benchLoading) return <div className="animate-pulse space-y-4">
        <div className="h-24 bg-surface dark:bg-surface-dark rounded-xl" />
        <div className="grid grid-cols-2 gap-4">
            <div className="h-64 bg-surface dark:bg-surface-dark rounded-xl" />
            <div className="h-64 bg-surface dark:bg-surface-dark rounded-xl" />
        </div>
    </div>

    const readyNow = benchStrength?.reduce((acc: number, curr: any) => acc + curr.ready_now, 0) || 0
    const avgBench = benchStrength?.reduce((acc: number, curr: any) => acc + curr.bench_strength, 0) / (benchStrength?.length || 1) || 0

    return (
        <div className="space-y-6">
            {/* Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <KPICard
                    title="Promotion Ready"
                    value={readyNow}
                    icon={Target}
                    subtitle="Succession candidates"
                    variant="success"
                    insight="Employees identified as capable of moving into more senior roles immediately."
                />
                <KPICard
                    title="Bench Strength"
                    value={`${(avgBench * 100).toFixed(0)}%`}
                    icon={Users}
                    subtitle="Org-wide avg"
                    variant={avgBench < 0.3 ? 'danger' : avgBench < 0.6 ? 'warning' : 'success'}
                    insight="A measure of how many roles have a qualified successor ready to step in."
                />
                <KPICard
                    title="High Potentials"
                    value={highPotentials?.length || 0}
                    icon={TrendingUp}
                    subtitle="Top tier talent"
                    insight="Employees showing the strongest combination of high performance and career potential."
                />
                <KPICard
                    title="Key Role Risks"
                    value={gaps?.length || 0}
                    icon={AlertCircle}
                    subtitle="Roles without successors"
                    variant={(gaps?.length || 0) > 0 ? 'danger' : 'default'}
                    insight="Critical positions that currently lack a clear internal successor."
                />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* 9-Box Matrix */}
                <Card title="Talent Calibration" subtitle="Performance vs Potential (Succession Matrix)">
                    <div className="h-[400px]">
                        <NineBoxGrid data={nineBoxData || []} />
                    </div>
                </Card>

                {/* Critical Gaps Table */}
                <Card title="Role Succession Risks" subtitle="Departments needing immediate successor planning">
                    <div className="space-y-4">
                        {gaps?.map((gap: any) => (
                            <div key={gap.dept} className="p-4 rounded-xl bg-surface-hover dark:bg-surface-dark-hover border border-border dark:border-border-dark">
                                <div className="flex justify-between items-start mb-2">
                                    <div className="font-bold text-text-primary dark:text-text-dark-primary">{gap.dept}</div>
                                    <Badge variant={gap.gap_severity === 'Critical' ? 'danger' : 'warning'}>
                                        {gap.gap_severity} Gap
                                    </Badge>
                                </div>
                                <div className="text-xs text-text-secondary dark:text-text-dark-secondary mb-3">
                                    Readiness: {gap.ready_now} Ready Now • {gap.ready_soon} Ready Soon
                                </div>
                                <div className="p-2 bg-accent/5 rounded-lg border border-accent/10 flex gap-2">
                                    <Info className="w-4 h-4 text-accent shrink-0 mt-0.5" />
                                    <p className="text-[10px] text-accent font-medium leading-normal">
                                        {gap.recommendation}
                                    </p>
                                </div>
                            </div>
                        ))}
                        {(gaps?.length === 0 || !gaps) && (
                            <div className="h-64 flex items-center justify-center text-text-muted text-sm">
                                No critical succession gaps detected
                            </div>
                        )}
                    </div>
                </Card>
            </div>

            {/* Identified Candidates */}
            <Card title="Succession Candidates" subtitle="High-potential employees identified for future leadership roles">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {highPotentials?.map((hp: any) => (
                        <div key={hp.employee_id} className="p-4 rounded-xl bg-surface-hover dark:bg-surface-dark-hover border border-border dark:border-border-dark flex items-center justify-between">
                            <div className="flex items-center gap-3">
                                <div className="w-10 h-10 rounded-full bg-accent/10 flex items-center justify-center">
                                    <Users className="w-5 h-5 text-accent" />
                                </div>
                                <div>
                                    <div className="font-bold text-sm text-text-primary dark:text-text-dark-primary">{hp.employee_id}</div>
                                    <div className="text-[10px] text-text-secondary dark:text-text-dark-secondary">{hp.dept}</div>
                                </div>
                            </div>
                            <div className="text-right">
                                <Badge variant="success" size="sm" className="mb-1">{hp.potential_level}</Badge>
                                <div className="text-[10px] text-text-muted">Rating: {hp.last_rating}</div>
                            </div>
                        </div>
                    ))}
                    {(highPotentials?.length === 0 || !highPotentials) && (
                        <div className="col-span-full h-32 flex items-center justify-center text-text-muted text-sm">
                            No candidates identified yet
                        </div>
                    )}
                </div>
            </Card>
        </div>
    )
}
