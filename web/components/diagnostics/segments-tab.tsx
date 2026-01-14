'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import { GlassCard } from '@/components/ui/glass-card'
import { Badge } from '@/components/ui/badge'
import { api } from '@/lib/api-client'
import { Users, TrendingUp, TrendingDown, AlertTriangle, Sparkles, X, ChevronRight, User } from 'lucide-react'
import { cn } from '@/lib/utils'

interface ClusterResult {
    success: boolean
    n_clusters: number
    silhouette_score: number | null
    cluster_descriptions: Record<number, string>
    feature_summary: Record<string, Record<number, number>>
    cluster_counts: Record<number, number>
    top_departments: Record<number, Record<string, number>>
    excluded_count: number
}

interface ClusterMember {
    EmployeeID: string
    Dept: string
    JobTitle: string
    Salary: number
    LastRating: number
}

interface MembersResponse {
    success: boolean
    cluster_id: number
    count: number
    members: ClusterMember[]
}

// Map cluster traits to HR-friendly personas
const personaMapping: Record<string, { name: string; icon: React.ReactNode; color: string; description: string }> = {
    'High Salary, High LastRating': {
        name: 'Star Performers',
        icon: <Sparkles className="w-5 h-5" />,
        color: 'text-emerald-500',
        description: 'Top talent with high compensation and excellent performance. Retention priority.'
    },
    'High Salary, High Tenure': {
        name: 'Senior Leaders',
        icon: <Badge className="w-5 h-5 bg-purple-500/20 text-purple-500" variant="outline">Sen</Badge>, // Using Badge as icon placeholder or similar
        // Actually better to use Lucide icon
        color: 'text-purple-500',
        description: 'Well-compensated, long-tenured employees driving organizational stability.'
    },
    'Low Salary, Low Tenure': {
        name: 'New Joiners',
        icon: <Users className="w-5 h-5" />,
        color: 'text-blue-500',
        description: 'Recently hired employees. Focus on onboarding and early engagement.'
    },
    'High Tenure, Low LastRating': {
        name: 'At-Risk Veterans',
        icon: <AlertTriangle className="w-5 h-5" />,
        color: 'text-amber-500',
        description: 'Long-tenured employees showing performance decline. Consider re-engagement.'
    },
    'Low Salary': {
        name: 'Compensation Review',
        icon: <TrendingDown className="w-5 h-5" />,
        color: 'text-red-500',
        description: 'Employees with below-average compensation. Review for pay equity.'
    },
    'High Tenure': {
        name: 'Loyal Contributors',
        icon: <TrendingUp className="w-5 h-5" />,
        color: 'text-indigo-500',
        description: 'Long-tenured employees. Valuable institutional knowledge.'
    },
    'High InterviewScore': {
        name: 'High Potential Hires',
        icon: <Sparkles className="w-5 h-5" />,
        color: 'text-cyan-500',
        description: 'Recent hires with exceptional interview scores. Fast-track potential.'
    },
    'Average Profile': {
        name: 'Core Workforce',
        icon: <Users className="w-5 h-5" />,
        color: 'text-gray-500',
        description: 'Employees near organizational averages across all dimensions.'
    }
}

// Helper to find best matching persona
function getPersona(description: string) {
    // 1. Exact match
    if (personaMapping[description]) return personaMapping[description]

    // 2. Keyword match (Prioritized)
    // We check specific keywords that strongly indicate a persona
    if (description.includes('High LastRating') && description.includes('High Salary')) return personaMapping['High Salary, High LastRating']
    if (description.includes('High Tenure') && description.includes('High Salary')) return personaMapping['High Salary, High Tenure']
    if (description.includes('Low Tenure') && description.includes('Low Salary')) return personaMapping['Low Salary, Low Tenure']
    if (description.includes('High Tenure') && description.includes('Low LastRating')) return personaMapping['High Tenure, Low LastRating']

    // 3. Single specific trait match (Fallback)
    if (description.includes('High InterviewScore')) return personaMapping['High InterviewScore']

    // 4. Fallback to descriptive name
    return {
        name: description, // Use the traits as the name if no specific persona matches
        icon: <Users className="w-5 h-5" />,
        color: 'text-slate-400',
        description: 'Employee segment identified by specific clustering traits.'
    }
}

function EmployeePortfolioModal({
    employee,
    isOpen,
    onClose
}: {
    employee: ClusterMember | null,
    isOpen: boolean,
    onClose: () => void
}) {
    // In a real app, fetch more details using employee.EmployeeID
    // For now, display available info nicely

    return (
        <>
            <AnimatePresence>
                {isOpen && employee && (
                    <div className="fixed inset-0 z-[110] flex items-center justify-end p-4 sm:p-6 pointer-events-none">
                        <div className="absolute inset-0 bg-black/20 backdrop-blur-sm pointer-events-auto" onClick={onClose} />
                        <motion.div
                            initial={{ opacity: 0, x: 100 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: 100 }}
                            className="relative w-full max-w-md h-full pointer-events-auto shadow-2xl"
                        >
                            <div className="h-full flex flex-col bg-white dark:bg-slate-900 border-l border-gray-200 dark:border-white/10">
                                <div className="p-6 border-b border-gray-200 dark:border-white/10 flex items-center justify-between bg-gray-50 dark:bg-white/5">
                                    <div>
                                        <h3 className="text-xl font-display font-semibold text-gray-900 dark:text-white">
                                            Performance Portfolio
                                        </h3>
                                        <p className="text-sm text-text-secondary dark:text-slate-400 mt-1">
                                            confidential • internal only
                                        </p>
                                    </div>
                                    <button
                                        onClick={onClose}
                                        className="p-2 hover:bg-black/5 dark:hover:bg-white/10 rounded-full transition-colors text-text-secondary dark:text-slate-400 hover:text-text-primary dark:hover:text-white"
                                    >
                                        <X className="w-6 h-6" />
                                    </button>
                                </div>

                                <div className="flex-1 overflow-y-auto p-6 space-y-6">
                                    <div className="flex items-center gap-4">
                                        <div className="w-16 h-16 rounded-full bg-blue-500/20 flex items-center justify-center text-blue-500 dark:text-blue-400 text-2xl font-bold">
                                            <User className="w-8 h-8" />
                                        </div>
                                        <div>
                                            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">{employee.EmployeeID}</h2>
                                            <p className="text-text-secondary dark:text-slate-400">{employee.JobTitle} • {employee.Dept}</p>
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-2 gap-4">
                                        <div className="p-4 rounded-xl bg-gray-50 dark:bg-white/5 border border-gray-200 dark:border-white/5">
                                            <p className="text-xs text-text-muted uppercase tracking-wider mb-1">Last Rating</p>
                                            <p className="text-2xl font-bold text-amber-500">{employee.LastRating.toFixed(1)}</p>
                                        </div>
                                        <div className="p-4 rounded-xl bg-gray-50 dark:bg-white/5 border border-gray-200 dark:border-white/5">
                                            <p className="text-xs text-text-muted uppercase tracking-wider mb-1">Base Salary</p>
                                            <p className="text-2xl font-bold text-emerald-500">${(employee.Salary / 1000).toFixed(1)}k</p>
                                        </div>
                                    </div>

                                    <div className="space-y-3">
                                        <h4 className="text-sm font-bold text-gray-900 dark:text-white uppercase tracking-wider">Analysis</h4>
                                        <p className="text-sm text-text-secondary dark:text-slate-300 leading-relaxed">
                                            Employee shows strong indicators for their current segment.
                                            Review recent feedback timestamps and 1:1 notes for more context.
                                        </p>
                                    </div>
                                </div>
                            </div>
                        </motion.div>
                    </div>
                )}
            </AnimatePresence>
        </>
    )
}

function MemberModal({
    clusterId,
    personaName,
    nClusters,
    isOpen,
    onClose
}: {
    clusterId: number | null,
    personaName: string,
    nClusters: number,
    isOpen: boolean,
    onClose: () => void
}) {
    const { data, isLoading } = useQuery<MembersResponse>({
        queryKey: ['analytics', 'cluster-members', clusterId, nClusters],
        queryFn: () => api.analytics.getClusterMembers(clusterId!, nClusters) as Promise<MembersResponse>,
        enabled: isOpen && clusterId !== null
    })

    const [selectedEmployee, setSelectedEmployee] = useState<ClusterMember | null>(null)

    return (
        <AnimatePresence>
            {isOpen && (
                <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        onClick={onClose}
                        className="absolute inset-0 bg-black/40 backdrop-blur-sm"
                    />
                    <motion.div
                        initial={{ opacity: 0, scale: 0.9, y: 20 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.9, y: 20 }}
                        className="relative w-full max-w-2xl max-h-[80vh] overflow-hidden pointer-events-auto"
                    >
                        <GlassCard className="h-full border-white/20 shadow-2xl p-0 flex flex-col bg-white dark:bg-slate-900 text-text-primary dark:text-white">
                            <div className="p-6 border-b border-gray-200 dark:border-white/10 flex items-center justify-between bg-gray-50 dark:bg-white/5">
                                <div>
                                    <h3 className="text-xl font-display font-semibold text-gray-900 dark:text-white flex items-center gap-2">
                                        <Users className="w-5 h-5 text-blue-500 dark:text-blue-400" />
                                        {personaName} Members
                                    </h3>
                                    <p className="text-sm text-text-secondary dark:text-slate-400 mt-1">
                                        {data?.count || 0} employees identified in this segment
                                    </p>
                                </div>
                                <button
                                    onClick={onClose}
                                    className="p-2 hover:bg-black/5 dark:hover:bg-white/10 rounded-full transition-colors text-text-secondary dark:text-slate-400 hover:text-text-primary dark:hover:text-white"
                                >
                                    <X className="w-6 h-6" />
                                </button>
                            </div>

                            <div className="flex-1 overflow-y-auto p-4 custom-scrollbar bg-gray-50/50 dark:bg-slate-900/40 min-h-[300px]">
                                {isLoading ? (
                                    <div className="space-y-3">
                                        {[1, 2, 3].map(i => (
                                            <div key={i} className="h-20 bg-gray-200 dark:bg-white/5 animate-pulse rounded-xl border border-gray-200 dark:border-white/5" />
                                        ))}
                                    </div>
                                ) : (
                                    <div className="grid gap-3">
                                        {data?.members.map((member) => (
                                            <button
                                                key={member.EmployeeID}
                                                onClick={() => setSelectedEmployee(member)}
                                                className="group w-full p-4 bg-white dark:bg-white/5 hover:bg-blue-50 dark:hover:bg-white/10 border border-gray-200 dark:border-white/5 rounded-xl flex items-center justify-between transition-all text-left shadow-sm hover:shadow-md"
                                            >
                                                <div className="flex items-center gap-4">
                                                    <div className="w-10 h-10 rounded-full bg-blue-100 dark:bg-blue-500/20 flex items-center justify-center text-blue-600 dark:text-blue-400">
                                                        <User className="w-5 h-5" />
                                                    </div>
                                                    <div>
                                                        <p className="font-semibold text-gray-900 dark:text-white">{member.EmployeeID}</p>
                                                        <p className="text-xs text-text-secondary dark:text-slate-400">{member.JobTitle} • {member.Dept}</p>
                                                    </div>
                                                </div>
                                                <div className="flex items-center gap-6">
                                                    <div className="text-right">
                                                        <p className="text-xs text-text-muted font-medium tracking-tight">Salary</p>
                                                        <p className="text-sm text-emerald-600 dark:text-emerald-400 font-semibold">${(member.Salary / 1000).toFixed(1)}k</p>
                                                    </div>
                                                    <div className="text-right">
                                                        <p className="text-xs text-text-muted font-medium tracking-tight">Rating</p>
                                                        <p className="text-sm text-amber-600 dark:text-amber-400 font-semibold">{member.LastRating.toFixed(1)}</p>
                                                    </div>
                                                    <ChevronRight className="w-4 h-4 text-text-muted group-hover:text-blue-500 dark:group-hover:text-white transition-colors" />
                                                </div>
                                            </button>
                                        ))}
                                    </div>
                                )}
                            </div>

                            <div className="p-4 border-t border-gray-200 dark:border-white/10 bg-gray-50 dark:bg-white/5 flex gap-2">
                                <Badge variant="outline" className="bg-blue-50 dark:bg-blue-500/10 text-[10px] text-blue-600 dark:text-blue-400 border-blue-200 dark:border-blue-500/20">
                                    Segment Index: {clusterId}
                                </Badge>
                                <p className="text-[10px] text-text-muted flex-1 text-right italic pt-1">
                                    Click an employee to view their internal performance portfolio.
                                </p>
                            </div>
                        </GlassCard>
                    </motion.div>
                </div>
            )}
            <EmployeePortfolioModal
                employee={selectedEmployee}
                isOpen={!!selectedEmployee}
                onClose={() => setSelectedEmployee(null)}
            />
        </AnimatePresence>
    )
}

export function SegmentsTab() {
    const [selectedCluster, setSelectedCluster] = useState<{ id: number, name: string } | null>(null)
    const { data, isLoading, isError } = useQuery<ClusterResult>({
        queryKey: ['analytics', 'clusters', 4],
        queryFn: () => api.analytics.getClusters(4) as Promise<ClusterResult>,
    })

    // Use the actual n_clusters from the response for consistency
    const actualNClusters = data?.n_clusters ?? 4

    if (isLoading) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="animate-pulse-subtle text-text-secondary dark:text-text-dark-secondary">
                    Segmenting workforce...
                </div>
            </div>
        )
    }

    if (isError || !data?.success) {
        return (
            <GlassCard className="p-6">
                <div className="text-center text-text-secondary dark:text-text-dark-secondary">
                    <AlertTriangle className="w-8 h-8 mx-auto mb-2 text-amber-500" />
                    <p>Unable to generate workforce segments. Ensure sufficient data is loaded.</p>
                </div>
            </GlassCard>
        )
    }

    const clusters = Object.entries(data.cluster_descriptions).map(([id, desc]) => ({
        id: parseInt(id),
        description: desc,
        count: data.cluster_counts[parseInt(id)] || 0,
        departments: Object.keys(data.top_departments[parseInt(id)] || {}),
        persona: getPersona(desc)
    }))

    return (
        <div className="space-y-6">
            <MemberModal
                isOpen={!!selectedCluster}
                clusterId={selectedCluster?.id ?? null}
                personaName={selectedCluster?.name ?? ''}
                nClusters={actualNClusters}
                onClose={() => setSelectedCluster(null)}
            />

            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-2xl font-display font-semibold text-gray-900 dark:text-white">
                        Active Workforce Personas
                    </h2>
                    <p className="text-text-secondary dark:text-text-dark-secondary text-sm mt-1">
                        AI-identified employee segments based on currently active staff ({data.cluster_counts ? Object.values(data.cluster_counts).reduce((a, b) => a + b, 0) : 0} employees)
                    </p>
                </div>
                <div className="flex flex-col items-end gap-1">
                    {data.silhouette_score && (
                        <Badge variant="outline" className="text-xs">
                            Confidence: {(data.silhouette_score * 100).toFixed(0)}%
                        </Badge>
                    )}
                    {data.excluded_count > 0 && (
                        <p className="text-[10px] text-amber-500 italic">
                            {data.excluded_count} employees excluded due to missing data.
                        </p>
                    )}
                </div>
            </div>

            {/* Persona Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {clusters.map((cluster) => (
                    <GlassCard
                        key={cluster.id}
                        className="p-5 hover:shadow-glow transition-all group relative cursor-pointer active:scale-[0.98]"
                        onClick={() => setSelectedCluster({ id: cluster.id, name: cluster.persona.name })}
                    >
                        <div className="flex items-start gap-4">
                            <div className={cn(
                                "w-12 h-12 rounded-xl flex items-center justify-center",
                                "bg-gradient-to-br from-white/80 to-white/40 dark:from-gray-800/80 dark:to-gray-800/40",
                                "border border-white/20 dark:border-white/10 group-hover:scale-110 transition-transform"
                            )}>
                                <span className={cluster.persona.color}>
                                    {cluster.persona.icon}
                                </span>
                            </div>
                            <div className="flex-1">
                                <div className="flex items-center justify-between">
                                    <h3 className="font-semibold text-gray-900 dark:text-white text-lg">
                                        {cluster.persona.name}
                                    </h3>
                                    <Badge variant="outline" className="text-xs font-medium bg-secondary/10">
                                        {cluster.count} employees
                                    </Badge>
                                </div>
                                <p className="text-sm text-text-secondary dark:text-text-dark-secondary mt-1">
                                    {cluster.persona.description}
                                </p>

                                <div className="mt-4 grid grid-cols-2 gap-y-3 gap-x-4 border-t border-white/10 pt-4">
                                    <div>
                                        <p className="text-[10px] uppercase font-bold text-text-secondary dark:text-text-dark-secondary tracking-wider">
                                            Defining Traits
                                        </p>
                                        <div className="mt-1.5 flex flex-wrap gap-1.5">
                                            {(cluster.description as string).split(', ').map((trait, i) => (
                                                <Badge key={i} variant="outline" className="px-1.5 py-0 text-[10px] bg-white/5">
                                                    {trait}
                                                </Badge>
                                            ))}
                                        </div>
                                    </div>
                                    <div>
                                        <p className="text-[10px] uppercase font-bold text-text-secondary dark:text-text-dark-secondary tracking-wider">
                                            Top Departments
                                        </p>
                                        <p className="text-xs text-text-secondary dark:text-text-dark-secondary mt-1">
                                            {cluster.departments.join(', ') || 'N/A'}
                                        </p>
                                    </div>
                                </div>

                                <div className="mt-4 flex gap-4">
                                    {data.feature_summary['Salary']?.[cluster.id] && (
                                        <div className="flex-1 bg-white/5 rounded-lg p-2 border border-white/5">
                                            <p className="text-[10px] text-text-secondary dark:text-text-dark-secondary">Avg Salary</p>
                                            <p className="text-sm font-semibold">${(data.feature_summary['Salary'][cluster.id] / 1000).toFixed(1)}k</p>
                                        </div>
                                    )}
                                    {data.feature_summary['LastRating']?.[cluster.id] && (
                                        <div className="flex-1 bg-white/5 rounded-lg p-2 border border-white/5">
                                            <p className="text-[10px] text-text-secondary dark:text-text-dark-secondary">Avg Rating</p>
                                            <p className="text-sm font-semibold">{(data.feature_summary['LastRating'][cluster.id]).toFixed(1)} / 5.0</p>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                        <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                            <Badge className="bg-blue-500 text-[9px]">View Members</Badge>
                        </div>
                    </GlassCard>
                ))}
            </div>

            {/* Strategic Insight */}
            <GlassCard className="p-6 bg-gradient-to-r from-emerald-500/10 to-blue-500/10 border-emerald-500/20">
                <div className="flex items-start gap-4">
                    <Sparkles className="w-6 h-6 text-emerald-500 flex-shrink-0 mt-0.5" />
                    <div>
                        <h4 className="font-semibold text-gray-900 dark:text-white text-lg">Actionable Insights</h4>
                        <div className="mt-2 space-y-2 text-sm text-text-secondary dark:text-text-dark-secondary leading-relaxed">
                            <p>
                                I've identified <strong>{clusters.length}</strong> distinct archetypes in your workforce.
                                These segments help you move beyond simple department averages to tailor HR interventions:
                            </p>
                            <ul className="list-disc list-inside space-y-1 ml-2">
                                <li>Invest in <strong>High Impact</strong> segments through leadership development.</li>
                                <li>Focus on <strong>New Talent</strong> with enhanced peer-mentorship programs.</li>
                                <li>Identify groups needing <strong>Compensation Review</strong> to ensure equity.</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </GlassCard>
        </div>
    )
}
