'use client'

import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api-client'
import { Card } from '@/components/ui/card'

interface ValidationEvidence { status: string; message: string; interpretation: string; metrics: Record<string, number> | null }
interface FeatureDiagnostic { feature: string; importance: number; reliability: number; status: string }

export function ModelLab() {
    const validation = useQuery({ queryKey: ['model-lab', 'validation'], queryFn: () => api.modelLab.getValidation(90) as Promise<ValidationEvidence> })
    const sensitivity = useQuery({ queryKey: ['model-lab', 'sensitivity'], queryFn: () => api.modelLab.getSensitivity() as Promise<FeatureDiagnostic[]> })
    if (validation.isLoading || sensitivity.isLoading) return <p role="status">Loading model evaluation evidence…</p>
    if (validation.isError || sensitivity.isError) return <p role="alert">Model evaluation evidence is unavailable.</p>
    const evidence = validation.data
    return <div className="space-y-6">
        <Card title="Model evaluation evidence" subtitle="Prospective accuracy requires predictions recorded before mature outcomes.">
            <p>{evidence?.message}</p><p className="mt-3 text-sm">{evidence?.interpretation}</p>
            {evidence?.metrics ? <dl>{Object.entries(evidence.metrics).map(([name, value]) => <div key={name}><dt>{name}</dt><dd>{Number.isFinite(value) ? value.toFixed(3) : 'Unavailable'}</dd></div>)}</dl> : <p className="mt-3 font-medium">Prospective accuracy metrics: Unavailable</p>}
        </Card>
        <Card title="Feature diagnostics" subtitle="Variance and correlation rules are descriptive checks; they do not establish an accuracy improvement.">
            {sensitivity.data?.length ? <table className="w-full text-sm"><thead><tr><th scope="col">Feature</th><th scope="col">Model importance</th><th scope="col">Heuristic index</th></tr></thead><tbody>{sensitivity.data.map(row => <tr key={row.feature}><th scope="row">{row.feature}</th><td>{Number.isFinite(row.importance) ? row.importance.toFixed(3) : 'Unavailable'}</td><td>{Number.isFinite(row.reliability) ? row.reliability.toFixed(2) : 'Unavailable'}</td></tr>)}</tbody></table> : <p>No fitted feature diagnostics are available.</p>}
            <p className="mt-3 text-sm">Model changes require independent evaluation before activation.</p>
        </Card>
    </div>
}
