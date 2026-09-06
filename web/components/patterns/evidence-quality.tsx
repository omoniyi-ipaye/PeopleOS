import { Database, ShieldCheck } from 'lucide-react'
import { MetricCard } from '@/components/ui/data-display'
import { StatusBadge } from '@/components/ui/status'

export interface EvidenceQualityProps {
  confidence: number
  coverage: number
  verifiedRuns?: number
  knownGaps?: number
  sufficiency?: 'SUFFICIENT' | 'LIMITED' | 'INSUFFICIENT' | string
}

function percent(value: number) {
  return `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`
}

export function EvidenceQuality({ confidence, coverage, verifiedRuns = 0, knownGaps = 0, sufficiency = 'LIMITED' }: EvidenceQualityProps) {
  const sufficiencyTone = sufficiency === 'SUFFICIENT' ? 'success' : sufficiency === 'INSUFFICIENT' ? 'danger' : 'warning'
  return (
    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4" aria-label="Evidence quality">
      <MetricCard label="Evidence confidence" value={percent(confidence)} detail="Confidence in evidence that was available" icon={<ShieldCheck className="h-4 w-4" />} status={<StatusBadge tone={confidence >= 0.8 ? 'success' : confidence >= 0.6 ? 'warning' : 'danger'}>{confidence >= 0.8 ? 'High' : confidence >= 0.6 ? 'Medium' : 'Low'}</StatusBadge>} />
      <MetricCard label="Evidence coverage" value={percent(coverage)} detail="How much of the planned investigation contributed" icon={<Database className="h-4 w-4" />} status={<StatusBadge tone={coverage >= 0.8 ? 'success' : coverage >= 0.5 ? 'warning' : 'danger'}>{coverage >= 0.8 ? 'Broad' : coverage >= 0.5 ? 'Partial' : 'Thin'}</StatusBadge>} />
      <MetricCard label="Verified tool runs" value={verifiedRuns} detail="Governed tool executions with verification evidence" />
      <MetricCard label="Known gaps" value={knownGaps} detail="Missing evidence PeopleOS will not infer" status={<StatusBadge tone={sufficiencyTone}>{sufficiency.toLowerCase()}</StatusBadge>} />
    </div>
  )
}
