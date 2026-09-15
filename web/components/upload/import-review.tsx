import { AlertTriangle, Check, ChevronDown, CircleHelp, FileCheck2, RefreshCw, Sparkles, UploadCloud } from 'lucide-react'
import { Button, StateSummary, StatusBadge, Surface } from '@/components/ui'
import type { ColumnMappingPreview, UploadPreviewResponse } from '@/types/api'

export type ImportMapping = Record<string, string | null>

interface ImportReviewProps {
  preview: UploadPreviewResponse
  mapping: ImportMapping
  reviewed: boolean
  needsRecheck: boolean
  isReviewing: boolean
  isActivating: boolean
  onMappingChange: (source: string, target: string | null) => void
  onAskLocalAI: () => void
  onRecheck: () => void
  onReviewedChange: (reviewed: boolean) => void
  onCancel: () => void
  onActivate: () => void
}

function toneForMapping(item: ColumnMappingPreview, duplicate: boolean) {
  if (duplicate || item.status === 'unmapped') return 'danger' as const
  if (item.status === 'needs_review') return 'warning' as const
  return 'success' as const
}

function methodLabel(item: ColumnMappingPreview) {
  if (item.method === 'llm') return 'Local AI suggestion'
  if (item.method === 'user_confirmed') return 'Your choice'
  if (item.method === 'similarity') return 'Name similarity'
  if (item.method === 'alias') return 'Recognized alias'
  if (item.method === 'exact') return 'Exact field name'
  if (item.method === 'ambiguous') return 'Needs a choice'
  return 'Not matched'
}

function confidenceLabel(item: ColumnMappingPreview) {
  if (item.method === 'llm') return `${Math.round(item.confidence * 100)}% suggested`
  if (item.method === 'exact' || item.method === 'alias' || item.method === 'user_confirmed') return 'High confidence'
  if (item.method === 'similarity') return `${Math.round(item.confidence * 100)}% similar`
  return 'No safe match'
}

function sampleLabel(item: ColumnMappingPreview) {
  if (!item.sample_values.length) return 'No non-empty examples'
  return item.sample_values.join(' · ')
}

export function ImportReview({
  preview,
  mapping,
  reviewed,
  needsRecheck,
  isReviewing,
  isActivating,
  onMappingChange,
  onAskLocalAI,
  onRecheck,
  onReviewedChange,
  onCancel,
  onActivate,
}: ImportReviewProps) {
  const targetCounts = Object.values(mapping).filter(Boolean).reduce<Record<string, number>>((counts, target) => {
    counts[target as string] = (counts[target as string] ?? 0) + 1
    return counts
  }, {})
  const duplicateTargets = new Set(Object.entries(targetCounts).filter(([, count]) => count > 1).map(([target]) => target))
  const hasMappingProblem = preview.missing_required_fields.length > 0 || duplicateTargets.size > 0
  const activationBlocked = !preview.can_activate || needsRecheck || !reviewed || hasMappingProblem

  return (
    <Surface padding="lg" className="space-y-6 border-violet-200/80 bg-gradient-to-br from-white to-violet-50/30 dark:border-violet-500/20 dark:from-slate-950 dark:to-violet-500/[0.04]" aria-labelledby="import-review-title">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-[0.16em] text-violet-600 dark:text-violet-300">
            <FileCheck2 className="h-4 w-4" aria-hidden="true" />
            Import review · Nothing active yet
          </div>
          <h2 id="import-review-title" className="mt-2 text-2xl font-semibold tracking-tight text-slate-950 dark:text-white">Confirm how PeopleOS should read this file</h2>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-text-secondary">
            {preview.filename} contains {preview.rows_detected.toLocaleString()} source rows. Review the suggestions, make any corrections, then run the integrity check before activation.
          </p>
        </div>
        <StatusBadge tone={preview.can_activate && !hasMappingProblem ? 'success' : 'warning'}>
          {preview.can_activate && !hasMappingProblem ? 'Ready for your review' : 'Needs attention'}
        </StatusBadge>
      </div>

      <div className="grid gap-3 sm:grid-cols-3">
        <div className="rounded-2xl border border-border bg-white/70 p-4 dark:bg-white/[0.03]"><div className="text-xs font-semibold text-text-muted">Source rows</div><div className="mt-1 text-xl font-semibold">{preview.rows_detected.toLocaleString()}</div><div className="mt-1 text-xs text-text-secondary">No rows are activated yet</div></div>
        <div className="rounded-2xl border border-border bg-white/70 p-4 dark:bg-white/[0.03]"><div className="text-xs font-semibold text-text-muted">Fields mapped</div><div className="mt-1 text-xl font-semibold">{preview.mappings.filter(item => item.target).length}/{preview.mappings.length}</div><div className="mt-1 text-xs text-text-secondary">Required fields are marked</div></div>
        <div className="rounded-2xl border border-border bg-white/70 p-4 dark:bg-white/[0.03]"><div className="text-xs font-semibold text-text-muted">Capabilities</div><div className="mt-1 text-xl font-semibold">{Object.values(preview.features_enabled).filter(Boolean).length}</div><div className="mt-1 text-xs text-text-secondary">Based on validated meaning</div></div>
      </div>

      {preview.blocking_issues.length > 0 && (
        <Surface tone="danger" padding="compact" role="alert">
          <div className="flex items-start gap-3"><AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" aria-hidden="true" /><div><h3 className="font-semibold">This file cannot be activated yet</h3><ul className="mt-2 space-y-1 text-sm leading-6">{preview.blocking_issues.map(issue => <li key={issue}>{issue}</li>)}</ul></div></div>
        </Surface>
      )}

      {preview.missing_required_fields.length > 0 && (
        <StateSummary tone="warning" title="Required fields still need a match" description={preview.missing_required_fields.join(', ')} />
      )}

      <div className="flex flex-col gap-3 rounded-2xl border border-sky-200 bg-sky-50/70 p-4 dark:border-sky-500/20 dark:bg-sky-500/[0.06] sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-start gap-3"><CircleHelp className="mt-0.5 h-5 w-5 shrink-0 text-sky-600 dark:text-sky-300" aria-hidden="true" /><div><div className="text-sm font-semibold text-sky-900 dark:text-sky-200">Suggestions are not activation decisions</div><p className="mt-1 text-xs leading-5 text-sky-800/80 dark:text-sky-200/80">PeopleOS will validate the selected fields and data values again. Local AI sees only column names and shape metadata, never workforce cell values.</p>{preview.llm.used && <p className="mt-1 text-xs font-semibold text-sky-800 dark:text-sky-200">Local AI added suggestions. Please review every highlighted choice.</p>}{preview.llm.reason && <p className="mt-1 text-xs text-sky-800/80 dark:text-sky-200/80">{preview.llm.reason}</p>}</div></div>
        {!preview.llm.used && <Button variant="secondary" size="sm" disabled={isReviewing || needsRecheck} isLoading={isReviewing} onClick={onAskLocalAI}><Sparkles className="h-4 w-4" aria-hidden="true" />Ask local AI to suggest</Button>}
      </div>

      <div>
        <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
          <div><h3 className="text-lg font-semibold text-slate-950 dark:text-white">Source fields</h3><p className="mt-1 text-sm text-text-secondary">Choose the PeopleOS meaning for each column. Leave a field as “Not used” when it has no safe meaning.</p></div>
          <div className="text-xs text-text-muted">{preview.available_fields.length} supported meanings</div>
        </div>
        <div className="mt-4 space-y-3">
          {preview.mappings.map(item => {
            const selectedTarget = Object.prototype.hasOwnProperty.call(mapping, item.source)
              ? mapping[item.source] ?? ''
              : item.target ?? ''
            const duplicate = Boolean(selectedTarget && duplicateTargets.has(selectedTarget))
            const tone = toneForMapping(item, duplicate)
            return (
              <div key={item.source} className="rounded-2xl border border-border bg-white/70 p-4 dark:bg-white/[0.03]">
                <div className="grid gap-4 lg:grid-cols-[minmax(150px,0.8fr)_minmax(220px,1.1fr)_minmax(180px,1fr)_auto] lg:items-center">
                  <div className="min-w-0"><div className="truncate text-sm font-semibold text-slate-950 dark:text-white" title={item.source}>{item.source}</div><div className="mt-1 text-xs text-text-muted">{sampleLabel(item)}</div></div>
                  <label className="text-xs font-semibold text-text-muted">PeopleOS field<select aria-label={`Map ${item.source}`} value={selectedTarget} onChange={event => onMappingChange(item.source, event.target.value || null)} className="mt-1 block h-10 w-full rounded-xl border border-border bg-background px-3 text-sm font-medium text-slate-800 shadow-sm focus:border-violet-500 focus:outline-none focus:ring-2 focus:ring-violet-500/30 dark:text-slate-100"><option value="">Not used</option>{preview.available_fields.map(field => <option key={field} value={field}>{field}{preview.missing_required_fields.includes(field) ? ' · required' : ''}</option>)}</select></label>
                  <div className="min-w-0"><div className="flex items-center gap-2 text-xs font-semibold text-text-secondary"><span>{methodLabel(item)}</span><span aria-hidden="true">·</span><span>{confidenceLabel(item)}</span></div>{item.reason && <div className="mt-1 text-xs leading-5 text-text-muted">{item.reason}</div>}{duplicate && <div className="mt-1 text-xs font-semibold text-red-700 dark:text-red-300">Choose this PeopleOS field only once.</div>}</div>
                  <StatusBadge tone={tone}>{item.required ? 'Required' : item.status === 'mapped' ? 'Mapped' : item.status === 'needs_review' ? 'Review' : 'Choose field'}</StatusBadge>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {preview.warnings.length > 0 && <details className="rounded-2xl border border-border bg-background-secondary/70 p-4"><summary className="flex cursor-pointer list-none items-center justify-between gap-3 text-sm font-semibold text-slate-800 dark:text-slate-100"><span>Validation notes ({preview.warnings.length})</span><ChevronDown className="h-4 w-4" aria-hidden="true" /></summary><ul className="mt-3 space-y-2 text-xs leading-5 text-text-secondary">{preview.warnings.map(warning => <li key={warning}>{warning}</li>)}</ul></details>}

      <div className="border-t border-border pt-5">
        <label className="flex items-start gap-3 text-sm text-text-secondary"><input type="checkbox" checked={reviewed} disabled={needsRecheck || isActivating} onChange={event => onReviewedChange(event.target.checked)} className="mt-1 h-4 w-4 rounded border-border text-violet-600 focus:ring-violet-500" /><span><span className="font-semibold text-slate-800 dark:text-slate-100">I have reviewed the field meanings and understand that activation will replace the current workforce source.</span><span className="mt-1 block text-xs leading-5 text-text-muted">PeopleOS will preserve the previous dataset version, but the active analytics and any model must be validated against this new snapshot.</span></span></label>
        {needsRecheck && <div className="mt-3 text-xs font-semibold text-amber-700 dark:text-amber-300">You changed a mapping. Run the integrity check again before activation.</div>}
        <div className="mt-5 flex flex-col-reverse gap-3 sm:flex-row sm:items-center sm:justify-between"><Button variant="ghost" onClick={onCancel} disabled={isActivating || isReviewing}>Choose a different file</Button><div className="flex flex-col gap-3 sm:flex-row"><Button variant="secondary" disabled={isActivating || isReviewing} isLoading={isReviewing} onClick={onRecheck}><RefreshCw className="h-4 w-4" aria-hidden="true" />Recheck data</Button><Button disabled={activationBlocked || isActivating} isLoading={isActivating} onClick={onActivate}><UploadCloud className="h-4 w-4" aria-hidden="true" />Activate validated workforce</Button></div></div>
        {activationBlocked && <p className="mt-3 text-right text-xs text-text-muted">{!preview.can_activate ? 'Resolve the validation issues first.' : hasMappingProblem ? 'Review or correct the highlighted mappings.' : needsRecheck ? 'Recheck the changed mapping.' : 'Confirm that you reviewed the mappings.'}</p>}
        {!activationBlocked && <p className="mt-3 flex items-center justify-end gap-1.5 text-xs font-semibold text-emerald-700 dark:text-emerald-300"><Check className="h-4 w-4" aria-hidden="true" />Ready to create a new governed dataset version</p>}
      </div>
    </Surface>
  )
}
