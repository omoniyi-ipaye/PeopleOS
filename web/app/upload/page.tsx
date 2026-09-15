'use client'

import { useCallback, useState } from 'react'
import Link from 'next/link'
import { useDropzone } from 'react-dropzone'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api-client'
import type { UploadPreviewResponse, UploadResponse, UploadStatus } from '@/types/api'
import { ArrowRight, CheckCircle2, Download, FileSpreadsheet, FileUp, Loader2, Sparkles, Trash2 } from 'lucide-react'
import { Button, EmptyState, MetricCard, Page, PageHeader, StateSummary, Surface, TrustDisclosure } from '@/components/ui'
import { ImportReview, type ImportMapping } from '@/components/upload/import-review'

interface PreviewRequest {
  file: File
  mapping?: ImportMapping
  useLLM?: boolean
}

export default function DataSourcesPage() {
  const queryClient = useQueryClient()
  const [result, setResult] = useState<UploadResponse | null>(null)
  const [preview, setPreview] = useState<UploadPreviewResponse | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [mapping, setMapping] = useState<ImportMapping>({})
  const [reviewed, setReviewed] = useState(false)
  const [needsRecheck, setNeedsRecheck] = useState(false)
  const [annualPay, setAnnualPay] = useState(false)
  const [currency, setCurrency] = useState('')
  const [confirmReset, setConfirmReset] = useState(false)

  const { data: status, isLoading: statusLoading, isError: statusError, refetch: retryStatus } = useQuery<UploadStatus>({
    queryKey: ['upload', 'status'],
    queryFn: () => api.upload.getStatus() as Promise<UploadStatus>,
  })

  const previewMutation = useMutation<UploadPreviewResponse, Error, PreviewRequest>({
    mutationFn: ({ file, mapping: selectedMapping, useLLM }) => api.upload.previewFile(file, {
      annual: annualPay,
      currency,
      mapping: selectedMapping,
      useLLM,
    }) as Promise<UploadPreviewResponse>,
    onSuccess: data => {
      setPreview(data)
      setMapping(Object.fromEntries(data.mappings.map(item => [item.source, item.target ?? null])))
      setReviewed(false)
      setNeedsRecheck(false)
    },
  })

  const activateMutation = useMutation<UploadResponse, Error, { file: File; mapping: ImportMapping }>({
    mutationFn: ({ file, mapping: selectedMapping }) => api.upload.uploadFile(file, { annual: annualPay, currency }, selectedMapping) as Promise<UploadResponse>,
    onSuccess: data => {
      setResult(data)
      setPreview(null)
      setSelectedFile(null)
      setMapping({})
      setReviewed(false)
      setNeedsRecheck(false)
      setAnnualPay(false)
      setCurrency('')
      setConfirmReset(false)
      void queryClient.invalidateQueries()
    },
  })

  const sampleMutation = useMutation<UploadResponse, Error, void>({
    mutationFn: () => api.upload.loadSample() as Promise<UploadResponse>,
    onSuccess: data => {
      setResult(data)
      void queryClient.invalidateQueries()
    },
  })

  const resetMutation = useMutation({
    mutationFn: () => api.upload.reset(),
    onSuccess: () => {
      setResult(null)
      setConfirmReset(false)
      void queryClient.invalidateQueries()
    },
  })

  const hasData = Boolean(status?.has_data)
  const busy = previewMutation.isPending || activateMutation.isPending || sampleMutation.isPending || resetMutation.isPending
  const startPreview = useCallback((file: File) => {
    setSelectedFile(file)
    setPreview(null)
    setMapping({})
    setReviewed(false)
    setNeedsRecheck(false)
    setResult(null)
    previewMutation.mutate({ file })
  }, [previewMutation])
  const onDrop = useCallback((files: File[]) => {
    if (files[0]) startPreview(files[0])
  }, [startPreview])
  const dropzone = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/json': ['.json'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
    },
    maxFiles: 1,
    disabled: statusLoading || statusError || busy,
  })
  const mutationError = previewMutation.error || activateMutation.error || sampleMutation.error || resetMutation.error

  function changeMapping(source: string, target: string | null) {
    setMapping(current => ({ ...current, [source]: target }))
    setReviewed(false)
    setNeedsRecheck(true)
  }

  function changePayDeclaration(change: () => void) {
    change()
    if (preview) {
      setReviewed(false)
      setNeedsRecheck(true)
    }
  }

  function cancelPreview() {
    setPreview(null)
    setSelectedFile(null)
    setMapping({})
    setReviewed(false)
    setNeedsRecheck(false)
    setResult(null)
    previewMutation.reset()
  }

  function askLocalAI() {
    if (selectedFile) previewMutation.mutate({ file: selectedFile, useLLM: true })
  }

  function recheck() {
    if (selectedFile) previewMutation.mutate({ file: selectedFile, mapping })
  }

  function activate() {
    if (selectedFile) activateMutation.mutate({ file: selectedFile, mapping })
  }

  return <Page>
    <PageHeader
      eyebrow="Data"
      title={hasData ? 'Your workforce data' : 'Bring your workforce into PeopleOS'}
      description={hasData ? 'See what PeopleOS is using, replace it when needed, or return to your insights.' : 'Drop in an Excel, CSV or JSON export. PeopleOS will review the meaning and validate it before using any number in an analysis.'}
    />

    {statusLoading ? <StateSummary title="Checking your current data" description="PeopleOS is confirming the active workforce source." tone="info" /> : statusError ? <EmptyState title="Your data source could not be checked" description="Retry before adding or replacing workforce data." action={<Button onClick={() => void retryStatus()}>Retry</Button>} /> : null}
    {mutationError && <div role="alert"><StateSummary title="We couldn&apos;t complete that import step" description={mutationError instanceof Error ? mutationError.message : 'Check the file and try again.'} tone="warning" /></div>}
    {result && <Surface padding="lg" role="status" aria-live="polite">
      <div className="flex items-start gap-4"><div className={`grid h-11 w-11 place-items-center rounded-xl ${result.success ? 'bg-emerald-50 text-emerald-600 dark:bg-emerald-500/10' : 'bg-red-50 text-red-600'}`}>{result.success ? <CheckCircle2 className="h-5 w-5" /> : <FileUp className="h-5 w-5" />}</div><div className="flex-1"><h2 className="text-lg font-semibold">{result.success ? 'Import complete' : 'This file needs attention'}</h2><p className="mt-2 text-sm leading-6 text-text-secondary">{result.message}</p>{result.success && <div className="mt-5 grid gap-3 sm:grid-cols-3"><MetricCard label="Records ready" value={result.rows_loaded.toLocaleString()} detail="Validated workforce rows" /><MetricCard label="Workforce insights" value="Ready" detail="Available immediately" tone="success" /><MetricCard label="Predictive insights" value={result.features_enabled?.predictive ? 'Eligible to train' : 'Not needed'} detail="Separate optional step" tone="neutral" /></div>}{result.success && <Link href="/" className="mt-5 inline-flex items-center gap-2 text-sm font-semibold text-accent">Open my workforce <ArrowRight className="h-4 w-4" /></Link>}</div></div>
    </Surface>}

    {hasData ? <>
      <Surface padding="lg" className="flex flex-col gap-5 sm:flex-row sm:items-center sm:justify-between">
        <div className="min-w-0"><div className="flex items-center gap-2"><CheckCircle2 className="h-5 w-5 text-emerald-600" /><h2 className="text-lg font-semibold">Your workforce is ready</h2></div><p className="mt-2 text-sm text-text-secondary">{status?.employee_count?.toLocaleString() ?? 'Your'} employee records are available for analysis{status?.reporting_currency ? ` · pay reported in ${status.reporting_currency}` : ''}.</p><div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-text-muted"><span>Active snapshot</span>{status?.source_name && <span className="max-w-full truncate" title={status.source_name}>Source: <span className="font-medium text-text-secondary">{status.source_name}</span></span>}{status?.dataset_version != null && <span>Dataset v{status.dataset_version}</span>}</div></div>
        <div className="flex flex-wrap gap-2"><Link href="/" className="inline-flex items-center gap-2 rounded-xl bg-violet-600 px-4 py-2.5 text-sm font-semibold text-white">Open Home <ArrowRight className="h-4 w-4" /></Link><Link href="/advisor" className="inline-flex items-center gap-2 rounded-xl border border-border px-4 py-2.5 text-sm font-semibold">Ask PeopleOS</Link></div>
      </Surface>
      {status?.features_enabled?.compensation === false && <StateSummary title="Pay insights are currently off" description="Your workforce analysis still works. To enable pay insights, replace the file with annual pay in one shared currency, or confirm those units below when uploading." tone="info" />}
    </> : <Surface padding="lg" className="overflow-hidden border-violet-200/70 bg-gradient-to-br from-white to-violet-50/40 dark:border-violet-500/20 dark:from-slate-950 dark:to-violet-500/[0.03]">
      <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_320px] lg:items-center">
        <div><div className="text-xs font-bold uppercase tracking-[0.16em] text-violet-600 dark:text-violet-300">First time here?</div><h2 className="mt-2 text-2xl font-semibold tracking-tight">Try PeopleOS before adding anything.</h2><p className="mt-3 max-w-2xl text-sm leading-7 text-text-secondary">Load a fictional workforce and explore the full experience safely. Nothing from the sample represents a real person.</p><Button className="mt-5" disabled={busy || statusError || Boolean(selectedFile)} isLoading={sampleMutation.isPending} onClick={() => sampleMutation.mutate()}><Sparkles className="h-4 w-4" />Explore with sample data</Button></div>
        <div className="rounded-2xl border border-violet-100 bg-white/80 p-5 dark:border-violet-500/20 dark:bg-slate-950/60"><div className="text-sm font-semibold">What you&apos;ll be able to do</div><div className="mt-3 space-y-2 text-sm text-text-secondary"><div>✓ See workforce patterns</div><div>✓ Ask natural-language questions</div><div>✓ Inspect evidence when you want it</div><div>✓ Explore scenarios without changing data</div></div></div>
      </div>
    </Surface>}

    {preview && selectedFile ? <ImportReview
      preview={preview}
      mapping={mapping}
      reviewed={reviewed}
      needsRecheck={needsRecheck}
      isReviewing={previewMutation.isPending}
      isActivating={activateMutation.isPending}
      onMappingChange={changeMapping}
      onAskLocalAI={askLocalAI}
      onRecheck={recheck}
      onReviewedChange={setReviewed}
      onCancel={cancelPreview}
      onActivate={activate}
    /> : <div className="grid gap-6 lg:grid-cols-[minmax(0,1.2fr)_minmax(300px,0.8fr)]">
      <Surface padding="none" className="overflow-hidden">
        <div {...dropzone.getRootProps()} aria-disabled={statusLoading || statusError || busy} className={`flex min-h-[390px] cursor-pointer flex-col items-center justify-center p-8 text-center transition ${dropzone.isDragActive ? 'bg-violet-50 dark:bg-violet-500/[0.04]' : 'hover:bg-background-secondary'}`}>
          <input {...dropzone.getInputProps({ 'aria-label': 'Choose workforce data file' })} />
          <div className="grid h-16 w-16 place-items-center rounded-2xl bg-violet-50 text-violet-600 dark:bg-violet-500/10 dark:text-violet-300">{previewMutation.isPending ? <Loader2 className="h-7 w-7 animate-spin" /> : <FileSpreadsheet className="h-7 w-7" />}</div>
          <h2 className="mt-5 text-xl font-semibold">{dropzone.isDragActive ? 'Drop your workforce file here' : hasData ? 'Replace workforce data' : 'Add your workforce data'}</h2>
          <p className="mt-2 max-w-md text-sm leading-6 text-text-secondary">Excel, CSV or JSON. PeopleOS will suggest field meanings, show you the interpretation, and validate the file before activation.</p>
          <Button type="button" className="mt-6" disabled={statusLoading || statusError || busy}><FileUp className="h-4 w-4" />Choose file</Button>
          <div className="mt-4 text-xs text-text-muted">.xlsx · .csv · .json</div>
        </div>
      </Surface>

      <div className="space-y-4">
        <Surface padding="lg"><h2 className="text-sm font-semibold">PeopleOS checks before activation</h2><ol className="mt-4 space-y-3 text-sm text-text-secondary"><li className="flex items-start gap-3"><span className="grid h-5 w-5 shrink-0 place-items-center rounded-full bg-violet-100 text-xs font-bold text-violet-700 dark:bg-violet-500/15 dark:text-violet-300">1</span><span>Suggests meanings for familiar HR column names</span></li><li className="flex items-start gap-3"><span className="grid h-5 w-5 shrink-0 place-items-center rounded-full bg-violet-100 text-xs font-bold text-violet-700 dark:bg-violet-500/15 dark:text-violet-300">2</span><span>Lets you correct ambiguous mappings</span></li><li className="flex items-start gap-3"><span className="grid h-5 w-5 shrink-0 place-items-center rounded-full bg-violet-100 text-xs font-bold text-violet-700 dark:bg-violet-500/15 dark:text-violet-300">3</span><span>Checks employee identifiers, values, pay units and data quality</span></li><li className="flex items-start gap-3"><span className="grid h-5 w-5 shrink-0 place-items-center rounded-full bg-violet-100 text-xs font-bold text-violet-700 dark:bg-violet-500/15 dark:text-violet-300">4</span><span>Activates only the workforce you explicitly review</span></li></ol></Surface>
        <Surface padding="lg"><div className="text-sm font-semibold">Need a template?</div><p className="mt-2 text-sm leading-6 text-text-secondary">Use the PeopleOS template when your export does not contain enough recognisable fields.</p><Button variant="secondary" size="sm" className="mt-4" onClick={() => api.upload.downloadTemplate()}><Download className="h-4 w-4" />Download template</Button></Surface>
      </div>
    </div>}

    <TrustDisclosure title="Pay units" summary="Only needed for compensation insights">
      <p>PeopleOS does not guess pay periods or convert currencies. If your file already includes annual PayPeriod/PayFrequency and one shared Currency, no action is needed.</p>
      <div className="mt-4 grid gap-4 sm:grid-cols-2">
        <label className="flex items-start gap-3 text-sm"><input type="checkbox" checked={annualPay} disabled={busy} onChange={event => changePayDeclaration(() => setAnnualPay(event.target.checked))} className="mt-1 h-4 w-4" /><span>All monetary pay values in this file are annual amounts.</span></label>
        <label className="text-sm">Shared reporting currency<input aria-label="Shared reporting currency" value={currency} disabled={busy} maxLength={3} onChange={event => changePayDeclaration(() => setCurrency(event.target.value.toUpperCase()))} placeholder="e.g. EUR" className="mt-2 block w-full rounded-lg border border-border bg-background p-3" /></label>
      </div>
    </TrustDisclosure>

    {hasData && <Surface padding="compact" className="border-red-200/80 bg-red-50/40 dark:border-red-500/20 dark:bg-red-500/[0.04]"><div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between"><div><h2 className="text-sm font-semibold text-red-900 dark:text-red-200">Clear active workforce</h2><p className="mt-1 text-xs leading-5 text-red-800/80 dark:text-red-200/80">This removes the current snapshot from active analysis until another workforce is activated. Previous dataset versions remain in local history.</p></div>{confirmReset ? <div className="flex flex-wrap items-center gap-2"><Button variant="ghost" size="sm" disabled={resetMutation.isPending} onClick={() => setConfirmReset(false)}>Cancel</Button><Button variant="danger" size="sm" disabled={statusLoading || statusError || busy || Boolean(selectedFile)} isLoading={resetMutation.isPending} onClick={() => resetMutation.mutate()}><Trash2 className="h-4 w-4" />Confirm clear</Button></div> : <Button variant="danger" size="sm" disabled={statusLoading || statusError || busy || Boolean(selectedFile)} onClick={() => setConfirmReset(true)}><Trash2 className="h-4 w-4" />Clear active workforce</Button>}</div></Surface>}
  </Page>
}
