'use client'

import { useCallback, useState } from 'react'
import Link from 'next/link'
import { useDropzone } from 'react-dropzone'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api-client'
import type { UploadStatus, UploadResponse } from '@/types/api'
import { ArrowRight, CheckCircle2, Download, FileUp, Loader2, Sparkles, Trash2 } from 'lucide-react'
import { Button, EmptyState, MetricCard, Page, PageHeader, SectionHeader, StateSummary, StatusBadge, Surface } from '@/components/ui'

export default function DataSourcesPage() {
  const queryClient = useQueryClient()
  const [result, setResult] = useState<UploadResponse | null>(null)

  const { data: status } = useQuery<UploadStatus>({ queryKey: ['upload', 'status'], queryFn: () => api.upload.getStatus() as Promise<UploadStatus> })
  const uploadMutation = useMutation<UploadResponse, Error, File>({ mutationFn: (file) => api.upload.uploadFile(file) as Promise<UploadResponse>, onSuccess: (data) => { setResult(data); void queryClient.resetQueries() } })
  const sampleMutation = useMutation<UploadResponse, Error, void>({ mutationFn: () => api.upload.loadSample() as Promise<UploadResponse>, onSuccess: (data) => { setResult(data); void queryClient.resetQueries() } })
  const resetMutation = useMutation({ mutationFn: () => api.upload.reset(), onSuccess: () => { setResult(null); void queryClient.resetQueries() } })

  const onDrop = useCallback((files: File[]) => { if (files[0]) uploadMutation.mutate(files[0]) }, [uploadMutation])
  const dropzone = useDropzone({ onDrop, accept: { 'text/csv': ['.csv'], 'application/json': ['.json'] }, maxFiles: 1 })
  const hasData = Boolean(status?.has_data)
  const busy = uploadMutation.isPending || sampleMutation.isPending

  return <Page>
    <PageHeader eyebrow="Govern · Data & Sources" title="Know exactly what data PeopleOS is using" description="Validate and activate workforce data without silently training a model. Dataset state, predictive readiness and optional text capability remain separate." />

    {hasData ? <StateSummary title="Dataset active" description={`${status?.employee_count?.toLocaleString() ?? 0} people are available for deterministic workforce analysis.`} tone="success" /> : <StateSummary title="No active dataset" description="Add a workforce dataset or use the sample to start the PeopleOS journey." tone="neutral" />}

    <div className="grid gap-6 lg:grid-cols-[minmax(0,1.15fr)_minmax(320px,0.85fr)]">
      <Surface padding="none" className="overflow-hidden">
        <div {...dropzone.getRootProps()} className={`flex min-h-[360px] cursor-pointer flex-col items-center justify-center p-8 text-center transition ${dropzone.isDragActive ? 'bg-accent/5' : 'hover:bg-background-secondary'}`}>
          <input {...dropzone.getInputProps()} />
          <div className="grid h-14 w-14 place-items-center rounded-2xl bg-accent/10 text-accent">{uploadMutation.isPending ? <Loader2 className="h-6 w-6 animate-spin" /> : <FileUp className="h-6 w-6" />}</div>
          <h2 className="mt-5 text-xl font-semibold">{dropzone.isDragActive ? 'Drop the dataset here' : 'Add workforce data'}</h2>
          <p className="mt-2 max-w-md text-sm leading-6 text-text-secondary">CSV or JSON. PeopleOS validates and activates the source without making model training an upload side effect.</p>
          <Button type="button" className="mt-6">Browse files</Button>
        </div>
      </Surface>

      <Surface padding="lg">
        <SectionHeader title="Dataset lifecycle" description="The sequence stays explicit so readiness claims remain trustworthy." />
        <div className="mt-5 space-y-4">
          <Lifecycle number="1" title="Validate" detail="Check schema, row count, duplicates and missing values." active />
          <Lifecycle number="2" title="Activate dataset" detail="Create a versioned source of truth for deterministic analysis." active />
          <Lifecycle number="3" title="Analyse immediately" detail="Workforce Health and People Intelligence use aggregate evidence." active />
          <Lifecycle number="4" title="Train only when needed" detail="Predictive modelling is a separate governed lifecycle." />
        </div>
      </Surface>
    </div>

    <section className="grid gap-4 md:grid-cols-3">
      <Surface padding="md" className="flex items-center justify-between gap-4"><div><div className="font-semibold">Schema template</div><div className="text-xs text-text-muted">Review required and optional fields</div></div><Button variant="secondary" size="sm" onClick={() => api.upload.downloadTemplate()}><Download className="h-4 w-4" />Download</Button></Surface>
      <Surface padding="md" className="flex items-center justify-between gap-4"><div><div className="font-semibold">Sample dataset</div><div className="text-xs text-text-muted">Explore the product safely</div></div><Button variant="secondary" size="sm" disabled={hasData || busy} isLoading={sampleMutation.isPending} onClick={() => sampleMutation.mutate()}><Sparkles className="h-4 w-4" />Load sample</Button></Surface>
      <Surface padding="md" className="flex items-center justify-between gap-4"><div><div className="font-semibold">Runtime data</div><div className="text-xs text-text-muted">Lifecycle history remains auditable</div></div><Button variant="danger" size="sm" disabled={!hasData || resetMutation.isPending} isLoading={resetMutation.isPending} onClick={() => resetMutation.mutate()}><Trash2 className="h-4 w-4" />Reset</Button></Surface>
    </section>

    {result && <Surface padding="lg">
      <div className="flex items-start gap-4"><div className={`grid h-10 w-10 place-items-center rounded-xl ${result.success ? 'bg-success/10 text-success' : 'bg-danger/10 text-danger'}`}>{result.success ? <CheckCircle2 className="h-5 w-5" /> : <FileUp className="h-5 w-5" />}</div><div className="flex-1"><div className="flex flex-wrap items-center gap-2"><h2 className="font-semibold">{result.success ? 'Dataset activated' : 'Data could not be activated'}</h2><StatusBadge tone={result.success ? 'success' : 'danger'}>{result.success ? 'Active' : 'Failed'}</StatusBadge></div><p className="mt-2 text-sm leading-6 text-text-secondary">{result.message}</p>{result.success && <div className="mt-5 grid gap-3 sm:grid-cols-3"><MetricCard label="Available now" value="Workforce analytics" detail="Deterministic aggregate analysis" /><MetricCard label="Predictive inputs" value={result.features_enabled?.predictive ? 'Ready' : 'More data needed'} detail="Training remains separate" tone={result.features_enabled?.predictive ? 'success' : 'neutral'} /><MetricCard label="Text intelligence" value={result.features_enabled?.nlp ? 'Source present' : 'Optional'} detail="Advanced capability tier" tone={result.features_enabled?.nlp ? 'info' : 'neutral'} /></div>}{result.success && <Link href="/" className="mt-5 inline-flex items-center gap-2 text-sm font-semibold text-accent">Open Decision Cockpit <ArrowRight className="h-4 w-4" /></Link>}</div></div>
    </Surface>}

    {!result && !hasData && <EmptyState title="PeopleOS needs a source of truth" description="A dataset is the prerequisite for every analytical claim. No data means no inferred workforce conclusion." />}
  </Page>
}

function Lifecycle({ number, title, detail, active }: { number: string; title: string; detail: string; active?: boolean }) { return <div className="flex gap-4"><div className={`grid h-9 w-9 shrink-0 place-items-center rounded-full text-sm font-semibold ${active ? 'bg-accent text-white' : 'bg-background-secondary text-text-muted'}`}>{number}</div><div><div className="font-semibold">{title}</div><div className="mt-1 text-sm leading-5 text-text-secondary">{detail}</div></div></div> }
