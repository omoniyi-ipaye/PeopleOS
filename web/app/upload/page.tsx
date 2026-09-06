'use client'

import { useCallback, useState } from 'react'
import Link from 'next/link'
import { useDropzone } from 'react-dropzone'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api-client'
import type { UploadStatus, UploadResponse } from '@/types/api'
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Download,
  FileUp,
  Loader2,
  Sparkles,
  Trash2,
} from 'lucide-react'

export default function DataSourcesPage() {
  const queryClient = useQueryClient()
  const [result, setResult] = useState<UploadResponse | null>(null)

  const { data: status } = useQuery<UploadStatus>({
    queryKey: ['upload', 'status'],
    queryFn: () => api.upload.getStatus() as Promise<UploadStatus>,
  })

  const uploadMutation = useMutation<UploadResponse, Error, File>({
    mutationFn: (file) => api.upload.uploadFile(file) as Promise<UploadResponse>,
    onSuccess: (data) => {
      setResult(data)
      queryClient.invalidateQueries()
    },
  })

  const sampleMutation = useMutation<UploadResponse, Error, void>({
    mutationFn: () => api.upload.loadSample() as Promise<UploadResponse>,
    onSuccess: (data) => {
      setResult(data)
      queryClient.invalidateQueries()
    },
  })

  const resetMutation = useMutation({
    mutationFn: () => api.upload.reset(),
    onSuccess: () => {
      setResult(null)
      queryClient.invalidateQueries()
    },
  })

  const onDrop = useCallback((files: File[]) => {
    if (files[0]) uploadMutation.mutate(files[0])
  }, [uploadMutation])

  const dropzone = useDropzone({
    onDrop,
    accept: { 'text/csv': ['.csv'], 'application/json': ['.json'] },
    maxFiles: 1,
  })

  const busy = uploadMutation.isPending || sampleMutation.isPending
  const hasData = Boolean(status?.has_data)

  return (
    <div className="mx-auto max-w-6xl space-y-6 pb-10">
      <section>
        <div className="mb-2 text-xs font-bold uppercase tracking-[0.18em] text-violet-600 dark:text-violet-300">Govern · Data & sources</div>
        <h1 className="text-3xl font-semibold tracking-tight text-slate-950 dark:text-white">Make the data state obvious.</h1>
        <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600 dark:text-slate-400">
          Activate a workforce dataset first. PeopleOS will tell you what is available immediately, what needs a trained model, and what requires richer source data.
        </p>
      </section>

      {hasData && (
        <section className="flex flex-col gap-4 rounded-[28px] border border-emerald-200 bg-emerald-50/70 p-5 dark:border-emerald-500/20 dark:bg-emerald-500/[0.05] md:flex-row md:items-center md:justify-between">
          <div className="flex items-start gap-3">
            <div className="grid h-10 w-10 place-items-center rounded-xl bg-emerald-100 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-300"><CheckCircle2 className="h-5 w-5" /></div>
            <div><div className="font-semibold text-slate-900 dark:text-white">Dataset active</div><div className="mt-1 text-sm text-slate-600 dark:text-slate-400">{status?.employee_count?.toLocaleString() ?? 0} people are available for deterministic workforce analysis.</div></div>
          </div>
          <button type="button" onClick={() => resetMutation.mutate()} disabled={resetMutation.isPending} className="inline-flex items-center justify-center gap-2 rounded-xl border border-red-200 bg-white px-4 py-2.5 text-sm font-semibold text-red-600 transition hover:bg-red-50 disabled:opacity-50 dark:border-red-500/20 dark:bg-transparent dark:text-red-300">
            {resetMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />} Reset runtime data
          </button>
        </section>
      )}

      <section className="grid gap-6 lg:grid-cols-[minmax(0,1.15fr)_minmax(320px,0.85fr)]">
        <div {...dropzone.getRootProps()} className={`group flex min-h-[360px] cursor-pointer flex-col items-center justify-center rounded-[30px] border-2 border-dashed p-8 text-center transition ${dropzone.isDragActive ? 'border-violet-500 bg-violet-50 dark:bg-violet-500/[0.05]' : 'border-slate-300 bg-white hover:border-violet-300 hover:bg-violet-50/30 dark:border-white/15 dark:bg-slate-900 dark:hover:border-violet-500/30'}`}>
          <input {...dropzone.getInputProps()} />
          <div className="grid h-16 w-16 place-items-center rounded-2xl bg-violet-100 text-violet-600 transition group-hover:scale-105 dark:bg-violet-500/10 dark:text-violet-300">{uploadMutation.isPending ? <Loader2 className="h-7 w-7 animate-spin" /> : <FileUp className="h-7 w-7" />}</div>
          <h2 className="mt-5 text-xl font-semibold text-slate-950 dark:text-white">{dropzone.isDragActive ? 'Drop the dataset here' : 'Add workforce data'}</h2>
          <p className="mt-2 max-w-md text-sm leading-6 text-slate-600 dark:text-slate-400">CSV or JSON. PeopleOS validates and activates the dataset without silently training a model.</p>
          <div className="mt-6 rounded-xl bg-slate-950 px-4 py-2.5 text-sm font-semibold text-white dark:bg-white dark:text-slate-950">Browse files</div>
        </div>

        <div className="space-y-4">
          <LifecycleStep number="1" title="Validate" detail="Check schema, row count, duplicates and missing values." active />
          <LifecycleStep number="2" title="Activate dataset" detail="Create a versioned source of truth for analysis." active />
          <LifecycleStep number="3" title="Analyse immediately" detail="Workforce Health and People Intelligence use deterministic aggregate evidence." active />
          <LifecycleStep number="4" title="Train a model only when needed" detail="Predictive risk becomes a separate governed lifecycle, not an upload side effect." />
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-2">
        <button type="button" onClick={() => api.upload.downloadTemplate()} className="flex items-center justify-between rounded-2xl border border-slate-200 bg-white p-5 text-left transition hover:border-violet-200 hover:bg-violet-50/30 dark:border-white/10 dark:bg-slate-900 dark:hover:border-violet-500/20">
          <div className="flex items-center gap-3"><div className="grid h-10 w-10 place-items-center rounded-xl bg-slate-100 text-slate-600 dark:bg-white/5 dark:text-slate-300"><Download className="h-5 w-5" /></div><div><div className="font-semibold">Download schema template</div><div className="mt-1 text-xs text-slate-500 dark:text-slate-400">See required and optional fields</div></div></div><ArrowRight className="h-4 w-4 text-slate-400" />
        </button>
        <button type="button" disabled={hasData || busy} onClick={() => sampleMutation.mutate()} className="flex items-center justify-between rounded-2xl border border-slate-200 bg-white p-5 text-left transition hover:border-violet-200 hover:bg-violet-50/30 disabled:cursor-not-allowed disabled:opacity-50 dark:border-white/10 dark:bg-slate-900 dark:hover:border-violet-500/20">
          <div className="flex items-center gap-3"><div className="grid h-10 w-10 place-items-center rounded-xl bg-violet-100 text-violet-600 dark:bg-violet-500/10 dark:text-violet-300">{sampleMutation.isPending ? <Loader2 className="h-5 w-5 animate-spin" /> : <Sparkles className="h-5 w-5" />}</div><div><div className="font-semibold">Load sample dataset</div><div className="mt-1 text-xs text-slate-500 dark:text-slate-400">Explore the full journey safely</div></div></div><ArrowRight className="h-4 w-4 text-slate-400" />
        </button>
      </section>

      {result && (
        <section className={`rounded-[28px] border p-6 ${result.success ? 'border-emerald-200 bg-white dark:border-emerald-500/20 dark:bg-slate-900' : 'border-red-200 bg-red-50 dark:border-red-500/20 dark:bg-red-500/[0.05]'}`}>
          <div className="flex items-start gap-3">
            <div className={`grid h-10 w-10 place-items-center rounded-xl ${result.success ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-300' : 'bg-red-100 text-red-700 dark:bg-red-500/10 dark:text-red-300'}`}>{result.success ? <CheckCircle2 className="h-5 w-5" /> : <AlertTriangle className="h-5 w-5" />}</div>
            <div className="flex-1">
              <h2 className="font-semibold text-slate-950 dark:text-white">{result.success ? 'Dataset activated' : 'Data could not be activated'}</h2>
              <p className="mt-1 text-sm leading-6 text-slate-600 dark:text-slate-400">{result.message}</p>
              {result.success && <div className="mt-5 grid gap-3 sm:grid-cols-3"><Capability label="Available now" value="Workforce analytics" good /><Capability label="Predictive inputs" value={result.features_enabled?.predictive ? 'Ready for training' : 'More data needed'} good={Boolean(result.features_enabled?.predictive)} /><Capability label="Text intelligence" value={result.features_enabled?.nlp ? 'Source data present' : 'Optional'} good={Boolean(result.features_enabled?.nlp)} /></div>}
              {result.success && <Link href="/" className="mt-5 inline-flex items-center gap-2 text-sm font-semibold text-violet-600 dark:text-violet-300">Open Decision Cockpit <ArrowRight className="h-4 w-4" /></Link>}
            </div>
          </div>
        </section>
      )}
    </div>
  )
}

function LifecycleStep({ number, title, detail, active }: { number: string; title: string; detail: string; active?: boolean }) {
  return <div className="flex gap-4 rounded-2xl border border-slate-200 bg-white p-4 dark:border-white/10 dark:bg-slate-900"><div className={`grid h-9 w-9 shrink-0 place-items-center rounded-full text-sm font-bold ${active ? 'bg-violet-600 text-white' : 'bg-slate-100 text-slate-500 dark:bg-white/5 dark:text-slate-400'}`}>{number}</div><div><div className="font-semibold text-slate-900 dark:text-white">{title}</div><div className="mt-1 text-sm leading-5 text-slate-500 dark:text-slate-400">{detail}</div></div></div>
}

function Capability({ label, value, good }: { label: string; value: string; good?: boolean }) {
  return <div className="rounded-2xl bg-slate-50 p-4 dark:bg-white/[0.04]"><div className="text-[11px] font-bold uppercase tracking-[0.14em] text-slate-400">{label}</div><div className="mt-2 flex items-center gap-2 text-sm font-semibold text-slate-800 dark:text-slate-200"><span className={`h-2 w-2 rounded-full ${good ? 'bg-emerald-500' : 'bg-slate-300 dark:bg-slate-600'}`} />{value}</div></div>
}
