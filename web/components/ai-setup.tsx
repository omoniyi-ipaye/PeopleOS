'use client'

import { useMemo, useState, useSyncExternalStore } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { BrainCircuit, CheckCircle2, Download, ExternalLink, Loader2, Power, ShieldCheck, Sparkles, TestTube2, XCircle } from 'lucide-react'
import { api } from '@/lib/api-client'
import { Button, SectionHeader, StatusBadge, Surface } from '@/components/ui'
import type { LLMStatus } from '@/types/api'

type AISetupContext = 'setup' | 'settings'

function errorMessage(error: unknown, fallback: string) {
  return error instanceof Error ? error.message : fallback
}

function formatBytes(value?: number | null) {
  if (!value || value < 1) return ''
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let amount = value
  let unit = 0
  while (amount >= 1024 && unit < units.length - 1) {
    amount /= 1024
    unit += 1
  }
  return `${amount >= 10 || unit === 0 ? amount.toFixed(0) : amount.toFixed(1)} ${units[unit]}`
}

function setupIsRunning(state?: string) {
  return state === 'starting' || state === 'pulling' || state === 'testing'
}

export function AISetup({ context = 'setup' }: { context?: AISetupContext }) {
  const queryClient = useQueryClient()
  const [modelChoice, setModelChoice] = useState<string | null>(null)
  const browserSupport = useSyncExternalStore(
    () => () => undefined,
    () => ((navigator as Navigator & { gpu?: unknown }).gpu ? 'supported' : 'unavailable'),
    () => 'checking',
  ) as 'checking' | 'supported' | 'unavailable'
  const llmQuery = useQuery<LLMStatus>({
    queryKey: ['llm', 'status'],
    queryFn: api.llm.getStatus,
    retry: false,
    refetchInterval: query => setupIsRunning(query.state.data?.setup_state) ? 1500 : false,
  })
  const setup = useMutation({
    mutationFn: (model: string) => api.llm.setup(model),
    onSuccess: next => {
      setModelChoice(next.setup_model ?? next.selected_model)
      queryClient.setQueryData(['llm', 'status'], next)
      void queryClient.invalidateQueries({ queryKey: ['platform', 'status'] })
      void queryClient.invalidateQueries({ queryKey: ['api', 'status'] })
      void queryClient.invalidateQueries({ queryKey: ['upload', 'status'] })
    },
  })
  const configure = useMutation({
    mutationFn: (request: { provider: 'none' | 'ollama'; enabled: boolean; model?: string }) => api.llm.configure(request),
    onSuccess: next => {
      setModelChoice(next.selected_model)
      queryClient.setQueryData(['llm', 'status'], next)
      void queryClient.invalidateQueries({ queryKey: ['platform', 'status'] })
      void queryClient.invalidateQueries({ queryKey: ['api', 'status'] })
      void queryClient.invalidateQueries({ queryKey: ['upload', 'status'] })
    },
  })
  const test = useMutation({ mutationFn: api.llm.test })

  const status = llmQuery.data
  const selectedModel = modelChoice ?? (status ? status.selected_model_installed ? status.selected_model : status.recommended_model : '')
  const setupRunning = setupIsRunning(status?.setup_state)
  const selectedModelInstalled = Boolean(status?.installed_models.some(model => model.name === selectedModel))
  const modelOptions = useMemo(() => {
    if (!status) return []
    const options = status.installed_models.map(model => model)
    if (selectedModel && !options.some(model => model.name === selectedModel)) {
      options.unshift({ name: selectedModel, remote: false })
    }
    return options
  }, [selectedModel, status])

  function startSetup() {
    if (selectedModel) setup.mutate(selectedModel)
  }

  function toggleAI(enabled: boolean) {
    if (!selectedModel) return
    if (enabled && !selectedModelInstalled) {
      setup.mutate(selectedModel)
      return
    }
    configure.mutate({ provider: enabled ? 'ollama' : 'none', enabled, model: selectedModel })
  }

  if (llmQuery.isLoading) {
    return <Surface padding="lg" aria-busy="true"><div className="flex items-center gap-3 text-sm text-text-secondary"><Loader2 className="h-4 w-4 animate-spin text-accent" />Checking local AI capability…</div></Surface>
  }

  if (llmQuery.isError || !status) {
    return <Surface tone="warning" padding="lg"><SectionHeader title="Local AI setup is unavailable" description="PeopleOS could not read the local AI service state. Deterministic workforce analysis remains available." /><p role="alert" className="text-sm leading-6 text-amber-900 dark:text-amber-200">{errorMessage(llmQuery.error, 'The local AI status check failed.')}</p></Surface>
  }

  const statusTone = status.ready ? 'success' : !status.enabled && status.selected_model_installed ? 'neutral' : setupRunning ? 'info' : status.ollama_installed ? 'warning' : 'neutral'
  const statusLabel = status.ready ? `Ready · ${status.selected_model}` : !status.enabled && status.selected_model_installed ? `Off · ${status.selected_model}` : setupRunning ? (status.setup_message ?? 'Setting up local AI…') : status.ollama_installed ? 'Not ready yet' : 'Ollama not installed'

  return <div className="space-y-4">
    <Surface padding="lg" className="border-violet-200/70 dark:border-violet-500/20">
      <SectionHeader
        title={context === 'setup' ? 'Optional local AI assistance' : 'Local AI assistance'}
        description="Use a model running on this computer to organise already-verified PeopleOS evidence. Calculations, permissions and evidence boundaries do not depend on the model."
        action={<StatusBadge tone={statusTone}>{statusLabel}</StatusBadge>}
      />

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(280px,0.72fr)]">
        <div className="space-y-5">
          <label className="flex cursor-pointer items-start gap-3 rounded-2xl border border-slate-200 p-4 transition hover:border-violet-300 dark:border-white/10 dark:hover:border-violet-500/30">
            <input
              type="checkbox"
              className="mt-1 h-4 w-4 accent-violet-600"
              checked={status.enabled}
              disabled={setupRunning || configure.isPending}
              onChange={event => toggleAI(event.target.checked)}
            />
            <span className="min-w-0">
              <span className="flex items-center gap-2 text-sm font-semibold text-slate-950 dark:text-white"><Power className="h-4 w-4 text-violet-600 dark:text-violet-300" />Use local AI assistance</span>
              <span className="mt-1 block text-xs leading-5 text-slate-600 dark:text-slate-400">Off keeps PeopleOS in deterministic mode. Turning it on is an owner preference stored on this computer.</span>
            </span>
          </label>

          <div>
            <label htmlFor="peopleos-local-model" className="text-sm font-semibold text-slate-950 dark:text-white">Local model</label>
            <select
              id="peopleos-local-model"
              value={selectedModel}
              disabled={setupRunning || configure.isPending || setup.isPending}
              onChange={event => setModelChoice(event.target.value)}
              className="mt-2 block w-full rounded-xl border border-slate-200 bg-white px-3 py-2.5 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-violet-500 dark:border-white/10 dark:bg-slate-950 dark:text-white"
            >
              {modelOptions.map(model => <option key={model.name} value={model.name}>{model.name}{model.size ? ` · ${formatBytes(model.size)}` : ''}</option>)}
              {!modelOptions.length && <option value={selectedModel}>{selectedModel || 'No local model selected'}</option>}
            </select>
            <p className="mt-2 text-xs leading-5 text-slate-500 dark:text-slate-400">Only models present in this computer&apos;s local Ollama catalog are shown. Cloud-tagged models are excluded.</p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Button variant="primary" disabled={setupRunning || setup.isPending || !selectedModel} isLoading={setup.isPending} onClick={startSetup}>
              <Sparkles className="h-4 w-4" />{status.ready && selectedModel === status.selected_model ? 'Re-test local model' : status.selected_model_installed ? 'Enable and test model' : 'Set up automatically'}
            </Button>
            <Button variant="secondary" disabled={!status.ready || setupRunning || test.isPending} isLoading={test.isPending} onClick={() => test.mutate()}>
              <TestTube2 className="h-4 w-4" />Test local AI
            </Button>
          </div>
          {setupRunning && <div role="status" aria-live="polite" className="rounded-xl border border-sky-200 bg-sky-50/70 p-3 text-xs text-sky-900 dark:border-sky-500/20 dark:bg-sky-500/[0.06] dark:text-sky-200"><div className="flex items-center justify-between gap-3"><span>{status.setup_message ?? 'Preparing local AI…'}</span><span>{status.setup_progress}%</span></div><div className="mt-2 h-2 overflow-hidden rounded-full bg-sky-100 dark:bg-sky-500/20"><div className="h-full rounded-full bg-sky-500 transition-all" style={{ width: `${Math.max(0, Math.min(100, status.setup_progress))}%` }} /></div></div>}
          {setup.isError && <p role="alert" className="text-sm text-red-700 dark:text-red-300">{errorMessage(setup.error, 'The local AI setup could not start.')}</p>}
          {status.setup_state === 'error' && <p role="alert" className="text-sm text-red-700 dark:text-red-300">{status.setup_message ?? 'Local AI setup did not finish.'}</p>}
          {configure.isError && <p role="alert" className="text-sm text-red-700 dark:text-red-300">{errorMessage(configure.error, 'The local AI preference could not be saved.')}</p>}
          {test.data && <div role="status" className={`rounded-xl border p-3 text-xs leading-5 ${test.data.passed ? 'border-emerald-200 bg-emerald-50/70 text-emerald-900 dark:border-emerald-500/20 dark:bg-emerald-500/[0.06] dark:text-emerald-200' : 'border-amber-200 bg-amber-50/70 text-amber-900 dark:border-amber-500/20 dark:bg-amber-500/[0.06] dark:text-amber-200'}`}><div className="flex items-center gap-2 font-semibold">{test.data.passed ? <CheckCircle2 className="h-4 w-4" /> : <XCircle className="h-4 w-4" />}{test.data.passed ? 'Local model passed the readiness test' : 'Local model responded, but did not pass the readiness test'}</div><div className="mt-1">{test.data.model} · {test.data.elapsed_ms.toLocaleString()} ms · no workforce data was sent</div></div>}
          {test.isError && <p role="alert" className="text-sm text-red-700 dark:text-red-300">{errorMessage(test.error, 'The local model test failed.')}</p>}
        </div>

        <div className="rounded-2xl border border-slate-200/80 bg-slate-50 p-4 dark:border-white/10 dark:bg-white/[0.03]">
          <div className="flex items-start gap-3"><ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-violet-600 dark:text-violet-300" /><div><div className="text-sm font-semibold text-slate-950 dark:text-white">What setup does</div><p className="mt-1 text-xs leading-5 text-slate-600 dark:text-slate-400">PeopleOS checks Ollama, starts its local service if needed, downloads the chosen model only when it is missing, then sends a fixed READY prompt without workforce data.</p></div></div>
          {!status.ollama_installed ? <div className="mt-4 rounded-xl border border-amber-200 bg-amber-50 p-3 text-xs leading-5 text-amber-900 dark:border-amber-500/20 dark:bg-amber-500/[0.06] dark:text-amber-200"><div className="font-semibold">Install Ollama first</div><p className="mt-1">Download the official local runtime, then return here. PeopleOS does not run an operating-system installer silently.</p><a className="mt-2 inline-flex items-center gap-1 font-semibold underline" href={status.download_guide} target="_blank" rel="noreferrer">Open Ollama download guide <ExternalLink className="h-3 w-3" /></a></div> : <div className="mt-4 text-xs leading-5 text-slate-600 dark:text-slate-400">{status.ollama_running ? 'Ollama is running on this computer.' : 'Ollama is installed but not running; automatic setup can start it.'}{status.reason && !status.ready ? ` ${status.reason}` : ''}</div>}
        </div>
      </div>
    </Surface>

    <Surface padding="lg" className="border-slate-200 dark:border-white/10">
      <SectionHeader title="Browser AI (WebLLM) · experimental" description="A separate browser-only option, not the current governed PeopleOS advisor." action={<StatusBadge tone={browserSupport === 'supported' ? 'info' : browserSupport === 'unavailable' ? 'neutral' : 'neutral'}>{browserSupport === 'checking' ? 'Checking browser' : browserSupport === 'supported' ? 'WebGPU detected' : 'WebGPU unavailable'}</StatusBadge>} />
      <div className="grid gap-4 md:grid-cols-[1fr_auto] md:items-center"><div className="flex items-start gap-3"><BrainCircuit className="mt-0.5 h-4 w-4 shrink-0 text-sky-600 dark:text-sky-300" /><p className="text-sm leading-6 text-slate-600 dark:text-slate-400">WebLLM can download and run an open model inside a compatible browser using WebGPU. The model is cached by the browser and inference does not use the PeopleOS FastAPI service. It needs a separate privacy and evidence boundary before it can handle workforce records, so this release keeps it informational rather than silently routing employee data into a browser model.</p></div><a className="inline-flex items-center gap-2 text-sm font-semibold text-violet-600 dark:text-violet-300" href="https://github.com/mlc-ai/web-llm" target="_blank" rel="noreferrer">Learn about WebLLM <Download className="h-4 w-4" /></a></div>
    </Surface>
  </div>
}
