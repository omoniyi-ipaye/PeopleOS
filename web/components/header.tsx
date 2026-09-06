'use client'

import { useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api-client'
import { Button } from '@/components/ui/button'
import { Database, RefreshCw, ShieldCheck, Sparkles } from 'lucide-react'

interface PlatformStatus {
  status: string
  data?: { loaded?: boolean; row_count?: number; active_dataset?: boolean }
  capabilities?: { predictive_model?: boolean; llm?: boolean; people_intelligence?: boolean }
  workspace?: { active_dataset?: boolean; active_model?: boolean; dataset_versions?: number; model_versions?: number }
}

export function Header() {
  const queryClient = useQueryClient()

  const { data: status } = useQuery<PlatformStatus>({
    queryKey: ['platform', 'status'],
    queryFn: () => api.getStatus() as Promise<PlatformStatus>,
    refetchInterval: 30_000,
  })

  const hasData = Boolean(status?.data?.loaded)
  const activeModel = Boolean(status?.workspace?.active_model)
  const llm = Boolean(status?.capabilities?.llm)

  return (
    <header className="flex min-h-16 items-center justify-between gap-4 border-b border-slate-200/80 bg-white/90 px-5 backdrop-blur-xl dark:border-white/10 dark:bg-slate-950/90 md:px-7">
      <div className="flex min-w-0 flex-wrap items-center gap-2">
        <div className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 shadow-sm dark:border-white/10 dark:bg-white/[0.04] dark:text-slate-300">
          <Database className={hasData ? 'h-3.5 w-3.5 text-emerald-500' : 'h-3.5 w-3.5 text-slate-400'} />
          {hasData ? `${status?.data?.row_count?.toLocaleString() ?? 0} people` : 'No active dataset'}
        </div>
        <div className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 shadow-sm dark:border-white/10 dark:bg-white/[0.04] dark:text-slate-300">
          <Sparkles className={activeModel ? 'h-3.5 w-3.5 text-violet-500' : 'h-3.5 w-3.5 text-slate-400'} />
          {activeModel ? 'Model active' : hasData ? 'Model not active' : 'Model unavailable'}
        </div>
        <div className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 shadow-sm dark:border-white/10 dark:bg-white/[0.04] dark:text-slate-300">
          <ShieldCheck className="h-3.5 w-3.5 text-emerald-500" />
          {llm ? 'Local AI available' : 'Deterministic fallback'}
        </div>
      </div>

      <Button variant="ghost" size="sm" aria-label="Refresh PeopleOS context" onClick={() => queryClient.invalidateQueries()}>
        <RefreshCw className="h-4 w-4" />
      </Button>
    </header>
  )
}
