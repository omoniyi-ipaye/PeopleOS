'use client'

import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Database, RefreshCw, ShieldCheck, Sparkles } from 'lucide-react'
import { api } from '@/lib/api-client'
import { Button } from '@/components/ui/button'
import { StatusBadge } from '@/components/ui/status'

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
    <header className="flex min-h-16 items-center justify-between gap-4 border-b border-slate-200/80 bg-white/90 px-5 backdrop-blur-xl dark:border-white/10 dark:bg-slate-950/90 md:px-7" aria-label="PeopleOS context">
      <div className="flex min-w-0 flex-wrap items-center gap-2">
        <StatusBadge tone={hasData ? 'success' : 'warning'}>
          <Database className="h-3.5 w-3.5" aria-hidden="true" />
          {hasData ? `${status?.data?.row_count?.toLocaleString() ?? 0} people` : 'No active dataset'}
        </StatusBadge>
        <StatusBadge tone={activeModel ? 'success' : hasData ? 'warning' : 'neutral'}>
          <Sparkles className="h-3.5 w-3.5" aria-hidden="true" />
          {activeModel ? 'Model active' : hasData ? 'Model not active' : 'Model unavailable'}
        </StatusBadge>
        <StatusBadge tone={llm ? 'info' : 'neutral'}>
          <ShieldCheck className="h-3.5 w-3.5" aria-hidden="true" />
          {llm ? 'Local AI available' : 'Deterministic fallback'}
        </StatusBadge>
      </div>
      <Button variant="ghost" size="icon" aria-label="Refresh PeopleOS context" onClick={() => queryClient.invalidateQueries()}>
        <RefreshCw className="h-4 w-4" aria-hidden="true" />
      </Button>
    </header>
  )
}
