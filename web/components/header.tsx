'use client'

import Link from 'next/link'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Database, RefreshCw, ShieldCheck } from 'lucide-react'
import { api } from '@/lib/api-client'
import { Button } from '@/components/ui/button'

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

  return (
    <header className="flex min-h-14 items-center justify-between gap-4 border-b border-slate-200/70 bg-white/85 px-5 backdrop-blur-xl dark:border-white/10 dark:bg-slate-950/85 md:px-7" aria-label="PeopleOS context">
      <div className="flex min-w-0 items-center gap-2 text-xs text-text-muted">
        <Database className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
        <span className="truncate">{hasData ? `${status?.data?.row_count?.toLocaleString() ?? 0} people in active data` : 'No active dataset'}</span>
        {hasData && <><span aria-hidden="true">·</span><span className="hidden sm:inline">{activeModel ? 'predictive model active' : 'evidence-first analysis'}</span></>}
      </div>
      <div className="flex items-center gap-1">
        <Link href="/platform" className="inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-medium text-text-muted transition hover:bg-background-secondary hover:text-text-primary">
          <ShieldCheck className="h-3.5 w-3.5" aria-hidden="true" />Trust
        </Link>
        <Button variant="ghost" size="icon" aria-label="Refresh PeopleOS context" onClick={() => queryClient.invalidateQueries()}>
          <RefreshCw className="h-4 w-4" aria-hidden="true" />
        </Button>
      </div>
    </header>
  )
}
