'use client'

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api-client'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Database, Upload, RefreshCw } from 'lucide-react'
import { UploadStatus } from '@/types/api'

export function Header() {
  const queryClient = useQueryClient()

  const { data: status } = useQuery<UploadStatus>({
    queryKey: ['upload', 'status'],
    queryFn: api.upload.getStatus as any,
  })

  const loadSampleMutation = useMutation({
    mutationFn: api.upload.loadSample,
    onSuccess: () => {
      queryClient.invalidateQueries()
    },
  })

  return (
    <header className="flex items-center justify-between h-16 px-6 bg-surface dark:bg-surface-dark border-b border-border dark:border-border-dark">
      <div className="flex items-center gap-4">
        {status?.has_data ? (
          <div className="flex items-center gap-2">
            <Database className="w-4 h-4 text-success" />
            <span className="text-sm text-text-secondary">
              {status.employee_count.toLocaleString()} employees loaded
            </span>
            {status.features_enabled?.predictive && (
              <Badge variant="default" size="sm">
                Predictive data ready
              </Badge>
            )}
          </div>
        ) : (
          <div className="flex items-center gap-2">
            <Database className="w-4 h-4 text-text-muted" />
            <span className="text-sm text-text-muted">No data loaded</span>
          </div>
        )}
      </div>

      <div className="flex items-center gap-3">
        {!status?.has_data && (
          <Button
            variant="secondary"
            size="sm"
            onClick={() => loadSampleMutation.mutate()}
            isLoading={loadSampleMutation.isPending}
          >
            <Upload className="w-4 h-4 mr-2" />
            Load Sample Data
          </Button>
        )}

        <Button
          variant="ghost"
          size="sm"
          aria-label="Refresh PeopleOS data"
          onClick={() => queryClient.invalidateQueries()}
        >
          <RefreshCw className="w-4 h-4" />
        </Button>
      </div>
    </header>
  )
}
