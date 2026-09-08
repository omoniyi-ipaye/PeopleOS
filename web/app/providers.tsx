'use client'

import { QueryClient, QueryClientProvider, useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'
import { api } from '@/lib/api-client'

function SnapshotObserver() {
  const client = useQueryClient()
  const previous = useRef<string | null | undefined>(undefined)
  const { data } = useQuery({
    queryKey: ['platform', 'status'],
    queryFn: () => api.getStatus() as Promise<{integrity?: {snapshot?: {generation: string}}}>,
    refetchInterval: 15000,
    refetchOnWindowFocus: true,
  })
  useEffect(() => {
    if (!data) return
    const generation = data.integrity?.snapshot?.generation ?? null
    if (previous.current !== undefined && previous.current !== generation) {
      // Clear old values as well as refetching: a failed new query must not retain
      // the previous workforce's measurements in a different dataset context.
      void client.resetQueries({predicate: (query) => !(query.queryKey[0] === 'platform' && query.queryKey[1] === 'status')})
    }
    previous.current = generation
  }, [data, client])
  return null
}

export function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 60 * 1000, // 1 minute
            refetchOnWindowFocus: false,
          },
        },
      })
  )

  return (
    <QueryClientProvider client={queryClient}>
      <SnapshotObserver />
      {children}
    </QueryClientProvider>
  )
}
