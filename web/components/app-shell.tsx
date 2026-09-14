'use client'

import { useQuery } from '@tanstack/react-query'
import { usePathname } from 'next/navigation'
import { Header } from '@/components/header'
import { Sidebar } from '@/components/sidebar'
import { api } from '@/lib/api-client'

interface PlatformStatus {
  data?: { loaded?: boolean }
}

export function isSetupRoute(pathname: string | null, status?: PlatformStatus) {
  return pathname === '/' && status?.data?.loaded !== true
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname()
  const { data: status } = useQuery<PlatformStatus>({
    queryKey: ['platform', 'status'],
    queryFn: () => api.getStatus() as Promise<PlatformStatus>,
  })
  const setupRoute = isSetupRoute(pathname, status)

  if (setupRoute) {
    return <main className="min-h-screen overflow-y-auto bg-slate-50 px-4 py-6 dark:bg-slate-950 md:px-8 md:py-10">{children}</main>
  }

  return <div className="flex h-screen overflow-hidden">
    <Sidebar />
    <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
      <Header />
      <main className="flex-1 overflow-y-auto px-4 py-5 md:px-7 md:py-7">
        {children}
      </main>
    </div>
  </div>
}
