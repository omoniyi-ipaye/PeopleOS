'use client'

import { useQuery } from '@tanstack/react-query'
import { usePathname } from 'next/navigation'
import { Header } from '@/components/header'
import { Sidebar } from '@/components/sidebar'
import { api } from '@/lib/api-client'
import { AppLockScreen, type AppLockStatus } from '@/components/app-lock'

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
  const appLock = useQuery<AppLockStatus>({
    queryKey: ['app-lock', 'status'],
    queryFn: api.appLock.getStatus,
    retry: false,
  })
  const setupRoute = isSetupRoute(pathname, status)

  if (appLock.isLoading) {
    return <main className="grid min-h-screen place-items-center bg-slate-50 px-4 dark:bg-slate-950"><div className="text-sm text-slate-500 dark:text-slate-400">Checking PeopleOS lock…</div></main>
  }

  if (appLock.isError) {
    return <main className="grid min-h-screen place-items-center bg-slate-50 px-4 dark:bg-slate-950"><div role="alert" className="max-w-md rounded-2xl border border-amber-200 bg-white p-6 text-center text-sm text-amber-800 dark:border-amber-500/20 dark:bg-slate-900 dark:text-amber-200"><p>PeopleOS could not verify its local lock state. The app is unavailable until the local configuration can be checked safely.</p><button type="button" onClick={() => void appLock.refetch()} className="mt-4 rounded-xl bg-slate-950 px-4 py-2 text-sm font-semibold text-white transition hover:bg-slate-800 dark:bg-white dark:text-slate-950 dark:hover:bg-slate-100">Retry connection</button></div></main>
  }

  if (appLock.data?.locked) {
    return <main className="min-h-screen overflow-y-auto bg-slate-50 px-4 py-6 dark:bg-slate-950 md:px-8 md:py-10"><AppLockScreen /></main>
  }

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
