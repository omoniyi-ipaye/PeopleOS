'use client'

import { useRef, useState } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { cn } from '@/lib/utils'
import {
  Brain,
  ChevronLeft,
  ChevronRight,
  Database,
  Home,
  LayoutGrid,
  Menu,
  Settings,
  ShieldCheck,
  SlidersHorizontal,
  Users,
  X,
} from 'lucide-react'

const primaryItems = [
  { name: 'Home', href: '/', icon: Home, matches: ['/'] },
  { name: 'Ask PeopleOS', href: '/advisor', icon: Brain, matches: ['/advisor', '/sessions'] },
  { name: 'Insights', href: '/insights', icon: LayoutGrid, matches: ['/insights', '/workforce-health', '/employee-experience', '/flight-risk', '/quality-of-hire', '/retention-forecast', '/search'] },
  { name: 'Plan', href: '/scenario-planner', icon: SlidersHorizontal, matches: ['/scenario-planner'] },
  { name: 'Data', href: '/upload', icon: Database, matches: ['/upload'] },
]

const secondaryItems = [
  { name: 'Trust & Privacy', href: '/platform', icon: ShieldCheck },
  { name: 'Settings', href: '/settings', icon: Settings },
]

function isActive(pathname: string | null, matches: string[]) {
  if (!pathname) return false
  return matches.some((match) => match === '/' ? pathname === '/' : pathname === match || pathname.startsWith(`${match}/`))
}

function ItemLink({ item, pathname, collapsed, onNavigate }: { item: { name: string; href: string; icon: typeof Home; matches?: string[] }; pathname: string | null; collapsed: boolean; onNavigate?: () => void }) {
  const active = item.matches ? isActive(pathname, item.matches) : pathname === item.href
  return <Link href={item.href} onClick={onNavigate} aria-label={item.name} aria-current={active ? 'page' : undefined} title={collapsed ? item.name : undefined} className={cn('group flex min-h-11 items-center gap-3 rounded-xl px-3 text-sm font-medium transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500', active ? 'bg-violet-50 text-violet-700 ring-1 ring-violet-100 dark:bg-violet-500/10 dark:text-violet-300 dark:ring-violet-500/20' : 'text-slate-600 hover:bg-slate-100 hover:text-slate-950 dark:text-slate-400 dark:hover:bg-white/5 dark:hover:text-white', collapsed && 'justify-center px-0')}><item.icon aria-hidden="true" className={cn('h-[18px] w-[18px] shrink-0', active && 'text-violet-600 dark:text-violet-300')} />{!collapsed && <span className="truncate">{item.name}</span>}</Link>
}

function NavigationLinks({ pathname, collapsed = false, onNavigate }: { pathname: string | null; collapsed?: boolean; onNavigate?: () => void }) {
  return <div className="flex min-h-0 flex-1 flex-col">
    <nav aria-label="Main navigation" className="flex-1 overflow-y-auto px-3 py-5">
      <div className="space-y-1">{primaryItems.map(item => <ItemLink key={item.href} item={item} pathname={pathname} collapsed={collapsed} onNavigate={onNavigate} />)}</div>
    </nav>
    <div className="border-t border-slate-200/70 px-3 py-4 dark:border-white/10">
      {!collapsed && <div className="mb-2 px-3 text-[10px] font-bold uppercase tracking-[0.16em] text-slate-400 dark:text-slate-500">PeopleOS</div>}
      <div className="space-y-1">{secondaryItems.map(item => <ItemLink key={item.href} item={item} pathname={pathname} collapsed={collapsed} onNavigate={onNavigate} />)}</div>
    </div>
  </div>
}

export function Sidebar() {
  const pathname = usePathname()
  const [collapsed, setCollapsed] = useState(false)
  const mobileNavigation = useRef<HTMLDialogElement | null>(null)
  return <>
    <button type="button" aria-label="Open navigation" aria-haspopup="dialog" aria-controls="mobile-navigation" onClick={() => mobileNavigation.current?.showModal()} className="absolute left-3 top-2 z-50 grid h-10 w-10 place-items-center rounded-xl text-slate-700 hover:bg-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500 dark:text-slate-200 dark:hover:bg-white/10 md:hidden"><Menu className="h-5 w-5" aria-hidden="true" /></button>
    <dialog ref={mobileNavigation} id="mobile-navigation" aria-label="PeopleOS navigation" onClick={event => { if (event.target === event.currentTarget) mobileNavigation.current?.close() }} className="m-0 h-dvh max-h-none w-[88vw] max-w-[320px] border-r border-slate-200 bg-white p-0 text-slate-950 shadow-xl backdrop:bg-slate-950/50 dark:border-white/10 dark:bg-slate-950 dark:text-white">
      <div className="flex h-full flex-col"><div className="flex h-16 shrink-0 items-center justify-between border-b border-slate-200 px-5 dark:border-white/10"><div><div className="text-lg font-bold">PeopleOS</div><div className="text-[11px] text-slate-500 dark:text-slate-400">Your local People Analyst</div></div><button type="button" aria-label="Close navigation" onClick={() => mobileNavigation.current?.close()} className="grid h-11 w-11 place-items-center rounded-xl hover:bg-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500 dark:hover:bg-white/10"><X className="h-5 w-5" aria-hidden="true" /></button></div><NavigationLinks pathname={pathname} onNavigate={() => mobileNavigation.current?.close()} /></div>
    </dialog>
    <aside className={cn('relative z-40 hidden h-screen shrink-0 flex-col border-r border-slate-200/70 bg-white/95 backdrop-blur-xl transition-all duration-300 dark:border-white/10 dark:bg-slate-950/95 md:flex', collapsed ? 'w-[76px]' : 'w-[248px]')}>
      <button type="button" aria-label={collapsed ? 'Expand navigation' : 'Collapse navigation'} onClick={() => setCollapsed(value => !value)} className="absolute -right-3 top-7 z-50 grid h-7 w-7 place-items-center rounded-full border border-slate-200 bg-white text-slate-500 shadow-sm transition hover:text-violet-600 dark:border-white/10 dark:bg-slate-900 dark:text-slate-400">{collapsed ? <ChevronRight className="h-3.5 w-3.5" /> : <ChevronLeft className="h-3.5 w-3.5" />}</button>
      <div className="flex h-20 items-center gap-3 border-b border-slate-200/60 px-5 dark:border-white/10"><div className="grid h-10 w-10 shrink-0 place-items-center rounded-2xl bg-violet-600 shadow-lg shadow-violet-600/15"><Users className="h-5 w-5 text-white" /></div>{!collapsed && <div className="min-w-0"><div className="text-lg font-bold tracking-tight text-slate-950 dark:text-white">PeopleOS</div><div className="truncate text-[11px] font-medium text-slate-500 dark:text-slate-400">Your local People Analyst</div></div>}</div>
      <NavigationLinks pathname={pathname} collapsed={collapsed} />
    </aside>
  </>
}
