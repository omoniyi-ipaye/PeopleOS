'use client'

import { useState } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { cn } from '@/lib/utils'
import {
  Activity,
  BarChartHorizontal,
  Brain,
  ChevronLeft,
  ChevronRight,
  Database,
  FileClock,
  GitBranch,
  Heart,
  Home,
  Search,
  Settings,
  ShieldCheck,
  Sparkles,
  UserPlus,
  Users,
} from 'lucide-react'

const sections = [
  {
    label: 'Understand',
    items: [
      { name: 'Decision Cockpit', href: '/', icon: Home },
      { name: 'Workforce Health', href: '/workforce-health', icon: Activity },
      { name: 'Employee Experience', href: '/employee-experience', icon: Heart },
      { name: 'Retention Signals', href: '/flight-risk', icon: BarChartHorizontal },
      { name: 'Quality of Hire', href: '/quality-of-hire', icon: UserPlus },
    ],
  },
  {
    label: 'Investigate',
    items: [
      { name: 'People Intelligence', href: '/advisor', icon: Brain },
      { name: 'Research', href: '/search', icon: Search },
      { name: 'Saved Investigations', href: '/sessions', icon: FileClock },
    ],
  },
  {
    label: 'Plan',
    items: [
      { name: 'Scenario Planner', href: '/scenario-planner', icon: GitBranch },
      { name: 'Retention Forecast', href: '/retention-forecast', icon: Sparkles },
    ],
  },
  {
    label: 'Govern',
    items: [
      { name: 'Data & Sources', href: '/upload', icon: Database },
      { name: 'Trust Center', href: '/platform', icon: ShieldCheck },
      { name: 'Settings', href: '/settings', icon: Settings },
    ],
  },
]

export function Sidebar() {
  const pathname = usePathname()
  const [collapsed, setCollapsed] = useState(false)

  return (
    <aside
      className={cn(
        'relative z-40 flex h-screen shrink-0 flex-col border-r border-slate-200/80 bg-white/95 shadow-sm backdrop-blur-xl transition-all duration-300 dark:border-white/10 dark:bg-slate-950/95',
        collapsed ? 'w-[76px]' : 'w-[260px]'
      )}
    >
      <button
        type="button"
        aria-label={collapsed ? 'Expand navigation' : 'Collapse navigation'}
        onClick={() => setCollapsed((value) => !value)}
        className="absolute -right-3 top-7 z-50 grid h-7 w-7 place-items-center rounded-full border border-slate-200 bg-white text-slate-500 shadow-sm transition hover:text-violet-600 dark:border-white/10 dark:bg-slate-900 dark:text-slate-400"
      >
        {collapsed ? <ChevronRight className="h-3.5 w-3.5" /> : <ChevronLeft className="h-3.5 w-3.5" />}
      </button>

      <div className="flex h-20 items-center gap-3 border-b border-slate-200/70 px-5 dark:border-white/10">
        <div className="grid h-10 w-10 shrink-0 place-items-center rounded-2xl bg-gradient-to-br from-violet-600 to-indigo-600 shadow-lg shadow-violet-600/20">
          <Users className="h-5 w-5 text-white" />
        </div>
        {!collapsed && (
          <div className="min-w-0">
            <div className="text-lg font-bold tracking-tight text-slate-950 dark:text-white">PeopleOS</div>
            <div className="truncate text-[11px] font-medium text-slate-500 dark:text-slate-400">People intelligence operating system</div>
          </div>
        )}
      </div>

      <nav className="flex-1 overflow-y-auto px-3 py-5">
        <div className="space-y-6">
          {sections.map((section) => (
            <section key={section.label}>
              {!collapsed && (
                <div className="mb-2 px-3 text-[10px] font-bold uppercase tracking-[0.18em] text-slate-400 dark:text-slate-500">
                  {section.label}
                </div>
              )}
              <div className="space-y-1">
                {section.items.map((item) => {
                  const active = pathname === item.href
                  return (
                    <Link
                      key={item.href}
                      href={item.href}
                      title={collapsed ? item.name : undefined}
                      className={cn(
                        'group flex min-h-10 items-center gap-3 rounded-xl px-3 text-sm font-medium transition',
                        active
                          ? 'bg-violet-50 text-violet-700 ring-1 ring-violet-100 dark:bg-violet-500/10 dark:text-violet-300 dark:ring-violet-500/20'
                          : 'text-slate-600 hover:bg-slate-100 hover:text-slate-950 dark:text-slate-400 dark:hover:bg-white/5 dark:hover:text-white',
                        collapsed && 'justify-center px-0'
                      )}
                    >
                      <item.icon className={cn('h-[18px] w-[18px] shrink-0', active && 'text-violet-600 dark:text-violet-300')} />
                      {!collapsed && <span className="truncate">{item.name}</span>}
                    </Link>
                  )
                })}
              </div>
            </section>
          ))}
        </div>
      </nav>

      <div className="border-t border-slate-200/70 p-4 dark:border-white/10">
        <div className={cn('rounded-2xl bg-slate-50 p-3 dark:bg-white/[0.04]', collapsed && 'px-2')}>
          <div className={cn('flex items-center gap-2', collapsed && 'justify-center')}>
            <span className="relative flex h-2.5 w-2.5 shrink-0">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-40" />
              <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-emerald-500" />
            </span>
            {!collapsed && <span className="text-xs font-medium text-slate-600 dark:text-slate-300">Governed local workspace</span>}
          </div>
        </div>
      </div>
    </aside>
  )
}
