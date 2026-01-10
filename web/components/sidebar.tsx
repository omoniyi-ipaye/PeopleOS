'use client'

import { useState } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { cn } from '@/lib/utils'
import {
  LayoutDashboard,
  Search,
  Brain,
  Users,
  Upload,
  Settings,
  FolderOpen,
  ChevronLeft,
  ChevronRight,
  ChevronDown,
  ChevronUp,
  Activity,
  UserPlus,
  ShieldAlert,
  BarChartHorizontal,
} from 'lucide-react'

const analyticsNavigation = [
  { name: 'Overview', href: '/', icon: LayoutDashboard },
  { name: 'Workforce Health', href: '/workforce-health', icon: Activity },
  { name: 'Flight Risk', href: '/flight-risk', icon: ShieldAlert },
  { name: 'Retention Forecast', href: '/retention-forecast', icon: BarChartHorizontal },
  { name: 'Quality of Hire', href: '/quality-of-hire', icon: UserPlus },
  { name: 'Review Search', href: '/search', icon: Search },
  { name: 'HR Advisor', href: '/advisor', icon: Brain },
]

const managementNavigation = [
  { name: 'Upload Data', href: '/upload', icon: Upload },
  { name: 'Sessions', href: '/sessions', icon: FolderOpen },
  { name: 'Settings', href: '/settings', icon: Settings },
]

export function Sidebar() {
  const pathname = usePathname()
  const [isCollapsed, setIsCollapsed] = useState(false)
  const [analyticsOpen, setAnalyticsOpen] = useState(true)
  const [managementOpen, setManagementOpen] = useState(true)

  const toggleTheme = () => {
    document.documentElement.classList.toggle('dark')
  }

  return (
    <div className={cn(
      "flex flex-col bg-surface dark:bg-surface-dark border-r border-border dark:border-border-dark transition-all duration-300 relative",
      isCollapsed ? "w-20" : "w-64"
    )}>
      {/* Collapse Toggle Button */}
      <button
        onClick={() => setIsCollapsed(!isCollapsed)}
        className="absolute -right-3 top-8 bg-surface dark:bg-surface-dark border border-border dark:border-border-dark rounded-full p-1 shadow-sm hover:bg-surface-hover dark:hover:bg-surface-dark-hover z-20 text-text-secondary dark:text-text-dark-secondary"
      >
        {isCollapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
      </button>

      {/* Logo */}
      <div className="flex items-center h-16 px-6 border-b border-border dark:border-border-dark overflow-hidden whitespace-nowrap">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 flex-shrink-0 bg-accent rounded-lg flex items-center justify-center">
            <Users className="w-5 h-5 text-white" />
          </div>
          {!isCollapsed && <span className="font-bold text-lg">PeopleOS</span>}
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-6 space-y-4 overflow-x-hidden overflow-y-auto custom-scrollbar">
        {/* Analytics Section */}
        <div>
          {!isCollapsed ? (
            <button
              onClick={() => setAnalyticsOpen(!analyticsOpen)}
              className="w-full flex items-center justify-between px-3 mb-2 text-[10px] font-bold text-text-muted uppercase tracking-widest hover:text-text-secondary transition-colors"
            >
              <span>Insights & Analytics</span>
              {analyticsOpen ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            </button>
          ) : (
            <div className="h-px bg-border dark:bg-border-dark my-4 mx-2 opacity-50" />
          )}

          <div className={cn("space-y-0.5 transition-all duration-300 ease-in-out",
            !isCollapsed && !analyticsOpen ? "max-h-0 opacity-0 overflow-hidden" : "max-h-[500px] opacity-100"
          )}>
            {analyticsNavigation.map((item) => {
              const isActive = pathname === item.href
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={cn(
                    'flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-all group relative',
                    isActive
                      ? 'bg-accent/10 text-accent'
                      : 'text-text-secondary dark:text-text-dark-secondary hover:text-text-primary dark:hover:text-text-dark-primary hover:bg-surface-hover dark:hover:bg-surface-dark-hover'
                  )}
                >
                  <item.icon className={cn("w-5 h-5 flex-shrink-0 transition-transform duration-200", !isActive && "group-hover:scale-110")} />
                  {!isCollapsed && <span>{item.name}</span>}
                  {isCollapsed && (
                    <div className="absolute left-full ml-4 px-2 py-1 bg-slate-900 text-white text-[10px] rounded opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity whitespace-nowrap z-50">
                      {item.name}
                    </div>
                  )}
                </Link>
              )
            })}
          </div>
        </div>

        {/* Divider */}
        {!isCollapsed && <div className="mx-3 h-px bg-border/40 dark:bg-border-dark/40" />}

        {/* Management Section */}
        <div>
          {!isCollapsed ? (
            <button
              onClick={() => setManagementOpen(!managementOpen)}
              className="w-full flex items-center justify-between px-3 mb-2 text-[10px] font-bold text-text-muted uppercase tracking-widest hover:text-text-secondary transition-colors"
            >
              <span>Management & Ops</span>
              {managementOpen ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            </button>
          ) : (
            <div className="h-px bg-border dark:bg-border-dark my-4 mx-2 opacity-50" />
          )}

          <div className={cn("space-y-0.5 transition-all duration-300 ease-in-out",
            !isCollapsed && !managementOpen ? "max-h-0 opacity-0 overflow-hidden" : "max-h-[500px] opacity-100"
          )}>
            {managementNavigation.map((item) => {
              const isActive = pathname === item.href
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={cn(
                    'flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-all group relative',
                    isActive
                      ? 'bg-accent/10 text-accent'
                      : 'text-text-secondary dark:text-text-dark-secondary hover:text-text-primary dark:hover:text-text-dark-primary hover:bg-surface-hover dark:hover:bg-surface-dark-hover'
                  )}
                >
                  <item.icon className={cn("w-5 h-5 flex-shrink-0 transition-transform duration-200", !isActive && "group-hover:scale-110")} />
                  {!isCollapsed && <span>{item.name}</span>}
                  {isCollapsed && (
                    <div className="absolute left-full ml-4 px-2 py-1 bg-slate-900 text-white text-[10px] rounded opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity whitespace-nowrap z-50">
                      {item.name}
                    </div>
                  )}
                </Link>
              )
            })}
          </div>
        </div>
      </nav>

      {/* Footer */}
      <div className="px-4 py-4 border-t border-border dark:border-border-dark space-y-4">
        <button
          onClick={toggleTheme}
          className="flex items-center gap-3 w-full px-3 py-2 rounded-lg text-sm font-medium text-text-secondary dark:text-text-dark-secondary hover:bg-surface-hover dark:hover:bg-surface-dark-hover transition-colors overflow-hidden whitespace-nowrap"
        >
          <div className="dark:hidden flex items-center gap-3 flex-shrink-0">
            <span className="w-5 h-5">🌙</span>
            {!isCollapsed && <span>Dark Mode</span>}
          </div>
          <div className="hidden dark:flex items-center gap-3 flex-shrink-0">
            <span className="w-5 h-5">☀️</span>
            {!isCollapsed && <span>Light Mode</span>}
          </div>
        </button>
        {!isCollapsed && (
          <div className="flex items-center gap-2 text-xs text-text-muted dark:text-text-dark-muted px-3">
            <div className="w-2 h-2 rounded-full bg-success flex-shrink-0" />
            <span className="truncate">Local-first analytics</span>
          </div>
        )}
      </div>
    </div>
  )
}

