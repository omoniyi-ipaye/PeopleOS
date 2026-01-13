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
  Heart,
  GitBranch,
  TrendingUp,
  UserCheck,
  Lightbulb,
  Wrench,
} from 'lucide-react'

// Reorganized into HR-friendly categories
const navigationSections = [
  {
    id: 'dashboard',
    label: 'Dashboard',
    items: [
      { name: 'Overview', href: '/', icon: LayoutDashboard },
    ],
  },
  {
    id: 'people',
    label: 'People Analytics',
    description: 'Understand your workforce',
    items: [
      { name: 'Workforce Health', href: '/workforce-health', icon: Activity },
      { name: 'Employee Experience', href: '/employee-experience', icon: Heart },
      { name: 'Flight Risk', href: '/flight-risk', icon: ShieldAlert },
    ],
  },
  {
    id: 'planning',
    label: 'Strategic Planning',
    description: 'Plan for the future',
    items: [
      { name: 'Scenario Planner', href: '/scenario-planner', icon: GitBranch },
      { name: 'Retention Forecast', href: '/retention-forecast', icon: BarChartHorizontal },
    ],
  },
  {
    id: 'talent',
    label: 'Talent Management',
    description: 'Build your talent pipeline',
    items: [
      { name: 'Quality of Hire', href: '/quality-of-hire', icon: UserPlus },
    ],
  },
  {
    id: 'tools',
    label: 'AI Tools',
    description: 'AI-powered insights',
    items: [
      { name: 'HR Advisor', href: '/advisor', icon: Brain },
      { name: 'Review Search', href: '/search', icon: Search },
    ],
  },
]

const managementNavigation = [
  { name: 'Upload Data', href: '/upload', icon: Upload },
  { name: 'Sessions', href: '/sessions', icon: FolderOpen },
  { name: 'Settings', href: '/settings', icon: Settings },
]

export function Sidebar() {
  const pathname = usePathname()
  const [isCollapsed, setIsCollapsed] = useState(false)
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    dashboard: true,
    people: true,
    planning: true,
    talent: true,
    tools: true,
  })
  const [managementOpen, setManagementOpen] = useState(true)

  const toggleSection = (sectionId: string) => {
    setExpandedSections(prev => ({
      ...prev,
      [sectionId]: !prev[sectionId]
    }))
  }

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
      <nav className="flex-1 px-3 py-4 space-y-1 overflow-x-hidden overflow-y-auto custom-scrollbar">
        {/* Main Navigation Sections */}
        {navigationSections.map((section) => (
          <div key={section.id} className="mb-2">
            {!isCollapsed ? (
              <button
                onClick={() => toggleSection(section.id)}
                className="w-full flex items-center justify-between px-3 py-1.5 text-[10px] font-bold text-text-muted uppercase tracking-widest hover:text-text-secondary transition-colors rounded-lg hover:bg-surface-hover/50"
              >
                <span>{section.label}</span>
                {expandedSections[section.id] ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
              </button>
            ) : (
              section.id !== 'dashboard' && <div className="h-px bg-border dark:bg-border-dark my-2 mx-2 opacity-50" />
            )}

            <div className={cn("space-y-0.5 transition-all duration-300 ease-in-out",
              !isCollapsed && !expandedSections[section.id] ? "max-h-0 opacity-0 overflow-hidden" : "max-h-[500px] opacity-100"
            )}>
              {section.items.map((item) => {
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
        ))}

        {/* Divider */}
        {!isCollapsed && <div className="mx-3 h-px bg-border/40 dark:bg-border-dark/40 my-2" />}

        {/* Management Section */}
        <div>
          {!isCollapsed ? (
            <button
              onClick={() => setManagementOpen(!managementOpen)}
              className="w-full flex items-center justify-between px-3 py-1.5 text-[10px] font-bold text-text-muted uppercase tracking-widest hover:text-text-secondary transition-colors rounded-lg hover:bg-surface-hover/50"
            >
              <span>Data & Settings</span>
              {managementOpen ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            </button>
          ) : (
            <div className="h-px bg-border dark:bg-border-dark my-2 mx-2 opacity-50" />
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
