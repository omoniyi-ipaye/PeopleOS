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
  Sparkles
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
      { name: 'PeopleOS Research', href: '/search', icon: Search },
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
      "flex flex-col h-screen transition-all duration-300 relative z-50 bg-white/80 dark:bg-black/60 backdrop-blur-xl border-r border-white/20 dark:border-white/5",
      isCollapsed ? "w-20" : "w-64"
    )}>
      {/* Collapse Toggle Button */}
      <button
        onClick={() => setIsCollapsed(!isCollapsed)}
        className="absolute -right-3 top-8 bg-surface dark:bg-black border border-white/20 dark:border-white/10 rounded-full p-1.5 shadow-lg hover:shadow-glow z-20 text-text-secondary dark:text-text-dark-secondary transition-all hover:scale-110"
      >
        {isCollapsed ? <ChevronRight className="w-3.5 h-3.5" /> : <ChevronLeft className="w-3.5 h-3.5" />}
      </button>

      {/* Logo */}
      <div className="flex items-center h-20 px-6 border-b border-white/10 overflow-hidden shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 flex-shrink-0 bg-gradient-to-br from-accent-400 to-accent-600 rounded-xl shadow-glow flex items-center justify-center relative group">
            <div className="absolute inset-0 bg-white/20 rounded-xl animate-pulse-slow opacity-0 group-hover:opacity-100 transition-opacity" />
            <Users className="w-5 h-5 text-white" />
          </div>
          {!isCollapsed && (
            <div className="flex flex-col animate-in fade-in slide-in-from-left-2 duration-300">
              <span className="font-display font-bold text-xl text-gradient tracking-tight">PeopleOS</span>
              <span className="text-[10px] text-text-muted dark:text-text-dark-muted font-medium tracking-wider uppercase whitespace-normal leading-3 max-w-[160px]">
                HR Intelligence Operating System
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-6 space-y-6 overflow-x-hidden overflow-y-auto custom-scrollbar">
        {/* Main Navigation Sections */}
        {navigationSections.map((section) => (
          <div key={section.id}>
            {!isCollapsed ? (
              <button
                onClick={() => toggleSection(section.id)}
                className="w-full flex items-center justify-between px-3 py-1.5 text-[10px] font-bold text-text-muted hover:text-accent uppercase tracking-widest transition-colors rounded-lg hover:bg-white/5 group"
              >
                <span>{section.label}</span>
                <div className="text-text-muted group-hover:text-accent transition-colors">
                  {expandedSections[section.id] ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                </div>
              </button>
            ) : (
              section.id !== 'dashboard' && <div className="h-px bg-white/10 my-2 mx-2" />
            )}

            <div className={cn("space-y-1 mt-1 transition-all duration-300 ease-in-out",
              !isCollapsed && !expandedSections[section.id] ? "max-h-0 opacity-0 overflow-hidden" : "max-h-[500px] opacity-100"
            )}>
              {section.items.map((item) => {
                const isActive = pathname === item.href
                return (
                  <Link
                    key={item.name}
                    href={item.href}
                    className={cn(
                      'flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all group relative overflow-hidden',
                      isActive
                        ? 'bg-accent/10 text-accent shadow-sm border border-accent/20 backdrop-blur-md'
                        : 'text-text-secondary dark:text-text-dark-secondary hover:text-text-primary dark:hover:text-text-dark-primary hover:bg-white/5 dark:hover:bg-white/5 hover:pl-4'
                    )}
                  >
                    {isActive && <div className="absolute left-0 top-0 bottom-0 w-1 bg-accent rounded-r-full" />}
                    <item.icon className={cn("w-5 h-5 flex-shrink-0 transition-transform duration-200", !isActive && "group-hover:scale-110", isActive && "text-accent")} />
                    {!isCollapsed && <span>{item.name}</span>}
                    {isCollapsed && (
                      <div className="absolute left-full ml-4 px-3 py-1.5 bg-slate-900/90 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity whitespace-nowrap z-50 backdrop-blur-md border border-white/10 shadow-xl">
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
        {!isCollapsed && <div className="mx-3 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent my-4" />}

        {/* Management Section */}
        <div>
          {!isCollapsed ? (
            <button
              onClick={() => setManagementOpen(!managementOpen)}
              className="w-full flex items-center justify-between px-3 py-1.5 text-[10px] font-bold text-text-muted hover:text-accent uppercase tracking-widest transition-colors rounded-lg hover:bg-white/5 group"
            >
              <span>Data & Settings</span>
              <div className="text-text-muted group-hover:text-accent transition-colors">
                {managementOpen ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
              </div>
            </button>
          ) : (
            <div className="h-px bg-white/10 my-2 mx-2" />
          )}

          <div className={cn("space-y-1 mt-1 transition-all duration-300 ease-in-out",
            !isCollapsed && !managementOpen ? "max-h-0 opacity-0 overflow-hidden" : "max-h-[500px] opacity-100"
          )}>
            {managementNavigation.map((item) => {
              const isActive = pathname === item.href
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={cn(
                    'flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all group relative overflow-hidden',
                    isActive
                      ? 'bg-accent/10 text-accent shadow-sm border border-accent/20 backdrop-blur-md'
                      : 'text-text-secondary dark:text-text-dark-secondary hover:text-text-primary dark:hover:text-text-dark-primary hover:bg-white/5 dark:hover:bg-white/5 hover:pl-4'
                  )}
                >
                  {isActive && <div className="absolute left-0 top-0 bottom-0 w-1 bg-accent rounded-r-full" />}
                  <item.icon className={cn("w-5 h-5 flex-shrink-0 transition-transform duration-200", !isActive && "group-hover:scale-110", isActive && "text-accent")} />
                  {!isCollapsed && <span>{item.name}</span>}
                  {isCollapsed && (
                    <div className="absolute left-full ml-4 px-3 py-1.5 bg-slate-900/90 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity whitespace-nowrap z-50 backdrop-blur-md border border-white/10 shadow-xl">
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
      <div className="px-4 py-4 border-t border-white/10 space-y-4 bg-white/5 backdrop-blur-sm">
        <button
          onClick={toggleTheme}
          className="flex items-center gap-3 w-full px-3 py-2.5 rounded-xl text-sm font-medium text-text-secondary dark:text-text-dark-secondary hover:bg-white/10 transition-colors overflow-hidden whitespace-nowrap border border-transparent hover:border-white/10 group"
        >
          <div className="dark:hidden flex items-center gap-3 flex-shrink-0">
            <span className="w-5 h-5 group-hover:rotate-12 transition-transform">🌙</span>
            {!isCollapsed && <span>Dark Mode</span>}
          </div>
          <div className="hidden dark:flex items-center gap-3 flex-shrink-0">
            <span className="w-5 h-5 group-hover:rotate-90 transition-transform">☀️</span>
            {!isCollapsed && <span>Light Mode</span>}
          </div>
        </button>
        {!isCollapsed && (
          <div className="flex items-center gap-2 text-xs text-text-muted dark:text-text-dark-muted px-3">
            <div className="relative">
              <div className="w-2 h-2 rounded-full bg-success shadow-[0_0_8px_rgba(34,197,94,0.6)] flex-shrink-0" />
              <div className="absolute inset-0 rounded-full bg-success animate-ping opacity-75" />
            </div>
            <span className="truncate font-medium">Local-first analytics</span>
          </div>
        )}
      </div>
    </div>
  )
}
