'use client'

import { cn } from '@/lib/utils'

interface Tab {
  id: string
  label: string
  disabled?: boolean
}

interface TabGroupProps<T extends string> {
  tabs: readonly Tab[] | Tab[]
  activeTab: T
  onTabChange: (tab: T) => void
  className?: string
  size?: 'sm' | 'md'
}

export function TabGroup<T extends string>({
  tabs,
  activeTab,
  onTabChange,
  className,
  size = 'md'
}: TabGroupProps<T>) {
  return (
    <div
      role="tablist"
      className={cn(
        "flex bg-surface dark:bg-surface-dark border border-border dark:border-border-dark p-1 rounded-lg shadow-sm overflow-x-auto no-scrollbar",
        className
      )}
    >
      {tabs.map((tab) => (
        <button
          key={tab.id}
          role="tab"
          aria-selected={activeTab === tab.id}
          aria-controls={`${tab.id}-panel`}
          disabled={tab.disabled}
          onClick={() => onTabChange(tab.id as T)}
          className={cn(
            'font-medium rounded-md transition-all whitespace-nowrap',
            size === 'sm' ? 'px-3 py-1 text-xs' : 'px-4 py-1.5 text-sm',
            activeTab === tab.id
              ? 'bg-accent text-white shadow-sm'
              : 'text-text-secondary dark:text-text-dark-secondary hover:text-text-primary dark:hover:text-text-dark-primary',
            tab.disabled && 'opacity-50 cursor-not-allowed'
          )}
        >
          {tab.label}
        </button>
      ))}
    </div>
  )
}

// Panel container that shows content for the active tab
interface TabPanelProps {
  id: string
  activeTab: string
  children: React.ReactNode
  className?: string
}

export function TabPanel({ id, activeTab, children, className }: TabPanelProps) {
  if (activeTab !== id) return null

  return (
    <div
      role="tabpanel"
      id={`${id}-panel`}
      aria-labelledby={id}
      className={className}
    >
      {children}
    </div>
  )
}
