import React from 'react'
import { cn } from '../../lib/utils'
import { LucideIcon, Info } from 'lucide-react'

interface KPICardProps {
  title: React.ReactNode
  value: string | number
  icon: LucideIcon
  trend?: string
  subtitle?: string
  insight: string
  variant?: 'default' | 'success' | 'warning' | 'danger'
  className?: string
}

export function KPICard({
  title,
  value,
  icon: Icon,
  trend,
  subtitle,
  insight,
  variant = 'default',
  className,
}: KPICardProps) {
  const iconColors = {
    default: 'bg-accent/10 text-accent',
    success: 'bg-success/10 text-success',
    warning: 'bg-warning/10 text-warning',
    danger: 'bg-danger/10 text-danger',
  }

  return (
    <div
      className={cn(
        'bg-surface dark:bg-surface-dark rounded-xl border border-border dark:border-border-dark p-4 shadow-sm hover:shadow-md transition-shadow',
        className
      )}
    >
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-center gap-1.5 mb-1">
            <div className="text-sm text-text-secondary dark:text-text-dark-secondary font-medium">{title}</div>
            {insight && (
              <div className="group relative">
                <Info className="w-3.5 h-3.5 text-text-muted dark:text-text-dark-muted cursor-help" />
                <div className="absolute bottom-full left-0 mb-2 w-64 p-3 bg-slate-900 dark:bg-slate-800 text-white text-[11px] rounded-lg shadow-xl opacity-0 group-hover:opacity-100 transition-opacity z-50 pointer-events-none border border-slate-700 leading-relaxed">
                  <div className="font-bold text-success mb-1 uppercase text-[9px]">Plain Language</div>
                  {insight}
                </div>
              </div>
            )}
          </div>
          <p className="text-2xl font-bold text-text-primary dark:text-text-dark-primary tracking-tight">{value}</p>
          {trend && (
            <p className="text-xs text-text-muted dark:text-text-dark-muted mt-1 font-medium">{trend}</p>
          )}
          {subtitle && (
            <p className="text-xs text-text-muted dark:text-text-dark-muted mt-1">{subtitle}</p>
          )}
        </div>
        <div
          className={cn(
            'w-10 h-10 rounded-lg flex items-center justify-center shadow-inner',
            iconColors[variant]
          )}
        >
          <Icon className="w-5 h-5" />
        </div>
      </div>
    </div>
  )
}
