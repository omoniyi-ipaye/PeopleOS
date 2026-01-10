import React from 'react'
import { cn } from '../../lib/utils'

interface CardProps {
  title?: React.ReactNode
  subtitle?: React.ReactNode
  children: React.ReactNode
  className?: string
  padding?: 'none' | 'sm' | 'md' | 'lg'
}

export function Card({
  title,
  subtitle,
  children,
  className,
  padding = 'md',
  ...props
}: CardProps & Omit<React.HTMLAttributes<HTMLDivElement>, 'title'>) {
  const paddingClasses = {
    none: '',
    sm: 'p-3',
    md: 'p-4',
    lg: 'p-6',
  }

  return (
    <div
      className={cn(
        'bg-surface dark:bg-surface-dark rounded-xl border border-border dark:border-border-dark shadow-sm',
        paddingClasses[padding],
        className
      )}
      {...props}
    >
      {(title || subtitle) && (
        <div className="mb-4">
          {title && (
            <div className="text-lg font-semibold text-text-primary dark:text-text-dark-primary">{title}</div>
          )}
          {subtitle && (
            <div className="text-sm text-text-secondary dark:text-text-dark-secondary mt-0.5">{subtitle}</div>
          )}
        </div>
      )}
      {children}
    </div>
  )
}

export function CardSkeleton({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        'bg-surface dark:bg-surface-dark rounded-xl border border-border dark:border-border-dark p-4 animate-pulse',
        className
      )}
    >
      <div className="h-4 bg-surface-hover dark:bg-surface-dark-hover rounded w-1/3 mb-4" />
      <div className="h-32 bg-surface-hover dark:bg-surface-dark-hover rounded" />
    </div>
  )
}
