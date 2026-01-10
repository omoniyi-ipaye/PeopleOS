'use client'

import { cn } from '@/lib/utils'
import { AlertTriangle, Database, RefreshCw, Upload, type LucideIcon } from 'lucide-react'

// ============================================
// Error State Component
// ============================================

interface ErrorStateProps {
  title?: string
  message?: string
  error?: Error | unknown
  onRetry?: () => void
  className?: string
}

export function ErrorState({
  title = 'Something went wrong',
  message,
  error,
  onRetry,
  className
}: ErrorStateProps) {
  const errorMessage = message || (error instanceof Error ? error.message : 'An unexpected error occurred. Please try again.')

  return (
    <div className={cn("flex flex-col items-center justify-center h-full gap-4 text-center py-12", className)}>
      <AlertTriangle className="w-12 h-12 text-danger" />
      <h2 className="text-xl font-semibold text-text-primary dark:text-text-dark-primary">
        {title}
      </h2>
      <p className="text-text-secondary dark:text-text-dark-secondary max-w-md">
        {errorMessage}
      </p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="flex items-center gap-2 px-4 py-2 bg-accent text-white rounded-lg hover:bg-accent/90 transition-colors"
        >
          <RefreshCw className="w-4 h-4" />
          Retry
        </button>
      )}
    </div>
  )
}

// ============================================
// Empty State Component
// ============================================

interface EmptyStateProps {
  icon?: LucideIcon
  title: string
  description?: string
  action?: {
    label: string
    href?: string
    onClick?: () => void
  }
  className?: string
}

export function EmptyState({
  icon: Icon = Database,
  title,
  description,
  action,
  className
}: EmptyStateProps) {
  return (
    <div className={cn("flex flex-col items-center justify-center h-full gap-4 text-center py-12", className)}>
      <Icon className="w-12 h-12 text-text-muted dark:text-text-dark-muted" />
      <h2 className="text-xl font-semibold text-text-primary dark:text-text-dark-primary">
        {title}
      </h2>
      {description && (
        <p className="text-text-secondary dark:text-text-dark-secondary max-w-md">
          {description}
        </p>
      )}
      {action && (
        action.href ? (
          <a
            href={action.href}
            className="flex items-center gap-2 px-4 py-2 bg-accent text-white rounded-lg hover:bg-accent/90 transition-colors"
          >
            <Upload className="w-4 h-4" />
            {action.label}
          </a>
        ) : (
          <button
            onClick={action.onClick}
            className="flex items-center gap-2 px-4 py-2 bg-accent text-white rounded-lg hover:bg-accent/90 transition-colors"
          >
            {action.label}
          </button>
        )
      )}
    </div>
  )
}

// ============================================
// Loading State Component
// ============================================

interface LoadingStateProps {
  message?: string
  className?: string
}

export function LoadingState({
  message = 'Loading...',
  className
}: LoadingStateProps) {
  return (
    <div className={cn("flex items-center justify-center h-full py-12", className)}>
      <div className="animate-pulse-subtle text-text-secondary dark:text-text-dark-secondary">
        {message}
      </div>
    </div>
  )
}

// ============================================
// Skeleton Components
// ============================================

interface SkeletonProps {
  className?: string
  style?: React.CSSProperties
}

export function Skeleton({ className, style }: SkeletonProps) {
  return (
    <div
      className={cn("animate-pulse bg-surface-secondary dark:bg-surface-dark-secondary rounded", className)}
      style={style}
    />
  )
}

export function CardSkeleton({ className }: SkeletonProps) {
  return (
    <div className={cn("p-6 border border-border dark:border-border-dark rounded-xl", className)}>
      <Skeleton className="h-4 w-1/3 mb-4" />
      <Skeleton className="h-8 w-1/2 mb-2" />
      <Skeleton className="h-3 w-2/3" />
    </div>
  )
}

export function TableSkeleton({ rows = 5 }: { rows?: number }) {
  return (
    <div className="space-y-3">
      <Skeleton className="h-10 w-full" />
      {Array.from({ length: rows }).map((_, i) => (
        <Skeleton key={i} className="h-12 w-full" />
      ))}
    </div>
  )
}

export function ChartSkeleton({ className }: SkeletonProps) {
  return (
    <div className={cn("flex items-end justify-around h-64 p-4 gap-2", className)}>
      {Array.from({ length: 8 }).map((_, i) => (
        <Skeleton
          key={i}
          className="w-8"
          style={{ height: `${Math.random() * 60 + 20}%` }}
        />
      ))}
    </div>
  )
}
