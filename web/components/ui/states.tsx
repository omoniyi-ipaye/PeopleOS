'use client'

import Link from 'next/link'
import { AlertTriangle, Database, RefreshCw, Upload, type LucideIcon } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { EmptyState as GovernedEmptyState } from '@/components/ui/data-display'
import { Surface } from '@/components/ui/surface'
import { cn } from '@/lib/utils'

interface ErrorStateProps {
  title?: string
  message?: string
  error?: Error | unknown
  onRetry?: () => void
  className?: string
}

export function ErrorState({ title = 'Something went wrong', message, error, onRetry, className }: ErrorStateProps) {
  const errorMessage = message || (error instanceof Error ? error.message : 'An unexpected error occurred. Please try again.')
  return (
    <GovernedEmptyState
      tone="danger"
      title={title}
      description={errorMessage}
      icon={<AlertTriangle className="h-5 w-5" />}
      action={onRetry ? <Button variant="secondary" onClick={onRetry}><RefreshCw className="h-4 w-4" />Retry</Button> : undefined}
    />
  )
}

interface EmptyStateProps {
  icon?: LucideIcon
  title: string
  description?: string
  action?: { label: string; href?: string; onClick?: () => void }
  className?: string
}

export function EmptyState({ icon: Icon = Database, title, description, action }: EmptyStateProps) {
  const actionNode = !action ? undefined : action.href ? (
    <Link href={action.href}><Button><Upload className="h-4 w-4" />{action.label}</Button></Link>
  ) : (
    <Button onClick={action.onClick}>{action.label}</Button>
  )
  return <GovernedEmptyState title={title} description={description ?? ''} icon={<Icon className="h-5 w-5" />} action={actionNode} />
}

interface LoadingStateProps { message?: string; className?: string }
export function LoadingState({ message = 'Loading…', className }: LoadingStateProps) {
  return <div role="status" aria-live="polite" className={cn('grid min-h-40 place-items-center text-sm text-slate-500 dark:text-slate-400', className)}><span className="animate-pulse-subtle">{message}</span></div>
}

interface SkeletonProps { className?: string; style?: React.CSSProperties }
export function Skeleton({ className, style }: SkeletonProps) {
  return <div aria-hidden="true" className={cn('animate-pulse rounded-lg bg-slate-100 dark:bg-white/5', className)} style={style} />
}

export function CardSkeleton({ className }: SkeletonProps) {
  return <Surface aria-busy="true" className={className}><Skeleton className="mb-4 h-4 w-1/3" /><Skeleton className="mb-2 h-8 w-1/2" /><Skeleton className="h-3 w-2/3" /></Surface>
}

export function TableSkeleton({ rows = 5 }: { rows?: number }) {
  return <div aria-busy="true" className="space-y-3"><Skeleton className="h-10 w-full" />{Array.from({ length: rows }).map((_, i) => <Skeleton key={i} className="h-12 w-full" />)}</div>
}

const chartHeights = [38, 64, 46, 78, 55, 70, 42, 60]
export function ChartSkeleton({ className }: SkeletonProps) {
  return <div aria-busy="true" className={cn('flex h-64 items-end justify-around gap-2 p-4', className)}>{chartHeights.map((height, i) => <Skeleton key={i} className="w-8" style={{ height: `${height}%` }} />)}</div>
}
