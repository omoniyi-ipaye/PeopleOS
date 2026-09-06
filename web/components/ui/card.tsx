import type { HTMLAttributes, ReactNode } from 'react'
import { cn } from '@/lib/utils'
import { Surface } from '@/components/ui/surface'
import { SectionHeader } from '@/components/ui/page'

export interface CardProps extends Omit<HTMLAttributes<HTMLElement>, 'title'> {
  title?: ReactNode
  subtitle?: ReactNode
  action?: ReactNode
  children: ReactNode
  padding?: 'none' | 'sm' | 'md' | 'lg'
  interactive?: boolean
}

export function Card({ title, subtitle, action, children, className, padding = 'md', interactive = false, ...props }: CardProps) {
  const surfacePadding = padding === 'none' ? 'none' : padding === 'sm' ? 'compact' : 'default'
  return (
    <Surface
      as="section"
      padding={surfacePadding}
      interactive={interactive}
      className={cn(padding === 'lg' && 'md:p-8', className)}
      {...props}
    >
      {(title || subtitle || action) && <SectionHeader title={title ?? ''} description={subtitle} action={action} />}
      {children}
    </Surface>
  )
}

export function CardSkeleton({ className }: { className?: string }) {
  return (
    <Surface aria-busy="true" className={cn('animate-pulse', className)}>
      <div className="mb-4 h-4 w-1/3 rounded bg-slate-100 dark:bg-white/5" />
      <div className="h-32 rounded-xl bg-slate-100 dark:bg-white/5" />
    </Surface>
  )
}
