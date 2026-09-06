import type { HTMLAttributes, ReactNode } from 'react'
import { Surface } from '@/components/ui/surface'
import { cn } from '@/lib/utils'

/**
 * @deprecated Prefer Surface or Card for new product work.
 * Kept as a transition wrapper for legacy analytics screens.
 */
export interface GlassCardProps extends HTMLAttributes<HTMLElement> {
  children: ReactNode
  variant?: 'default' | 'hover' | 'active'
}

export function GlassCard({ children, variant = 'default', className, ...props }: GlassCardProps) {
  return (
    <Surface
      as="section"
      interactive={variant === 'hover'}
      tone={variant === 'active' ? 'accent' : 'neutral'}
      className={cn('relative overflow-hidden', className)}
      {...props}
    >
      {children}
    </Surface>
  )
}
