import type { HTMLAttributes, ReactNode } from 'react'
import { cn } from '@/lib/utils'
import { designTokens, semanticTone, type SemanticTone } from '@/design-system/tokens'

export interface SurfaceProps extends HTMLAttributes<HTMLElement> {
  as?: 'div' | 'section' | 'article' | 'aside'
  tone?: SemanticTone
  /** compact/default are canonical. md/lg remain supported aliases for product composition readability. */
  padding?: 'none' | 'compact' | 'default' | 'md' | 'lg'
  elevation?: 'none' | 'base' | 'raised'
  interactive?: boolean
  children: ReactNode
}

export function Surface({
  as: Component = 'section',
  tone = 'neutral',
  padding = 'default',
  elevation = 'base',
  interactive = false,
  className,
  children,
  ...props
}: SurfaceProps) {
  const normalizedPadding = padding === 'md' ? 'compact' : padding === 'lg' ? 'default' : padding
  const paddingClass = normalizedPadding === 'none' ? '' : normalizedPadding === 'compact' ? designTokens.spacing.compact : designTokens.spacing.panel
  const elevationClass = elevation === 'none' ? '' : elevation === 'raised' ? designTokens.elevation.raised : designTokens.elevation.base

  return (
    <Component
      className={cn(
        'border',
        designTokens.radius.surface,
        semanticTone[tone].surface,
        paddingClass,
        elevationClass,
        interactive && `${designTokens.motion.surface} hover:-translate-y-0.5 hover:shadow-lg`,
        className
      )}
      {...props}
    >
      {children}
    </Component>
  )
}
