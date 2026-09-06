import { forwardRef } from 'react'
import { Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'
import { designTokens } from '@/design-system/tokens'

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger' | 'outline'
  size?: 'sm' | 'md' | 'lg' | 'icon'
  isLoading?: boolean
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'primary', size = 'md', isLoading = false, disabled, children, type = 'button', ...props }, ref) => {
    const variantClasses = {
      primary: 'border-violet-600 bg-violet-600 text-white shadow-sm hover:border-violet-500 hover:bg-violet-500 active:bg-violet-700',
      secondary: 'border-slate-200 bg-white text-slate-800 shadow-sm hover:bg-slate-50 dark:border-white/10 dark:bg-slate-900 dark:text-slate-100 dark:hover:bg-white/5',
      outline: 'border-slate-300 bg-transparent text-slate-700 hover:bg-slate-50 dark:border-white/15 dark:text-slate-200 dark:hover:bg-white/5',
      ghost: 'border-transparent bg-transparent text-slate-600 hover:bg-slate-100 hover:text-slate-950 dark:text-slate-300 dark:hover:bg-white/5 dark:hover:text-white',
      danger: 'border-red-600 bg-red-600 text-white shadow-sm hover:border-red-500 hover:bg-red-500 active:bg-red-700',
    }

    const sizeClasses = {
      sm: 'h-8 px-3 text-xs',
      md: 'h-10 px-4 text-sm',
      lg: 'h-12 px-5 text-sm',
      icon: 'h-10 w-10 p-0',
    }

    return (
      <button
        ref={ref}
        type={type}
        disabled={disabled || isLoading}
        aria-busy={isLoading || undefined}
        className={cn(
          'inline-flex shrink-0 items-center justify-center gap-2 border font-semibold',
          designTokens.radius.control,
          designTokens.motion.interactive,
          designTokens.focus,
          'disabled:pointer-events-none disabled:opacity-50',
          variantClasses[variant],
          sizeClasses[size],
          className
        )}
        {...props}
      >
        {isLoading && <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />}
        {children}
      </button>
    )
  }
)

Button.displayName = 'Button'
