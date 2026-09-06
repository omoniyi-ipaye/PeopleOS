'use client'

import * as React from 'react'
import { cn } from '@/lib/utils'

interface TooltipContextValue {
  open: boolean
  setOpen: (open: boolean) => void
  contentId: string
}

const TooltipContext = React.createContext<TooltipContextValue | null>(null)

function useTooltip() {
  const context = React.useContext(TooltipContext)
  if (!context) throw new Error('Tooltip components must be used inside <Tooltip>.')
  return context
}

export function TooltipProvider({ children }: { children: React.ReactNode }) {
  return <>{children}</>
}

export function Tooltip({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = React.useState(false)
  const contentId = React.useId()
  return (
    <TooltipContext.Provider value={{ open, setOpen, contentId }}>
      <span className="relative inline-flex">{children}</span>
    </TooltipContext.Provider>
  )
}

export function TooltipTrigger({ children, asChild = false }: { children: React.ReactNode; asChild?: boolean }) {
  const { setOpen, contentId } = useTooltip()
  const handlers = {
    'aria-describedby': contentId,
    onMouseEnter: () => setOpen(true),
    onMouseLeave: () => setOpen(false),
    onFocus: () => setOpen(true),
    onBlur: () => setOpen(false),
  }

  if (asChild && React.isValidElement(children)) {
    return React.cloneElement(children as React.ReactElement<Record<string, unknown>>, handlers)
  }

  return <button type="button" className="inline-flex" {...handlers}>{children}</button>
}

export function TooltipContent({ children, className }: { children: React.ReactNode; className?: string }) {
  const { open, contentId } = useTooltip()
  if (!open) return null
  return (
    <span
      id={contentId}
      role="tooltip"
      className={cn(
        'pointer-events-none absolute bottom-full left-1/2 z-[100] mb-2 w-max max-w-xs -translate-x-1/2 rounded-xl border border-slate-200 bg-white px-2.5 py-2 text-xs leading-5 text-slate-700 shadow-xl dark:border-white/10 dark:bg-slate-900 dark:text-slate-200',
        className
      )}
    >
      {children}
    </span>
  )
}
