'use client'

import * as React from 'react'
import { cn } from '../../lib/utils'

export function TooltipProvider({ children }: { children: React.ReactNode }) {
    return <>{children}</>
}

export function Tooltip({ children }: { children: React.ReactNode }) {
    const [open, setOpen] = React.useState(false)
    return (
        <div
            className="relative inline-block"
            onMouseEnter={() => setOpen(true)}
            onMouseLeave={() => setOpen(false)}
        >
            {children}
        </div>
    )
}

export function TooltipTrigger({ children, asChild }: { children: React.ReactNode, asChild?: boolean }) {
    return <>{children}</>
}

export function TooltipContent({ children, className }: { children: React.ReactNode, className?: string }) {
    return (
        <div className={cn(
            "absolute z-[100] bottom-full left-1/2 -translate-x-1/2 mb-2 w-max max-w-xs p-2 bg-surface dark:bg-surface-dark border border-border dark:border-border-dark rounded-lg shadow-xl text-xs text-text-primary dark:text-text-dark-primary animate-in fade-in zoom-in duration-200 pointer-events-none",
            className
        )}>
            {children}
        </div>
    )
}
