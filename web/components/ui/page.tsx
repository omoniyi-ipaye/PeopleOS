import type { ReactNode } from 'react'
import { cn } from '@/lib/utils'

export function Page({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cn('mx-auto w-full max-w-[1440px] space-y-6 pb-12', className)}>{children}</div>
}

export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow?: string
  title: ReactNode
  description?: ReactNode
  actions?: ReactNode
}) {
  return (
    <header className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
      <div className="min-w-0">
        {eyebrow && <div className="mb-2 text-xs font-bold uppercase tracking-[0.16em] text-violet-600 dark:text-violet-300">{eyebrow}</div>}
        <h1 className="text-3xl font-semibold tracking-tight text-slate-950 dark:text-white md:text-4xl">{title}</h1>
        {description && <div className="mt-2 max-w-3xl text-sm leading-6 text-slate-600 dark:text-slate-400">{description}</div>}
      </div>
      {actions && <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div>}
    </header>
  )
}

export function SectionHeader({ title, description, action }: { title: ReactNode; description?: ReactNode; action?: ReactNode }) {
  return (
    <div className="mb-4 flex items-start justify-between gap-4">
      <div>
        <h2 className="text-base font-semibold text-slate-950 dark:text-white">{title}</h2>
        {description && <div className="mt-1 text-xs leading-5 text-slate-500 dark:text-slate-400">{description}</div>}
      </div>
      {action}
    </div>
  )
}
