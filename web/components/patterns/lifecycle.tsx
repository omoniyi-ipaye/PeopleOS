import type { ReactNode } from 'react'
import { CheckCircle2, Circle, Clock3, XCircle } from 'lucide-react'
import { cn } from '@/lib/utils'
import { StatusBadge } from '@/components/ui/status'

export type LifecycleState = 'complete' | 'active' | 'pending' | 'blocked'

export interface LifecycleStepProps {
  title: string
  description?: ReactNode
  state: LifecycleState
  meta?: ReactNode
}

const stateConfig = {
  complete: { icon: CheckCircle2, tone: 'success' as const, label: 'Complete' },
  active: { icon: Clock3, tone: 'accent' as const, label: 'Active' },
  pending: { icon: Circle, tone: 'neutral' as const, label: 'Pending' },
  blocked: { icon: XCircle, tone: 'danger' as const, label: 'Blocked' },
}

export function Lifecycle({ steps, label = 'Lifecycle' }: { steps: LifecycleStepProps[]; label?: string }) {
  return (
    <ol aria-label={label} className="space-y-2">
      {steps.map((step, index) => {
        const config = stateConfig[step.state]
        const Icon = config.icon
        return (
          <li key={`${step.title}-${index}`} className="flex gap-3 rounded-2xl border border-slate-200 bg-white p-4 dark:border-white/10 dark:bg-slate-900">
            <div className={cn('mt-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-xl', step.state === 'complete' ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-300' : step.state === 'active' ? 'bg-violet-100 text-violet-700 dark:bg-violet-500/10 dark:text-violet-300' : step.state === 'blocked' ? 'bg-red-100 text-red-700 dark:bg-red-500/10 dark:text-red-300' : 'bg-slate-100 text-slate-500 dark:bg-white/5 dark:text-slate-400')}>
              <Icon className="h-4 w-4" aria-hidden="true" />
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="font-semibold text-slate-900 dark:text-white">{step.title}</div>
                <StatusBadge tone={config.tone}>{config.label}</StatusBadge>
              </div>
              {step.description && <div className="mt-1 text-sm leading-5 text-slate-500 dark:text-slate-400">{step.description}</div>}
              {step.meta && <div className="mt-2 text-xs text-slate-400 dark:text-slate-500">{step.meta}</div>}
            </div>
          </li>
        )
      })}
    </ol>
  )
}
