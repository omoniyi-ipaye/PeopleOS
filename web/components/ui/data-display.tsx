import { isValidElement, type ElementType, type ReactElement, type ReactNode } from 'react'
import { ArrowRight, CircleHelp } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Surface } from '@/components/ui/surface'
import { StatusBadge } from '@/components/ui/status'
import type { SemanticTone } from '@/design-system/tokens'

type IconSlot = ElementType | ReactElement

function renderIcon(icon: IconSlot | undefined, fallback = true) {
  if (isValidElement(icon)) return icon
  if (icon) {
    const Icon = icon as ElementType
    return <Icon className="h-4 w-4" />
  }
  return fallback ? <CircleHelp className="h-4 w-4" /> : null
}

export function MetricCard({
  label,
  value,
  detail,
  icon,
  tone = 'neutral',
  status,
}: {
  label: string
  value: ReactNode
  detail?: ReactNode
  icon?: IconSlot
  tone?: SemanticTone
  status?: ReactNode
}) {
  return (
    <Surface padding="compact" className="min-w-0">
      <div className="flex items-start justify-between gap-3">
        <div className={cn('grid h-9 w-9 place-items-center rounded-xl', tone === 'accent' ? 'bg-violet-100 text-violet-700 dark:bg-violet-500/10 dark:text-violet-300' : tone === 'success' ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-300' : tone === 'warning' ? 'bg-amber-100 text-amber-700 dark:bg-amber-500/10 dark:text-amber-300' : tone === 'danger' ? 'bg-red-100 text-red-700 dark:bg-red-500/10 dark:text-red-300' : 'bg-slate-100 text-slate-600 dark:bg-white/5 dark:text-slate-300')}>{renderIcon(icon)}</div>
        {status}
      </div>
      <div className="mt-4 text-xs font-semibold text-slate-500 dark:text-slate-400">{label}</div>
      <div className="mt-1 truncate text-2xl font-semibold tracking-tight text-slate-950 dark:text-white">{value}</div>
      {detail && <div className="mt-1 text-xs leading-5 text-slate-500 dark:text-slate-400">{detail}</div>}
    </Surface>
  )
}

export function EmptyState({
  title,
  description = '',
  icon,
  action,
  tone = 'neutral',
}: {
  title: string
  description?: ReactNode
  icon?: IconSlot
  action?: ReactNode
  tone?: SemanticTone
}) {
  return (
    <Surface tone={tone} className="flex min-h-64 flex-col items-center justify-center text-center">
      {icon && <div className="mb-4 grid h-12 w-12 place-items-center rounded-2xl bg-white/70 text-slate-600 shadow-sm dark:bg-white/5 dark:text-slate-300">{renderIcon(icon, false)}</div>}
      <h2 className="text-lg font-semibold text-slate-950 dark:text-white">{title}</h2>
      {description && <div className="mt-2 max-w-xl text-sm leading-6 text-slate-600 dark:text-slate-400">{description}</div>}
      {action && <div className="mt-5">{action}</div>}
    </Surface>
  )
}

export function ActionLink({ children }: { children: ReactNode }) {
  return <span className="inline-flex items-center gap-1 text-sm font-semibold text-violet-600 dark:text-violet-300">{children}<ArrowRight className="h-4 w-4" /></span>
}

export function StateSummary(props: ({ label: string; value: string; tone?: SemanticTone } | { title: string; description: ReactNode; tone?: SemanticTone })) {
  if ('title' in props) {
    return <Surface tone={props.tone ?? 'neutral'} padding="compact"><div className="font-semibold text-slate-950 dark:text-white">{props.title}</div><div className="mt-1 text-sm leading-6 text-slate-600 dark:text-slate-400">{props.description}</div></Surface>
  }
  return <div className="flex items-center justify-between gap-4 border-b border-slate-100 py-3 last:border-0 dark:border-white/5"><span className="text-sm text-slate-600 dark:text-slate-300">{props.label}</span><StatusBadge tone={props.tone ?? 'neutral'}>{props.value}</StatusBadge></div>
}
