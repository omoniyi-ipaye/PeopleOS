import type { ReactNode } from 'react'
import { AlertTriangle, CheckCircle2, Circle, Info, XCircle } from 'lucide-react'
import { cn } from '@/lib/utils'
import { semanticTone, type SemanticTone } from '@/design-system/tokens'

const toneIcon = {
  neutral: Circle,
  info: Info,
  success: CheckCircle2,
  warning: AlertTriangle,
  danger: XCircle,
  accent: Circle,
} as const

export function StatusBadge({ tone = 'neutral', children, className }: { tone?: SemanticTone; children: ReactNode; className?: string }) {
  const Icon = toneIcon[tone]
  return (
    <span className={cn('inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-semibold', semanticTone[tone].surface, semanticTone[tone].text, className)}>
      <Icon className="h-3.5 w-3.5" aria-hidden="true" />
      {children}
    </span>
  )
}

export function StatusDot({ tone = 'neutral', label }: { tone?: SemanticTone; label: string }) {
  const dot = tone === 'success' ? 'bg-emerald-500' : tone === 'warning' ? 'bg-amber-500' : tone === 'danger' ? 'bg-red-500' : tone === 'info' ? 'bg-sky-500' : tone === 'accent' ? 'bg-violet-500' : 'bg-slate-400'
  return <span className="inline-flex items-center gap-2 text-xs font-medium text-slate-600 dark:text-slate-300"><span className={cn('h-2 w-2 rounded-full', dot)} aria-hidden="true" />{label}</span>
}
