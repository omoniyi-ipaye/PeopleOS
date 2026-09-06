export const designTokens = {
  radius: {
    control: 'rounded-xl',
    surface: 'rounded-2xl',
    panel: 'rounded-3xl',
    pill: 'rounded-full',
  },
  spacing: {
    page: 'p-4 md:p-6 lg:p-8',
    section: 'space-y-6',
    panel: 'p-5 md:p-6',
    compact: 'p-4',
  },
  elevation: {
    base: 'shadow-sm',
    raised: 'shadow-lg shadow-slate-900/5 dark:shadow-black/20',
    overlay: 'shadow-2xl shadow-slate-900/15 dark:shadow-black/40',
  },
  motion: {
    interactive: 'transition-colors duration-150 ease-out',
    surface: 'transition-all duration-200 ease-out',
  },
  focus: 'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500 focus-visible:ring-offset-2 focus-visible:ring-offset-white dark:focus-visible:ring-offset-slate-950',
} as const

export const semanticTone = {
  neutral: {
    surface: 'border-slate-200 bg-white dark:border-white/10 dark:bg-slate-900',
    subtle: 'bg-slate-50 dark:bg-white/[0.04]',
    text: 'text-slate-700 dark:text-slate-300',
    icon: 'text-slate-500 dark:text-slate-400',
  },
  info: {
    surface: 'border-sky-200 bg-sky-50/70 dark:border-sky-500/20 dark:bg-sky-500/[0.06]',
    text: 'text-sky-800 dark:text-sky-300',
    icon: 'text-sky-600 dark:text-sky-300',
  },
  success: {
    surface: 'border-emerald-200 bg-emerald-50/70 dark:border-emerald-500/20 dark:bg-emerald-500/[0.06]',
    text: 'text-emerald-800 dark:text-emerald-300',
    icon: 'text-emerald-600 dark:text-emerald-300',
  },
  warning: {
    surface: 'border-amber-200 bg-amber-50/70 dark:border-amber-500/20 dark:bg-amber-500/[0.06]',
    text: 'text-amber-800 dark:text-amber-300',
    icon: 'text-amber-600 dark:text-amber-300',
  },
  danger: {
    surface: 'border-red-200 bg-red-50/70 dark:border-red-500/20 dark:bg-red-500/[0.06]',
    text: 'text-red-800 dark:text-red-300',
    icon: 'text-red-600 dark:text-red-300',
  },
  accent: {
    surface: 'border-violet-200 bg-violet-50/70 dark:border-violet-500/20 dark:bg-violet-500/[0.06]',
    text: 'text-violet-800 dark:text-violet-300',
    icon: 'text-violet-600 dark:text-violet-300',
  },
} as const

export type SemanticTone = keyof typeof semanticTone
