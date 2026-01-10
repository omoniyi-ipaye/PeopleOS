'use client'

import { cn } from '@/lib/utils'

interface NineBoxSummary {
  category: string
  count: number
  percentage: number
}

interface NineBoxGridProps {
  data: NineBoxSummary[]
}

const GRID_LAYOUT = [
  ['Potential Gems', 'High Potentials', 'Stars'],
  ['Inconsistent', 'Core Contributors', 'High Performers'],
  ['Underperformers', 'Effective', 'Solid Performers'],
]

const CELL_COLORS: Record<string, string> = {
  'Stars': 'bg-success/20 dark:bg-success/30 border-success/40 text-success',
  'High Performers': 'bg-success/15 dark:bg-success/20 border-success/30 text-success',
  'Solid Performers': 'bg-accent/15 dark:bg-accent/20 border-accent/30 text-accent',
  'High Potentials': 'bg-success/10 dark:bg-success/15 border-success/20 text-success',
  'Core Contributors': 'bg-accent/10 dark:bg-accent/15 border-accent/20 text-accent',
  'Effective': 'bg-surface dark:bg-surface-dark border-border dark:border-border-dark text-text-secondary dark:text-text-dark-secondary',
  'Potential Gems': 'bg-warning/15 dark:bg-warning/20 border-warning/30 text-warning',
  'Inconsistent': 'bg-warning/10 dark:bg-warning/15 border-warning/20 text-warning',
  'Underperformers': 'bg-danger/10 dark:bg-danger/20 border-danger/20 text-danger',
}

export function NineBoxGrid({ data }: NineBoxGridProps) {
  const dataMap = new Map(data.map((d) => [d.category, d]))

  return (
    <div className="space-y-4">
      {/* Y-axis label */}
      <div className="flex items-center">
        <div className="w-8 -rotate-90 text-[10px] uppercase font-bold text-text-muted dark:text-text-dark-muted text-center whitespace-nowrap tracking-wider">
          POTENTIAL →
        </div>
        <div className="flex-1">
          {/* Grid */}
          <div className="grid grid-cols-3 gap-2">
            {GRID_LAYOUT.flat().map((category) => {
              const cellData = dataMap.get(category)
              return (
                <div
                  key={category}
                  className={cn(
                    'p-3 rounded-xl border text-center transition-all hover:scale-[1.02] shadow-sm',
                    CELL_COLORS[category] || 'bg-surface dark:bg-surface-dark border-border dark:border-border-dark'
                  )}
                >
                  <div className="text-[10px] uppercase font-bold opacity-80 truncate tracking-tight">
                    {category}
                  </div>
                  <div className="text-xl font-bold mt-1 text-text-primary dark:text-text-dark-primary">
                    {cellData?.count || 0}
                  </div>
                  <div className="text-[10px] font-medium opacity-60">
                    {cellData?.percentage.toFixed(1) || 0}%
                  </div>
                </div>
              )
            })}
          </div>
          {/* X-axis label */}
          <div className="text-[10px] uppercase font-bold text-text-muted dark:text-text-dark-muted text-center mt-3 tracking-wider">
            PERFORMANCE →
          </div>
        </div>
      </div>
    </div>
  )
}
