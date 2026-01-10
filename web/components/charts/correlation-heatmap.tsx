'use client'

import { cn } from '@/lib/utils'

interface CorrelationData {
  feature: string
  correlation: number
  abs_correlation: number
}

interface CorrelationHeatmapProps {
  data: CorrelationData[]
}

function getCorrelationColor(value: number): string {
  const absValue = Math.abs(value)
  if (value > 0) {
    // Positive correlation - red shades (risk increases)
    if (absValue > 0.3) return 'bg-danger/40'
    if (absValue > 0.2) return 'bg-danger/30'
    if (absValue > 0.1) return 'bg-danger/20'
    return 'bg-danger/10'
  } else {
    // Negative correlation - green shades (risk decreases)
    if (absValue > 0.3) return 'bg-success/40'
    if (absValue > 0.2) return 'bg-success/30'
    if (absValue > 0.1) return 'bg-success/20'
    return 'bg-success/10'
  }
}

export function CorrelationHeatmap({ data }: CorrelationHeatmapProps) {
  if (!data || data.length === 0) {
    return (
      <div className="text-center text-text-secondary py-8">
        No correlation data available
      </div>
    )
  }

  return (
    <div className="space-y-2">
      {data.slice(0, 10).map((item) => (
        <div
          key={item.feature}
          className="flex items-center gap-3"
        >
          <div className="w-32 text-sm text-text-secondary truncate">
            {item.feature}
          </div>
          <div className="flex-1 h-8 flex items-center">
            <div
              className={cn(
                'h-full rounded-r transition-all flex items-center justify-end px-2',
                getCorrelationColor(item.correlation)
              )}
              style={{
                width: `${Math.abs(item.correlation) * 100 * 3}%`,
                minWidth: '20px',
              }}
            >
              <span className="text-xs font-mono text-white">
                {item.correlation > 0 ? '+' : ''}
                {(item.correlation * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        </div>
      ))}

      {/* Legend */}
      <div className="flex items-center justify-center gap-6 mt-4 pt-4 border-t border-border-dark">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded bg-danger/30" />
          <span className="text-xs text-text-muted">Increases Risk</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded bg-success/30" />
          <span className="text-xs text-text-muted">Decreases Risk</span>
        </div>
      </div>
    </div>
  )
}
