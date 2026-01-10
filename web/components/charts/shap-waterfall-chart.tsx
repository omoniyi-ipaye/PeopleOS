'use client'

import { useMemo } from 'react'

interface ShapFeature {
  feature: string
  value: number
  contribution: number
}

interface ShapWaterfallChartProps {
  features: ShapFeature[]
  baseValue: number
  prediction: number
}

export function ShapWaterfallChart({
  features,
  baseValue,
  prediction,
}: ShapWaterfallChartProps) {
  const chartData = useMemo(() => {
    // Sort by absolute contribution
    const sorted = [...features].sort(
      (a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)
    )

    // Calculate cumulative values
    let cumulative = baseValue
    const bars = sorted.map((f) => {
      const start = cumulative
      cumulative += f.contribution
      return {
        ...f,
        start,
        end: cumulative,
        isPositive: f.contribution >= 0,
      }
    })

    return bars
  }, [features, baseValue])

  const maxValue = Math.max(
    baseValue,
    prediction,
    ...chartData.map((d) => Math.max(d.start, d.end))
  )
  const minValue = Math.min(
    baseValue,
    prediction,
    ...chartData.map((d) => Math.min(d.start, d.end))
  )
  const range = maxValue - minValue || 1

  const getBarPosition = (value: number) => {
    return ((value - minValue) / range) * 100
  }

  const formatValue = (value: number) => {
    if (Math.abs(value) >= 1000) {
      return value.toFixed(0)
    }
    return value.toFixed(3)
  }

  return (
    <div className="space-y-3">
      {/* Base Value */}
      <div className="flex items-center gap-3 text-sm">
        <div className="w-32 text-text-secondary dark:text-text-dark-secondary truncate">Base Value</div>
        <div className="flex-1 relative h-6">
          <div
            className="absolute top-0 h-full w-1 bg-text-muted dark:bg-text-dark-muted"
            style={{ left: `${getBarPosition(baseValue)}%` }}
          />
          <div
            className="absolute top-1/2 -translate-y-1/2 text-xs text-text-muted dark:text-text-dark-muted whitespace-nowrap"
            style={{ left: `${getBarPosition(baseValue)}%`, marginLeft: '8px' }}
          >
            {formatValue(baseValue)}
          </div>
        </div>
      </div>

      {/* Feature Contributions */}
      {chartData.map((item, index) => (
        <div key={index} className="flex items-center gap-3 text-sm">
          <div className="w-32 text-text-secondary truncate" title={item.feature}>
            {item.feature}
          </div>
          <div className="flex-1 relative h-6">
            {/* Connector line */}
            <div
              className="absolute top-1/2 h-px bg-border dark:bg-border-dark"
              style={{
                left: `${Math.min(getBarPosition(item.start), getBarPosition(item.end))}%`,
                width: `${Math.abs(getBarPosition(item.end) - getBarPosition(item.start))}%`,
              }}
            />
            {/* Bar */}
            <div
              className={`absolute top-1 h-4 rounded ${item.isPositive ? 'bg-danger/80' : 'bg-success/80'
                }`}
              style={{
                left: `${Math.min(getBarPosition(item.start), getBarPosition(item.end))}%`,
                width: `${Math.max(1, Math.abs(getBarPosition(item.end) - getBarPosition(item.start)))}%`,
              }}
            />
            {/* Value label */}
            <div
              className={`absolute top-1/2 -translate-y-1/2 text-xs whitespace-nowrap ${item.isPositive ? 'text-danger' : 'text-success'
                }`}
              style={{
                left: `${getBarPosition(item.end)}%`,
                marginLeft: item.isPositive ? '8px' : '-40px',
              }}
            >
              {item.isPositive ? '+' : ''}
              {formatValue(item.contribution)}
            </div>
          </div>
          <div className="w-16 text-right text-xs text-text-muted dark:text-text-dark-muted">
            = {item.value !== null ? formatValue(item.value) : 'N/A'}
          </div>
        </div>
      ))}

      {/* Prediction Value */}
      <div className="flex items-center gap-3 text-sm pt-2 border-t border-border dark:border-border-dark">
        <div className="w-32 font-medium text-text-primary dark:text-text-dark-primary">Prediction</div>
        <div className="flex-1 relative h-6">
          <div
            className="absolute top-0 h-full w-2 bg-accent rounded shadow-[0_0_8px_rgba(59,130,246,0.5)]"
            style={{ left: `${getBarPosition(prediction)}%` }}
          />
          <div
            className="absolute top-1/2 -translate-y-1/2 text-xs font-bold text-accent whitespace-nowrap"
            style={{ left: `${getBarPosition(prediction)}%`, marginLeft: '16px' }}
          >
            {formatValue(prediction)}
          </div>
        </div>
        <div className="w-16" />
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center gap-6 pt-4 text-xs text-text-secondary dark:text-text-dark-secondary font-medium">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded bg-danger/80" />
          <span>Increases Risk</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded bg-success/80" />
          <span>Decreases Risk</span>
        </div>
      </div>
    </div>
  )
}
