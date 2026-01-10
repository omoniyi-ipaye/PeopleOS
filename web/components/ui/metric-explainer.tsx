'use client'

import { cn } from '@/lib/utils'
import { HelpCircle, TrendingUp, TrendingDown, Minus } from 'lucide-react'
import { useState } from 'react'

interface MetricExplainerProps {
  title: string
  value: string | number
  explanation: string
  benchmark?: string
  status?: 'good' | 'warning' | 'critical' | 'neutral'
  trend?: {
    direction: 'up' | 'down' | 'flat'
    value: string
    isGood?: boolean
  }
  className?: string
  compact?: boolean
}

const statusColors = {
  good: 'bg-success/10 border-success/30 text-success',
  warning: 'bg-warning/10 border-warning/30 text-warning',
  critical: 'bg-danger/10 border-danger/30 text-danger',
  neutral: 'bg-surface-secondary border-border text-text-secondary',
}

const statusDot = {
  good: 'bg-success',
  warning: 'bg-warning',
  critical: 'bg-danger',
  neutral: 'bg-text-muted',
}

export function MetricExplainer({
  title,
  value,
  explanation,
  benchmark,
  status = 'neutral',
  trend,
  className,
  compact = false
}: MetricExplainerProps) {
  const [showExplanation, setShowExplanation] = useState(false)

  if (compact) {
    return (
      <div className={cn("relative group", className)}>
        <div className="flex items-center gap-2">
          <span className="text-2xl font-bold">{value}</span>
          <HelpCircle
            className="w-4 h-4 text-text-muted cursor-help"
            onMouseEnter={() => setShowExplanation(true)}
            onMouseLeave={() => setShowExplanation(false)}
          />
        </div>
        {showExplanation && (
          <div className="absolute z-50 top-full left-0 mt-2 p-3 bg-surface dark:bg-surface-dark border border-border dark:border-border-dark rounded-lg shadow-lg max-w-xs">
            <p className="text-sm text-text-secondary dark:text-text-dark-secondary">{explanation}</p>
            {benchmark && (
              <p className="text-xs text-text-muted dark:text-text-dark-muted mt-2 pt-2 border-t border-border dark:border-border-dark">
                Benchmark: {benchmark}
              </p>
            )}
          </div>
        )}
      </div>
    )
  }

  return (
    <div className={cn(
      "p-4 rounded-lg border",
      statusColors[status],
      className
    )}>
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center gap-2">
          <div className={cn("w-2 h-2 rounded-full", statusDot[status])} />
          <span className="font-medium text-text-primary dark:text-text-dark-primary">{title}</span>
        </div>
        <div className="flex items-center gap-2">
          {trend && (
            <div className={cn(
              "flex items-center gap-1 text-xs",
              trend.isGood ? 'text-success' : trend.direction !== 'flat' ? 'text-danger' : 'text-text-muted'
            )}>
              {trend.direction === 'up' && <TrendingUp className="w-3 h-3" />}
              {trend.direction === 'down' && <TrendingDown className="w-3 h-3" />}
              {trend.direction === 'flat' && <Minus className="w-3 h-3" />}
              <span>{trend.value}</span>
            </div>
          )}
          <span className="text-xl font-mono font-bold">{value}</span>
        </div>
      </div>

      <p className="text-sm text-text-secondary dark:text-text-dark-secondary">{explanation}</p>

      {benchmark && (
        <p className="text-xs text-text-muted dark:text-text-dark-muted mt-3 pt-2 border-t border-current/20">
          Benchmark: {benchmark}
        </p>
      )}
    </div>
  )
}

// Inline explanation component for use within tables or lists
interface InlineExplainerProps {
  value: string | number
  explanation: string
  status?: 'good' | 'warning' | 'critical' | 'neutral'
}

export function InlineExplainer({ value, explanation, status = 'neutral' }: InlineExplainerProps) {
  const [showTip, setShowTip] = useState(false)

  const valueColors = {
    good: 'text-success',
    warning: 'text-warning',
    critical: 'text-danger',
    neutral: 'text-text-primary dark:text-text-dark-primary',
  }

  return (
    <div className="relative inline-flex items-center gap-1">
      <span className={cn("font-mono", valueColors[status])}>{value}</span>
      <HelpCircle
        className="w-3 h-3 text-text-muted cursor-help"
        onMouseEnter={() => setShowTip(true)}
        onMouseLeave={() => setShowTip(false)}
      />
      {showTip && (
        <div className="absolute z-50 bottom-full left-0 mb-2 p-2 bg-slate-900 text-white text-xs rounded shadow-lg max-w-xs whitespace-normal">
          {explanation}
        </div>
      )}
    </div>
  )
}

// Correlation interpretation badge
interface CorrelationBadgeProps {
  value: number
  showValue?: boolean
}

export function CorrelationBadge({ value, showValue = true }: CorrelationBadgeProps) {
  const absValue = Math.abs(value)
  let label: string
  let colorClass: string

  if (absValue >= 0.5) {
    label = 'Strong'
    colorClass = 'bg-success/10 text-success border-success/30'
  } else if (absValue >= 0.3) {
    label = 'Moderate'
    colorClass = 'bg-accent/10 text-accent border-accent/30'
  } else if (absValue >= 0.1) {
    label = 'Weak'
    colorClass = 'bg-warning/10 text-warning border-warning/30'
  } else {
    label = 'None'
    colorClass = 'bg-surface-secondary text-text-muted border-border'
  }

  return (
    <span className={cn(
      "inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs border",
      colorClass
    )}>
      {showValue && <span className="font-mono">{value.toFixed(2)}</span>}
      <span>{label}</span>
    </span>
  )
}

// Hazard Ratio interpretation
interface HazardRatioDisplayProps {
  value: number
  factor: string
}

export function HazardRatioDisplay({ value, factor }: HazardRatioDisplayProps) {
  let interpretation: string
  let colorClass: string

  if (value > 1) {
    const increase = Math.round((value - 1) * 100)
    interpretation = `${increase}% more likely to leave`
    colorClass = 'text-danger'
  } else if (value < 1) {
    const decrease = Math.round((1 - value) * 100)
    interpretation = `${decrease}% less likely to leave`
    colorClass = 'text-success'
  } else {
    interpretation = 'No effect'
    colorClass = 'text-text-muted'
  }

  return (
    <div className="flex items-center justify-between py-2">
      <span className="text-text-secondary dark:text-text-dark-secondary">{factor}</span>
      <div className="flex items-center gap-3">
        <span className={cn("font-mono", colorClass)}>{value.toFixed(2)}</span>
        <span className={cn("text-xs", colorClass)}>{interpretation}</span>
      </div>
    </div>
  )
}

// Explanation box for complex concepts
interface ExplanationBoxProps {
  title: string
  children: React.ReactNode
  variant?: 'info' | 'tip' | 'warning'
  className?: string
}

export function ExplanationBox({ title, children, variant = 'info', className }: ExplanationBoxProps) {
  const variantStyles = {
    info: 'bg-accent/5 border-accent/20',
    tip: 'bg-success/5 border-success/20',
    warning: 'bg-warning/5 border-warning/20',
  }

  return (
    <div className={cn(
      "rounded-lg border p-4",
      variantStyles[variant],
      className
    )}>
      <h4 className="font-bold text-sm mb-2 text-text-primary dark:text-text-dark-primary">{title}</h4>
      <div className="text-sm text-text-secondary dark:text-text-dark-secondary">
        {children}
      </div>
    </div>
  )
}

// Risk interpretation guide
export function RiskFactorGuide() {
  return (
    <ExplanationBox title="Understanding Risk Factors" variant="info">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-2">
        <div>
          <span className="text-danger font-mono font-bold">Risk Multiplier &gt; 1.0</span>
          <p className="text-text-secondary mt-1">Increases likelihood of leaving</p>
          <p className="text-xs text-text-muted mt-1">Example: 1.35 = 35% more likely to leave</p>
        </div>
        <div>
          <span className="text-success font-mono font-bold">Risk Multiplier &lt; 1.0</span>
          <p className="text-text-secondary mt-1">Decreases likelihood of leaving</p>
          <p className="text-xs text-text-muted mt-1">Example: 0.72 = 28% less likely to leave</p>
        </div>
      </div>
    </ExplanationBox>
  )
}

// Correlation interpretation guide
export function CorrelationGuide() {
  return (
    <div className="text-xs text-text-muted p-3 bg-surface-secondary dark:bg-surface-dark-secondary rounded-lg">
      <strong className="text-text-primary dark:text-text-dark-primary">Reading Correlations:</strong>
      <ul className="mt-2 space-y-1">
        <li className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-success" />
          <span><span className="text-success font-medium">0.5 to 1.0</span> = Strong positive relationship</span>
        </li>
        <li className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-accent" />
          <span><span className="text-accent font-medium">0.3 to 0.5</span> = Moderate relationship</span>
        </li>
        <li className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-warning" />
          <span><span className="text-warning font-medium">0.1 to 0.3</span> = Weak relationship</span>
        </li>
        <li className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-text-muted" />
          <span><span className="text-text-muted font-medium">Below 0.1</span> = No meaningful relationship</span>
        </li>
      </ul>
    </div>
  )
}
