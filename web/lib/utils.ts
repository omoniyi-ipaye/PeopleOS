import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatNumber(num: number): string {
  return num.toLocaleString('en-US')
}

export function formatCurrency(num: number): string {
  return `$${num.toLocaleString('en-US', { maximumFractionDigits: 0 })}`
}

export function formatPercent(num: number, decimals = 1): string {
  return `${(num * 100).toFixed(decimals)}%`
}

export function getRiskColor(category: string): string {
  switch (category.toLowerCase()) {
    case 'high':
      return '#ef4444' // danger
    case 'medium':
      return '#f59e0b' // warning
    case 'low':
      return '#22c55e' // success
    default:
      return '#3b82f6' // accent
  }
}

export function getRiskBgColor(category: string): string {
  switch (category.toLowerCase()) {
    case 'high':
      return 'bg-danger-50 dark:bg-danger/10'
    case 'medium':
      return 'bg-warning-50 dark:bg-warning/10'
    case 'low':
      return 'bg-success-50 dark:bg-success/10'
    default:
      return 'bg-accent-50 dark:bg-accent/10'
  }
}

export function getRiskTextColor(category: string): string {
  switch (category.toLowerCase()) {
    case 'high':
      return 'text-danger'
    case 'medium':
      return 'text-warning'
    case 'low':
      return 'text-success'
    default:
      return 'text-accent'
  }
}
