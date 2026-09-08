'use client'

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts'

interface DepartmentData {
  dept: string
  headcount: number
  avg_salary?: number | null
  turnover_rate?: number | null
}

interface DepartmentBarChartProps {
  data: DepartmentData[]
  dataKey?: 'headcount' | 'avg_salary' | 'turnover_rate'
}

const COLORS = [
  'var(--chart-accent-strong, var(--accent))',
  'var(--chart-accent, var(--accent))',
  'var(--chart-accent-soft, var(--accent))',
  'var(--chart-accent-muted, var(--accent))',
  'var(--chart-accent-faint, var(--accent))',
]

export function DepartmentBarChart({ data, dataKey = 'headcount' }: DepartmentBarChartProps) {
  const chartData = data
    .filter((d) => d[dataKey] != null && Number.isFinite(d[dataKey]))
    .sort((a, b) => (b[dataKey] || 0) - (a[dataKey] || 0))

  if (!chartData.length) return <p role="status">Department measurements are unavailable for this metric.</p>
  const metricLabel = dataKey === 'turnover_rate' ? 'Observed attrition share (not period turnover)' : dataKey === 'avg_salary' ? 'Mean recorded salary (source units)' : 'Active employees'

  const formatValue = (value: number) => {
    if (dataKey === 'avg_salary') return value.toLocaleString()
    if (dataKey === 'turnover_rate') return `${(value * 100).toFixed(1)}%`
    return value.toString()
  }

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 80, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
        <XAxis type="number" stroke="var(--text-muted)" fontSize={12} tickFormatter={formatValue} />
        <YAxis type="category" dataKey="dept" stroke="var(--text-muted)" fontSize={12} width={75} tickLine={false} />
        <Tooltip
          contentStyle={{
            backgroundColor: 'var(--surface)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-control, 8px)',
            color: 'var(--text-primary)',
            fontSize: '12px',
            boxShadow: 'var(--shadow-raised)',
          }}
          itemStyle={{ color: 'inherit' }}
          formatter={(value) => [value == null || !Number.isFinite(Number(value)) ? 'Unavailable' : formatValue(Number(value)), metricLabel]}
        />
        <Bar dataKey={dataKey} radius={[0, 4, 4, 0]}>
          {chartData.map((_, index) => (
            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
