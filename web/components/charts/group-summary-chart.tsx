'use client'

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

export interface GroupSummaryChartDatum {
  group: string
  value: number
}

interface GroupSummaryChartProps {
  data: GroupSummaryChartDatum[]
  valueLabel: string
  formatValue: (value: number) => string
}

const COLORS = [
  'var(--chart-accent-strong, var(--accent))',
  'var(--chart-accent, var(--accent))',
  'var(--chart-accent-soft, var(--accent))',
  'var(--chart-accent-muted, var(--accent))',
  'var(--chart-accent-faint, var(--accent))',
]

function shortLabel(value: string) {
  return value.length > 18 ? `${value.slice(0, 17)}…` : value
}

export function GroupSummaryChart({ data, valueLabel, formatValue }: GroupSummaryChartProps) {
  const chartData = data.filter(item => Number.isFinite(item.value))

  if (!chartData.length) return <p role="status" className="text-sm text-text-secondary">This breakdown is unavailable for the current data.</p>

  return (
    <div className="h-[300px] w-full" aria-label={`${valueLabel} chart`}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} layout="vertical" margin={{ top: 8, right: 24, left: 8, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
          <XAxis type="number" stroke="var(--text-muted)" fontSize={12} tickFormatter={formatValue} tickLine={false} />
          <YAxis type="category" dataKey="group" stroke="var(--text-muted)" fontSize={12} width={112} tickFormatter={shortLabel} tickLine={false} />
          <Tooltip
            cursor={{ fill: 'var(--surface-secondary)' }}
            contentStyle={{
              backgroundColor: 'var(--surface)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius-control, 8px)',
              color: 'var(--text-primary)',
              fontSize: '12px',
              boxShadow: 'var(--shadow-raised)',
            }}
            formatter={(value) => [formatValue(Number(value ?? 0)), valueLabel]}
          />
          <Bar dataKey="value" radius={[0, 6, 6, 0]}>
            {chartData.map((item, index) => <Cell key={item.group} fill={COLORS[index % COLORS.length]} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
