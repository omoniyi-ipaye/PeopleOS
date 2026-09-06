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

interface TenureData {
  tenure_range: string
  count: number
  turnover_rate?: number
}

interface TenureDistributionChartProps {
  data: TenureData[]
}

const COLORS = [
  'var(--chart-success, var(--success))',
  'var(--chart-accent, var(--accent))',
  'var(--chart-accent-soft, var(--accent))',
  'var(--chart-accent-muted, var(--accent))',
  'var(--chart-warning, var(--warning))',
]

export function TenureDistributionChart({ data }: TenureDistributionChartProps) {
  const sortOrder = ['<1 year', '1-2 years', '2-5 years', '5-10 years', '10+ years']
  const chartData = [...data].sort(
    (a, b) => sortOrder.indexOf(a.tenure_range) - sortOrder.indexOf(b.tenure_range)
  )

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
        <XAxis dataKey="tenure_range" stroke="var(--text-muted)" fontSize={12} tickLine={false} />
        <YAxis stroke="var(--text-muted)" fontSize={12} tickLine={false} />
        <Tooltip
          contentStyle={{
            backgroundColor: 'var(--surface)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-control, 8px)',
            color: 'var(--text-primary)',
          }}
          formatter={(value, name) => {
            const numeric = Number(value ?? 0)
            const key = String(name ?? '')
            if (key === 'count') return [`${numeric} employees`, 'Count']
            if (key === 'turnover_rate') return [`${(numeric * 100).toFixed(1)}%`, 'Turnover Rate']
            return [numeric, key]
          }}
        />
        <Bar dataKey="count" radius={[4, 4, 0, 0]}>
          {chartData.map((_, index) => <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
