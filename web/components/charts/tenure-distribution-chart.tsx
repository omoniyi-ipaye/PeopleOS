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

const COLORS = ['#22c55e', '#3b82f6', '#60a5fa', '#93c5fd', '#f59e0b']

export function TenureDistributionChart({ data }: TenureDistributionChartProps) {
  // Sort data by tenure range order
  const sortOrder = ['<1 year', '1-2 years', '2-5 years', '5-10 years', '10+ years']
  const chartData = [...data].sort(
    (a, b) => sortOrder.indexOf(a.tenure_range) - sortOrder.indexOf(b.tenure_range)
  )

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart
        data={chartData}
        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
        <XAxis
          dataKey="tenure_range"
          stroke="var(--text-muted)"
          fontSize={12}
          tickLine={false}
        />
        <YAxis stroke="var(--text-muted)" fontSize={12} tickLine={false} />
        <Tooltip
          contentStyle={{
            backgroundColor: 'var(--surface)',
            border: '1px solid var(--border)',
            borderRadius: '8px',
            color: 'var(--text-primary)',
          }}
          formatter={(value: number, name: string) => {
            if (name === 'count') return [`${value} employees`, 'Count']
            if (name === 'turnover_rate')
              return [`${(value * 100).toFixed(1)}%`, 'Turnover Rate']
            return [value, name]
          }}
        />
        <Bar dataKey="count" radius={[4, 4, 0, 0]}>
          {chartData.map((_, index) => (
            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
