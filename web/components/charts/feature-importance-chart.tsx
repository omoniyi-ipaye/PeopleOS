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

interface FeatureImportance {
  feature: string
  importance: number
}

interface FeatureImportanceChartProps {
  data: FeatureImportance[]
}

export function FeatureImportanceChart({ data }: FeatureImportanceChartProps) {
  const chartData = data.slice(0, 10).map((d, i) => ({
    ...d,
    importance: d.importance * 100, // Convert to percentage
    color: i === 0 ? '#ef4444' : i < 3 ? '#f59e0b' : '#3b82f6',
  }))

  return (
    <ResponsiveContainer width="100%" height={350}>
      <BarChart
        data={chartData}
        layout="vertical"
        margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" horizontal={false} />
        <XAxis
          type="number"
          stroke="var(--chart-axis)"
          fontSize={12}
          tickFormatter={(value) => `${value.toFixed(0)}%`}
        />
        <YAxis
          type="category"
          dataKey="feature"
          stroke="var(--chart-axis)"
          fontSize={12}
          width={95}
          tickLine={false}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: 'var(--surface)',
            border: '1px solid var(--border)',
            borderRadius: '8px',
            color: 'var(--text-primary)',
          }}
          itemStyle={{ color: 'inherit' }}
          formatter={(value: number) => [`${value.toFixed(1)}%`, 'Importance']}
        />
        <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
          {chartData.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={entry.color} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
