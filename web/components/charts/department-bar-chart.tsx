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
  avg_salary?: number
  turnover_rate?: number
}

interface DepartmentBarChartProps {
  data: DepartmentData[]
  dataKey?: 'headcount' | 'avg_salary' | 'turnover_rate'
}

const COLORS = ['#3b82f6', '#60a5fa', '#93c5fd', '#bfdbfe', '#dbeafe']

export function DepartmentBarChart({
  data,
  dataKey = 'headcount',
}: DepartmentBarChartProps) {
  const chartData = data
    .filter((d) => d[dataKey] !== undefined && d[dataKey] !== null)
    .sort((a, b) => (b[dataKey] || 0) - (a[dataKey] || 0))
    .slice(0, 8)

  const formatValue = (value: number) => {
    if (dataKey === 'avg_salary') {
      return `$${value.toLocaleString()}`
    }
    if (dataKey === 'turnover_rate') {
      return `${(value * 100).toFixed(1)}%`
    }
    return value.toString()
  }

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart
        data={chartData}
        layout="vertical"
        margin={{ top: 5, right: 30, left: 80, bottom: 5 }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
        <XAxis type="number" stroke="var(--text-muted)" fontSize={12} />
        <YAxis
          type="category"
          dataKey="dept"
          stroke="var(--text-muted)"
          fontSize={12}
          width={75}
          tickLine={false}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: 'var(--surface)',
            border: '1px solid var(--border)',
            borderRadius: '8px',
            color: 'var(--text-primary)',
            fontSize: '12px',
            boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)',
          }}
          itemStyle={{ color: 'inherit' }}
          formatter={(value: number) => [formatValue(value), dataKey]}
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
