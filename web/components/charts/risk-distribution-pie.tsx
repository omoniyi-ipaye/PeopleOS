'use client'

import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Legend,
  Tooltip,
} from 'recharts'
import { getRiskColor } from '@/lib/utils'

interface RiskDistribution {
  high_risk: number
  medium_risk: number
  low_risk: number
  high_risk_pct: number
  medium_risk_pct: number
  low_risk_pct: number
}

interface RiskDistributionPieProps {
  data: RiskDistribution
}

export function RiskDistributionPie({ data }: RiskDistributionPieProps) {
  const chartData = [
    { name: 'High Risk', value: data.high_risk, color: getRiskColor('high') },
    { name: 'Medium Risk', value: data.medium_risk, color: getRiskColor('medium') },
    { name: 'Low Risk', value: data.low_risk, color: getRiskColor('low') },
  ].filter((d) => d.value > 0)

  if (chartData.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-text-secondary dark:text-text-dark-secondary">
        No risk data available
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={300}>
      <PieChart>
        <Pie
          data={chartData}
          cx="50%"
          cy="50%"
          innerRadius={60}
          outerRadius={90}
          paddingAngle={2}
          dataKey="value"
          label={({ name, percent }) =>
            `${name} (${(percent * 100).toFixed(0)}%)`
          }
          labelLine={false}
        >
          {chartData.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={entry.color} />
          ))}
        </Pie>
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
          formatter={(value: number, name: string) => [
            `${value} employees`,
            name,
          ]}
        />
        <Legend
          verticalAlign="bottom"
          height={36}
          formatter={(value) => (
            <span className="text-xs text-text-secondary dark:text-text-dark-secondary">{value}</span>
          )}
        />
      </PieChart>
    </ResponsiveContainer>
  )
}
