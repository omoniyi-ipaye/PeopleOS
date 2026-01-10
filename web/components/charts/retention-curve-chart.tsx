'use client'

import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Area,
    ComposedChart
} from 'recharts'
import type { SurvivalPoint } from '@/types/api'

interface RetentionCurveChartProps {
    data: SurvivalPoint[]
}

export function RetentionCurveChart({ data }: RetentionCurveChartProps) {
    if (!data || data.length === 0) return null

    // Ensure time_years is computed if not present
    const chartData = data.map(point => ({
        ...point,
        time_years: point.time_years ?? point.time_months / 12
    }))

    return (
        <ResponsiveContainer width="100%" height={300}>
            <ComposedChart
                data={chartData}
                margin={{ top: 10, right: 30, left: 10, bottom: 10 }}
            >
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                <XAxis
                    dataKey="time_years"
                    type="number"
                    domain={[0, 'dataMax']}
                    name="Tenure"
                    unit="y"
                    stroke="var(--text-muted)"
                    fontSize={12}
                    tickLine={false}
                    label={{ value: 'Tenure (Years)', position: 'insideBottom', offset: -5, fill: 'var(--text-muted)', fontSize: 10 }}
                />
                <YAxis
                    stroke="var(--text-muted)"
                    fontSize={12}
                    tickLine={false}
                    domain={[0, 1]}
                    tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                    label={{ value: 'Survival Probability', angle: -90, position: 'insideLeft', fill: 'var(--text-muted)', fontSize: 10 }}
                />
                <Tooltip
                    contentStyle={{
                        backgroundColor: 'var(--surface)',
                        border: '1px solid var(--border)',
                        borderRadius: '8px',
                        color: 'var(--text-primary)',
                    }}
                    formatter={(value: number, name: string) => {
                        if (name === 'survival_probability') return [`${(value * 100).toFixed(1)}%`, 'Retention Chance']
                        if (name === 'at_risk') return [value, 'Employees at Risk']
                        return [value, name]
                    }}
                    labelFormatter={(label) => `Tenure: ${label} years`}
                />
                <Area
                    type="stepAfter"
                    dataKey="survival_probability"
                    fill="var(--accent)"
                    fillOpacity={0.1}
                    stroke="none"
                />
                <Line
                    type="stepAfter"
                    dataKey="survival_probability"
                    stroke="var(--accent)"
                    strokeWidth={3}
                    dot={false}
                    activeDot={{ r: 6 }}
                />
            </ComposedChart>
        </ResponsiveContainer>
    )
}
