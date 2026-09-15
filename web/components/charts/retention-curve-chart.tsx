'use client'

import {
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

    const chartData = data.map(point => ({
        ...point,
        time_years: point.time_months / 12
    }))

    const firstYear = chartData[0]?.time_years ?? 0
    const lastYear = chartData[chartData.length - 1]?.time_years ?? firstYear

    return (
        <div
            role="img"
            aria-label={`Recorded retention history chart showing the share of the cohort still here by tenure, from ${firstYear.toFixed(1)} to ${lastYear.toFixed(1)} years.`}
        >
        <ResponsiveContainer width="100%" height={300}>
            <ComposedChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
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
                    tickFormatter={(value) => `${(Number(value) * 100).toFixed(0)}%`}
                    label={{ value: 'Share still here', angle: -90, position: 'insideLeft', fill: 'var(--text-muted)', fontSize: 10 }}
                />
                <Tooltip
                    contentStyle={{
                        backgroundColor: 'var(--surface)',
                        border: '1px solid var(--border)',
                        borderRadius: '8px',
                        color: 'var(--text-primary)',
                    }}
                    formatter={(value, name) => {
                        const numeric = Number(value ?? 0)
                        const key = String(name ?? '')
                        if (key === 'survival_probability') return [`${(numeric * 100).toFixed(1)}%`, 'Share still here']
                        if (key === 'at_risk') return [numeric, 'People still in recorded follow-up']
                        return [numeric, key]
                    }}
                    labelFormatter={(label) => `Tenure: ${Number(label).toFixed(1)} years`}
                />
                <Area type="stepAfter" dataKey="survival_probability" fill="var(--accent)" fillOpacity={0.1} stroke="none" />
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
        </div>
    )
}
