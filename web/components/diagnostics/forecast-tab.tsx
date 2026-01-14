'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { GlassCard } from '@/components/ui/glass-card'
import { Badge } from '@/components/ui/badge'
import { api } from '@/lib/api-client'
import { TrendingUp, TrendingDown, Calendar, AlertTriangle, ArrowRight } from 'lucide-react'
import { cn } from '@/lib/utils'
import {
    AreaChart,
    Area,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Legend,
} from 'recharts'

interface ForecastPoint {
    date: string
    value: number
    lower?: number
    upper?: number
}

interface ForecastResult {
    success: boolean
    metric: string
    history: ForecastPoint[]
    forecast: ForecastPoint[]
    reason?: string
}

type MetricType = 'headcount' | 'Salary'

export function ForecastTab() {
    const [metric, setMetric] = useState<MetricType>('headcount')

    const { data, isLoading, isError } = useQuery<ForecastResult>({
        queryKey: ['analytics', 'forecast', metric],
        queryFn: () => api.analytics.getForecast(metric, 12) as Promise<ForecastResult>,
    })

    if (isLoading) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="animate-pulse-subtle text-text-secondary dark:text-text-dark-secondary">
                    Generating forecast...
                </div>
            </div>
        )
    }

    if (isError || !data?.success) {
        return (
            <GlassCard className="p-6">
                <div className="text-center text-text-secondary dark:text-text-dark-secondary">
                    <AlertTriangle className="w-8 h-8 mx-auto mb-2 text-amber-500" />
                    <p>Unable to generate forecast. {data?.reason || 'Ensure sufficient historical data is available.'}</p>
                </div>
            </GlassCard>
        )
    }

    // Combine history and forecast for the chart
    const chartData = [
        ...data.history.map(p => ({
            date: new Date(p.date).toLocaleDateString('en-US', { month: 'short', year: '2-digit' }),
            actual: p.value,
            forecast: null,
            lower: null,
            upper: null,
            isHistory: true
        })),
        ...data.forecast.map(p => ({
            date: new Date(p.date).toLocaleDateString('en-US', { month: 'short', year: '2-digit' }),
            actual: null,
            forecast: p.value,
            lower: p.lower,
            upper: p.upper,
            isHistory: false
        }))
    ]

    // Calculate trend
    const lastHistorical = data.history[data.history.length - 1]?.value || 0
    const lastForecast = data.forecast[data.forecast.length - 1]?.value || 0
    const trend = lastForecast - lastHistorical
    const trendPercent = lastHistorical > 0 ? ((trend / lastHistorical) * 100).toFixed(1) : 0
    const isPositive = trend >= 0

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-2xl font-display font-semibold text-gray-900 dark:text-white">
                        Future Trends
                    </h2>
                    <p className="text-text-secondary dark:text-text-dark-secondary text-sm mt-1">
                        12-month forecast based on historical patterns
                    </p>
                </div>
                {/* Metric Toggle */}
                <div className="glass p-1 rounded-xl flex gap-1">
                    <button
                        onClick={() => setMetric('headcount')}
                        className={cn(
                            "px-4 py-2 rounded-lg text-sm font-medium transition-all",
                            metric === 'headcount'
                                ? 'bg-white dark:bg-slate-800 shadow-lg text-text-primary dark:text-white'
                                : 'text-text-secondary dark:text-slate-400 hover:text-text-primary'
                        )}
                    >
                        Headcount
                    </button>
                    <button
                        onClick={() => setMetric('Salary')}
                        className={cn(
                            "px-4 py-2 rounded-lg text-sm font-medium transition-all",
                            metric === 'Salary'
                                ? 'bg-white dark:bg-slate-800 shadow-lg text-text-primary dark:text-white'
                                : 'text-text-secondary dark:text-slate-400 hover:text-text-primary'
                        )}
                    >
                        Avg Salary
                    </button>
                </div>
            </div>

            {/* Trend Summary */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <GlassCard className="p-4">
                    <div className="flex items-center gap-3">
                        <div className={cn(
                            "w-10 h-10 rounded-xl flex items-center justify-center",
                            isPositive ? "bg-emerald-500/20 text-emerald-500" : "bg-red-500/20 text-red-500"
                        )}>
                            {isPositive ? <TrendingUp className="w-5 h-5" /> : <TrendingDown className="w-5 h-5" />}
                        </div>
                        <div>
                            <p className="text-sm text-text-secondary dark:text-text-dark-secondary">12-Month Trend</p>
                            <p className={cn(
                                "text-lg font-semibold",
                                isPositive ? "text-emerald-600 dark:text-emerald-400" : "text-red-600 dark:text-red-400"
                            )}>
                                {isPositive ? '+' : ''}{trendPercent}%
                            </p>
                        </div>
                    </div>
                </GlassCard>

                <GlassCard className="p-4">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl flex items-center justify-center bg-blue-500/20 text-blue-500">
                            <Calendar className="w-5 h-5" />
                        </div>
                        <div>
                            <p className="text-sm text-text-secondary dark:text-text-dark-secondary">Current</p>
                            <p className="text-lg font-semibold text-gray-900 dark:text-white">
                                {metric === 'headcount' ? lastHistorical.toFixed(0) : `$${(lastHistorical / 1000).toFixed(0)}k`}
                            </p>
                        </div>
                    </div>
                </GlassCard>

                <GlassCard className="p-4">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl flex items-center justify-center bg-purple-500/20 text-purple-500">
                            <ArrowRight className="w-5 h-5" />
                        </div>
                        <div>
                            <p className="text-sm text-text-secondary dark:text-text-dark-secondary">Projected</p>
                            <p className="text-lg font-semibold text-gray-900 dark:text-white">
                                {metric === 'headcount' ? lastForecast.toFixed(0) : `$${(lastForecast / 1000).toFixed(0)}k`}
                            </p>
                        </div>
                    </div>
                </GlassCard>
            </div>

            {/* Chart */}
            <GlassCard className="p-6">
                <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
                    {metric === 'headcount' ? 'Headcount Projection' : 'Average Salary Projection'}
                </h3>
                <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                            <defs>
                                <linearGradient id="colorActual" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                                </linearGradient>
                                <linearGradient id="colorForecast" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                            <XAxis dataKey="date" stroke="#9ca3af" fontSize={12} />
                            <YAxis stroke="#9ca3af" fontSize={12} />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: 'rgba(0,0,0,0.8)',
                                    borderRadius: '8px',
                                    border: 'none',
                                    color: '#fff'
                                }}
                            />
                            <Legend />
                            <Area
                                type="monotone"
                                dataKey="actual"
                                stroke="#3b82f6"
                                fillOpacity={1}
                                fill="url(#colorActual)"
                                name="Historical"
                                connectNulls={false}
                            />
                            <Area
                                type="monotone"
                                dataKey="forecast"
                                stroke="#10b981"
                                fillOpacity={1}
                                fill="url(#colorForecast)"
                                name="Forecast"
                                strokeDasharray="5 5"
                                connectNulls={false}
                            />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>
            </GlassCard>
        </div>
    )
}
