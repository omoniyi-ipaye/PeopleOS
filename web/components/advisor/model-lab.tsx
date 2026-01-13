'use client'

import React, { useState, useEffect } from 'react'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { LoadingState, ErrorState } from '@/components/ui/states'
import { api } from '@/lib/api-client'
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    Cell
} from 'recharts'
import { CheckCircle, AlertTriangle, Settings, RefreshCw, Zap, HelpCircle, Info } from 'lucide-react'

// HR-Friendly Glossary
const HR_GLOSSARY = {
    precision: {
        term: "Risk Accuracy",
        definition: "Of all employees flagged as 'High Risk', what percentage actually left? Higher is better.",
        example: "80% means 8 out of 10 flagged employees truly left."
    },
    recall: {
        term: "Risk Coverage",
        definition: "Of all employees who left, what percentage did we correctly flag in advance? Higher is better.",
        example: "70% means we caught 7 out of 10 departures ahead of time."
    },
    f1_score: {
        term: "Overall Reliability",
        definition: "A balanced score combining accuracy and coverage. This is the single best measure of model health.",
        example: "Above 70% = Excellent, 40-70% = Good, Below 40% = Needs more data."
    },
    true_positives: {
        term: "Correct Alerts",
        definition: "Employees the AI correctly identified as high-risk who then actually left the company."
    },
    missed_exits: {
        term: "Surprise Departures",
        definition: "Employees who left without being flagged as high-risk. These are the ones the model didn't catch."
    },
    importance: {
        term: "Predictive Power",
        definition: "How much this data point contributes to the AI's predictions. Higher = more influential on risk scores."
    },
    reliability: {
        term: "Data Consistency",
        definition: "How stable and trustworthy this data is. Low reliability suggests noisy or incomplete records."
    },
    noisy_features: {
        term: "Problem Data Fields",
        definition: "Data columns that are hurting prediction accuracy due to inconsistency or irrelevance."
    }
}

// Tooltip component for HR explanations
function HRTooltip({ term }: { term: keyof typeof HR_GLOSSARY }) {
    const [show, setShow] = useState(false)
    const glossary = HR_GLOSSARY[term]

    return (
        <div className="relative inline-block">
            <button
                onClick={() => setShow(!show)}
                onMouseEnter={() => setShow(true)}
                onMouseLeave={() => setShow(false)}
                className="ml-1 text-text-secondary hover:text-accent transition-colors"
            >
                <HelpCircle className="h-3.5 w-3.5" />
            </button>
            {show && (
                <div className="absolute z-50 left-0 bottom-full mb-2 w-64 p-3 bg-background dark:bg-background-dark border border-border dark:border-border-dark rounded-lg shadow-xl text-left">
                    <p className="font-semibold text-sm text-accent mb-1">{glossary.term}</p>
                    <p className="text-xs text-text-secondary mb-2">{glossary.definition}</p>
                    {'example' in glossary && (
                        <p className="text-[10px] italic text-text-secondary border-t border-border pt-2">
                            Example: {glossary.example}
                        </p>
                    )}
                </div>
            )}
        </div>
    )
}

export function ModelLab() {
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)
    const [validation, setValidation] = useState<any>(null)
    const [sensitivity, setSensitivity] = useState<any[]>([])
    const [refinementPlan, setRefinementPlan] = useState<any>(null)
    const [optimizing, setOptimizing] = useState(false)
    const [showGlossary, setShowGlossary] = useState(false)

    const fetchData = async () => {
        setLoading(true)
        try {
            const [valData, sensData, planData] = await Promise.all([
                api.modelLab.getValidation(90),
                api.modelLab.getSensitivity(),
                api.modelLab.getRefinementPlan()
            ])
            setValidation(valData)
            setSensitivity(sensData as any[])
            setRefinementPlan(planData)
            setError(null)
        } catch (err) {
            console.error('Failed to fetch model lab data:', err)
            setError('Unable to load model validation data. Please ensure you have uploaded historical data.')
        } finally {
            setLoading(false)
        }
    }

    useEffect(() => {
        fetchData()
    }, [])

    const handleOptimize = async () => {
        setOptimizing(true)
        try {
            await api.modelLab.optimize()
            await fetchData()
        } catch (err) {
            console.error('Optimization failed:', err)
        } finally {
            setOptimizing(false)
        }
    }

    if (loading) return <LoadingState message="Auditing AI accuracy against your historical data..." />
    if (error) return <ErrorState message={error} onRetry={fetchData} />

    const validationMetrics = validation?.metrics || {}
    // Use HR-friendly names in the chart
    const chartData = [
        { name: 'Risk Accuracy', value: (validationMetrics.precision || 0) * 100, key: 'precision' },
        { name: 'Risk Coverage', value: (validationMetrics.recall || 0) * 100, key: 'recall' },
        { name: 'Overall Reliability', value: (validationMetrics.f1_score || 0) * 100, key: 'f1_score' }
    ]

    return (
        <div className="space-y-6 animate-in fade-in duration-500">
            {/* Header with glossary toggle */}
            <div className="flex justify-between items-start">
                <div>
                    <h2 className="text-2xl font-bold tracking-tight text-text-primary dark:text-text-dark-primary">Model Validation Lab</h2>
                    <p className="text-text-secondary dark:text-text-dark-secondary">
                        Check how well the AI predicts who might leave your organization.
                    </p>
                </div>
                <div className="flex gap-2">
                    <Button
                        variant="ghost"
                        onClick={() => setShowGlossary(!showGlossary)}
                        className="text-sm"
                    >
                        <Info className="mr-2 h-4 w-4" />
                        {showGlossary ? 'Hide' : 'Show'} Glossary
                    </Button>
                    <Button
                        onClick={handleOptimize}
                        disabled={optimizing || refinementPlan?.status === 'healthy'}
                        className="shadow-lg"
                    >
                        {optimizing ? <RefreshCw className="mr-2 h-4 w-4 animate-spin" /> : <Zap className="mr-2 h-4 w-4" />}
                        {optimizing ? 'Optimizing...' : 'One-Click Refine'}
                    </Button>
                </div>
            </div>

            {/* HR Glossary Panel */}
            {showGlossary && (
                <Card className="bg-accent/5 border-accent/20">
                    <div className="p-4">
                        <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                            <Info className="h-5 w-5 text-accent" />
                            What Do These Terms Mean?
                        </h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
                            {Object.entries(HR_GLOSSARY).slice(0, 4).map(([key, val]) => (
                                <div key={key} className="p-3 bg-background dark:bg-background-dark rounded-lg border border-border dark:border-border-dark">
                                    <p className="font-semibold text-accent">{val.term}</p>
                                    <p className="text-xs text-text-secondary mt-1">{val.definition}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                </Card>
            )}

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Predictive Health Card - renamed for HR clarity */}
                <Card
                    title="AI Accuracy Check (90 Day Audit)"
                    subtitle={
                        validationMetrics.sample_size
                            ? `Compared ${validationMetrics.sample_size} past predictions to what actually happened.`
                            : "Waiting for more historical data to run a full audit..."
                    }
                    className="md:col-span-2"
                >
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                        <div className="h-[200px]">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={chartData} layout="vertical" margin={{ left: 30 }}>
                                    <CartesianGrid strokeDasharray="3 3" horizontal={false} opacity={0.3} />
                                    <XAxis type="number" domain={[0, 100]} hide />
                                    <YAxis dataKey="name" type="category" width={120} stroke="currentColor" fontSize={11} />
                                    <Tooltip
                                        formatter={(val: number) => [`${val.toFixed(1)}%`, 'Score']}
                                        contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
                                    />
                                    <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={24}>
                                        {chartData.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={index === 2 ? 'hsl(var(--accent))' : 'hsl(var(--accent) / 0.4)'} />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                        <div className="flex flex-col justify-center space-y-4 bg-muted/30 p-4 rounded-lg border border-border dark:border-border-dark">
                            <div className="flex items-start gap-3">
                                <CheckCircle className="h-5 w-5 text-success shrink-0 mt-0.5" />
                                <div>
                                    <p className="text-sm font-medium flex items-center">
                                        Correct Alerts: {validationMetrics.true_positives || 0}
                                        <HRTooltip term="true_positives" />
                                    </p>
                                    <p className="text-xs text-text-secondary">Employees correctly flagged as high-risk who then left.</p>
                                </div>
                            </div>
                            <div className="flex items-start gap-3">
                                <AlertTriangle className="h-5 w-5 text-warning shrink-0 mt-0.5" />
                                <div>
                                    <p className="text-sm font-medium flex items-center">
                                        Surprise Departures: {validationMetrics.missed_exits || 0}
                                        <HRTooltip term="missed_exits" />
                                    </p>
                                    <p className="text-xs text-text-secondary">People who left without being flagged as high-risk.</p>
                                </div>
                            </div>
                            <div className="p-3 bg-accent/5 rounded border border-accent/10">
                                <p className="text-[11px] font-semibold text-accent uppercase tracking-wider">What This Means</p>
                                <p className="text-xs">{validation?.interpretation || validation?.message || 'Building baseline — check back after 60-90 days of tracking.'}</p>
                            </div>
                        </div>
                    </div>
                </Card>

                {/* Refinement Plan Card */}
                <Card title="Improvement Opportunities" className="flex flex-col">
                    <div className="space-y-4 flex-1">
                        <div className="p-3 bg-muted/30 rounded-lg text-sm italic border border-border dark:border-border-dark">
                            "{refinementPlan?.reasoning}"
                        </div>

                        <div className="space-y-2">
                            <p className="text-xs font-semibold text-text-secondary uppercase tracking-wider flex items-center">
                                Suggested Actions
                            </p>
                            {refinementPlan?.suggested_actions?.map((action: string, i: number) => (
                                <div key={i} className="flex items-center gap-2 text-sm">
                                    <div className="h-1.5 w-1.5 rounded-full bg-accent" />
                                    {action}
                                </div>
                            ))}
                            {(!refinementPlan?.suggested_actions || refinementPlan.suggested_actions.length === 0) && (
                                <div className="text-sm text-success font-medium py-2">✓ Your data is clean — no improvements needed.</div>
                            )}
                        </div>

                        <div className="pt-4 border-t border-border dark:border-border-dark grid grid-cols-2 gap-2 text-center">
                            <div className="p-2 bg-accent/5 rounded border border-accent/10">
                                <p className="text-xl font-bold text-accent">{refinementPlan?.metrics?.noisy_features || 0}</p>
                                <p className="text-[10px] text-text-secondary uppercase flex items-center justify-center gap-1">
                                    Problem Fields
                                    <HRTooltip term="noisy_features" />
                                </p>
                            </div>
                            <div className="p-2 bg-accent/5 rounded border border-accent/10">
                                <p className="text-xl font-bold text-accent">{refinementPlan?.metrics?.estimated_accuracy_lift || '0%'}</p>
                                <p className="text-[10px] text-text-secondary uppercase">Potential Improvement</p>
                            </div>
                        </div>
                    </div>
                </Card>

                {/* Feature Sensitivity Table - with HR-friendly headers */}
                <Card
                    title="Data Quality Check"
                    subtitle="Which data fields drive predictions and how reliable is each one?"
                    className="md:col-span-3"
                >
                    <div className="relative overflow-x-auto rounded-lg border border-border dark:border-border-dark">
                        <table className="w-full text-sm text-left">
                            <thead className="text-xs text-text-secondary uppercase bg-muted/50 border-b border-border dark:border-border-dark">
                                <tr>
                                    <th className="px-6 py-3 font-semibold">Data Field</th>
                                    <th className="px-6 py-3 font-semibold text-center">
                                        <span className="flex items-center justify-center gap-1">
                                            Predictive Power
                                            <HRTooltip term="importance" />
                                        </span>
                                    </th>
                                    <th className="px-6 py-3 font-semibold text-center">
                                        <span className="flex items-center justify-center gap-1">
                                            Data Consistency
                                            <HRTooltip term="reliability" />
                                        </span>
                                    </th>
                                    <th className="px-6 py-3 font-semibold">Status</th>
                                    <th className="px-6 py-3 font-semibold">What To Do</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-border dark:divide-border-dark text-text-primary dark:text-text-dark-primary">
                                {sensitivity.map((feat, i) => (
                                    <tr key={i} className="hover:bg-muted/30 transition-colors">
                                        <td className="px-6 py-4 font-medium">{feat.feature}</td>
                                        <td className="px-6 py-4 text-center">
                                            <div className="flex items-center justify-center gap-2">
                                                <div className="w-16 h-1.5 bg-muted rounded-full overflow-hidden">
                                                    <div className="h-full bg-accent" style={{ width: `${feat.importance * 100}%` }} />
                                                </div>
                                                <span className="text-[11px] font-mono">{(feat.importance * 100).toFixed(1)}%</span>
                                            </div>
                                        </td>
                                        <td className="px-6 py-4 text-center text-[11px] font-mono">
                                            <Badge variant={feat.reliability > 0.8 ? 'outline' : 'info'} className="text-[10px]">
                                                {(feat.reliability * 100).toFixed(0)}%
                                            </Badge>
                                        </td>
                                        <td className="px-6 py-4">
                                            <Badge variant={feat.status === 'Stable' ? 'default' : 'warning'}>
                                                {feat.status === 'Stable' ? '✓ Healthy' : feat.status}
                                            </Badge>
                                        </td>
                                        <td className="px-6 py-4 text-text-secondary text-xs italic">{feat.recommendation}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </Card>
            </div>
        </div>
    )
}
