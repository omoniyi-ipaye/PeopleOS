'use client'

import React from 'react'
import {
    Brain,
    X,
    TrendingDown,
    Layers,
    ShieldCheck,
    Percent,
    Calculator,
} from 'lucide-react'
import { Badge } from '../ui/badge'

interface PredictiveMathModalProps {
    onClose: () => void
}

export function PredictiveMathModal({ onClose }: PredictiveMathModalProps) {
    return (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-background/80 dark:bg-background-dark/80 backdrop-blur-sm animate-in fade-in duration-200">
            <div className="bg-surface dark:bg-surface-dark border border-border dark:border-border-dark rounded-2xl shadow-2xl w-full max-w-2xl overflow-hidden animate-in zoom-in-95 duration-200">
                {/* Header */}
                <div className="px-6 py-4 border-b border-border dark:border-border-dark flex items-center justify-between bg-accent/5">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-accent/10 flex items-center justify-center">
                            <Calculator className="w-6 h-6 text-accent" />
                        </div>
                        <div>
                            <h2 className="text-xl font-bold text-text-primary dark:text-text-dark-primary italic">How the AI Thinks</h2>
                            <p className="text-[10px] text-accent font-bold uppercase tracking-widest">Prediction Methodology</p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-surface-hover dark:hover:bg-surface-dark-hover rounded-full transition-colors"
                    >
                        <X className="w-5 h-5 text-text-muted" />
                    </button>
                </div>

                {/* Content */}
                <div className="p-8 space-y-8 max-h-[70vh] overflow-y-auto custom-scrollbar">
                    <section className="space-y-3">
                        <div className="flex items-center gap-2 text-sm font-bold text-text-primary dark:text-text-dark-primary uppercase tracking-tight">
                            <ShieldCheck className="w-4 h-4 text-success" />
                            Fairness-Focused Analysis
                        </div>
                        <p className="text-sm text-text-secondary dark:text-text-dark-secondary leading-relaxed">
                            Our AI looks at hundreds of patterns to understand why people leave. We ensure the model is fair by removing protected attributes (like Age or Gender) from the decision-making process, focusing only on professional factors like role stability and career growth.
                        </p>
                    </section>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="p-4 rounded-xl bg-background dark:bg-background-dark border border-border dark:border-border-dark space-y-3">
                            <div className="flex items-center gap-2 text-xs font-bold text-text-primary dark:text-text-dark-primary">
                                <Layers className="w-4 h-4 text-accent" />
                                Key Drivers
                            </div>
                            <ul className="space-y-2">
                                <li className="flex justify-between items-center text-[11px]">
                                    <span className="text-text-secondary">Tenure/Time-in-Role</span>
                                    <Badge variant="success" size="sm">High Impact</Badge>
                                </li>
                                <li className="flex justify-between items-center text-[11px]">
                                    <span className="text-text-secondary">Salary vs. Dept Avg</span>
                                    <Badge variant="warning" size="sm">Medium Impact</Badge>
                                </li>
                                <li className="flex justify-between items-center text-[11px]">
                                    <span className="text-text-secondary">Last Rating Delta</span>
                                    <Badge variant="info" size="sm">High Impact</Badge>
                                </li>
                            </ul>
                        </div>

                        <div className="p-4 rounded-xl bg-background dark:bg-background-dark border border-border dark:border-border-dark space-y-3">
                            <div className="flex items-center gap-2 text-xs font-bold text-text-primary dark:text-text-dark-primary">
                                <TrendingDown className="w-4 h-4 text-danger" />
                                Risk Categories
                            </div>
                            <div className="space-y-2">
                                <div className="flex justify-between items-center text-[11px]">
                                    <span className="text-text-secondary">High Risk</span>
                                    <span className="font-bold text-danger text-xs">&gt; 0.7 score</span>
                                </div>
                                <div className="flex justify-between items-center text-[11px]">
                                    <span className="text-text-secondary">Medium Risk</span>
                                    <span className="font-bold text-warning text-xs">0.4 - 0.7 score</span>
                                </div>
                                <div className="flex justify-between items-center text-[11px]">
                                    <span className="text-text-secondary">Baseline Low</span>
                                    <span className="font-bold text-success text-xs">&lt; 0.4 score</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <section className="p-6 bg-accent/5 rounded-2xl border border-accent/10">
                        <div className="flex items-center gap-2 text-sm font-bold text-accent uppercase tracking-tight mb-4">
                            <Percent className="w-4 h-4" />
                            How to Read the Scores
                        </div>
                        <p className="text-xs text-text-secondary dark:text-text-dark-secondary leading-relaxed">
                            Predictions are displayed as probabilities. A score of 0.75 means there is a 75% chance of the event occurring within the timeframe based on current patterns.
                            <br /><br />
                            <em>*Pro-tip: Higher scores indicate a stronger match with historical departure patterns, suggesting it's a good time for a retention conversation or stay-interview.</em>
                        </p>
                    </section>
                </div>

                {/* Footer */}
                <div className="p-6 bg-surface-hover dark:bg-surface-dark-hover border-t border-border dark:border-border-dark text-center">
                    <p className="text-[10px] text-text-muted dark:text-text-dark-muted font-medium">
                        PeopleOS AI uses Scikit-Learn and XGBoost for deterministic calculations. Insights are advisory only.
                    </p>
                </div>
            </div>
        </div>
    )
}
