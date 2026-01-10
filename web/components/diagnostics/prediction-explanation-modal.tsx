'use client'

import React from 'react'
import { X, Brain, Target, BarChart3, Zap, ShieldCheck } from 'lucide-react'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'

interface PredictionExplanationModalProps {
    isOpen: boolean
    onClose: () => void
}

export function PredictionExplanationModal({
    isOpen,
    onClose,
}: PredictionExplanationModalProps) {
    if (!isOpen) return null

    return (
        <div
            className="fixed inset-0 z-[60] flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
            onClick={onClose}
        >
            <Card
                className="w-full max-w-2xl max-h-[90vh] overflow-auto shadow-2xl animate-in fade-in zoom-in duration-200"
                onClick={(e: React.MouseEvent) => e.stopPropagation()}
            >
                <div className="flex items-start justify-between mb-6">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-accent/10 dark:bg-accent/20 flex items-center justify-center">
                            <Brain className="w-6 h-6 text-accent" />
                        </div>
                        <div>
                            <h2 className="text-xl font-bold text-text-primary dark:text-text-dark-primary">How Predictions Work</h2>
                            <p className="text-sm text-text-secondary dark:text-text-dark-secondary">
                                XGBoost + SHAP Explainable AI
                            </p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 rounded-lg hover:bg-surface-hover dark:hover:bg-surface-dark-hover transition-colors"
                    >
                        <X className="w-5 h-5 text-text-muted dark:text-text-dark-muted" />
                    </button>
                </div>

                <div className="space-y-8 py-2">
                    {/* Step 1: Data Processing */}
                    <div className="flex gap-4">
                        <div className="mt-1 p-2 bg-accent/10 rounded-lg shrink-0 h-fit">
                            <Zap className="w-5 h-5 text-accent" />
                        </div>
                        <div>
                            <h4 className="font-bold text-text-primary dark:text-text-dark-primary mb-1">1. Intelligent Feature Engineering</h4>
                            <p className="text-sm text-text-secondary dark:text-text-dark-secondary leading-relaxed">
                                Raw employee data is transformed into predictive features. We analyze tenure, salary positioning, performance trends, and department dynamics to build a comprehensive risk profile.
                            </p>
                        </div>
                    </div>

                    {/* Step 2: Model Training */}
                    <div className="flex gap-4">
                        <div className="mt-1 p-2 bg-success/10 rounded-lg shrink-0 h-fit">
                            <Target className="w-5 h-5 text-success" />
                        </div>
                        <div>
                            <h4 className="font-bold text-text-primary dark:text-text-dark-primary mb-1">2. Gradient Boosted Intelligence</h4>
                            <p className="text-sm text-text-secondary dark:text-text-dark-secondary leading-relaxed">
                                Our model utilizes XGBoost, an industry-leading gradient boosting algorithm. It learns from historical patterns to identify non-linear relationships that simple statistics might miss.
                            </p>
                        </div>
                    </div>

                    {/* Step 3: SHAP Explanations */}
                    <div className="flex gap-4">
                        <div className="mt-1 p-2 bg-warning/10 rounded-lg shrink-0 h-fit">
                            <BarChart3 className="w-5 h-5 text-warning" />
                        </div>
                        <div>
                            <h4 className="font-bold text-text-primary dark:text-text-dark-primary mb-1">3. Explainable AI (SHAP Values)</h4>
                            <p className="text-sm text-text-secondary dark:text-text-dark-secondary leading-relaxed">
                                We believe AI shouldn't be a black box. SHAP (SHapley Additive exPlanations) values break down exactly how each factor contributed to an individual's specific risk score.
                            </p>
                        </div>
                    </div>

                    {/* Privacy Note */}
                    <div className="p-4 rounded-xl bg-surface-hover dark:bg-surface-dark-hover border border-border/50 dark:border-border-dark/50 flex gap-3">
                        <ShieldCheck className="w-5 h-5 text-text-muted shrink-0" />
                        <div>
                            <div className="text-xs font-bold text-text-muted uppercase tracking-widest mb-1">Privacy & Ethics</div>
                            <p className="text-[11px] text-text-secondary leading-normal italic">
                                All predictions are intended as decision-support tools. PeopleOS does not include protected demographic classes in predictive modeling to ensure unbiased outcomes.
                            </p>
                        </div>
                    </div>
                </div>

                <div className="flex justify-end gap-3 mt-8 pt-6 border-t border-border dark:border-border-dark">
                    <Button onClick={onClose} className="px-8">
                        Got it, thanks!
                    </Button>
                </div>
            </Card>
        </div>
    )
}
