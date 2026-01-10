'use client'

import { useState, useEffect } from 'react'
import { X, ShieldAlert, TrendingUp, TrendingDown, Clock, User, Briefcase, MapPin } from 'lucide-react'
import { cn } from '../lib/utils'
import { Badge } from './ui/badge'

interface RiskFactor {
    factor: string
    impact: 'High' | 'Medium' | 'Low'
    direction: 'Increase Risk' | 'Decrease Risk'
    score: number
    description: string
}

interface Employee {
    EmployeeID: string
    Dept?: string
    JobTitle?: string
    Location?: string
    current_tenure_years?: number
    attrition_risk_12mo?: number
    risk_score?: number // Added for Quality of Hire
    risk_category: 'High' | 'Medium' | 'Low'
    risk_factors?: RiskFactor[]
}

interface RiskAnalysisModalProps {
    isOpen: boolean
    onClose: () => void
    employee: Employee | null
}

export function RiskAnalysisModal({ isOpen, onClose, employee }: RiskAnalysisModalProps) {
    useEffect(() => {
        const handleEscape = (e: KeyboardEvent) => {
            if (e.key === 'Escape') onClose()
        }
        if (isOpen) {
            document.addEventListener('keydown', handleEscape)
            document.body.style.overflow = 'hidden'
        }
        return () => {
            document.removeEventListener('keydown', handleEscape)
            document.body.style.overflow = 'unset'
        }
    }, [isOpen, onClose])

    if (!isOpen || !employee) return null

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            {/* Backdrop */}
            <div
                className="absolute inset-0 bg-black/60 backdrop-blur-sm transition-opacity"
                onClick={onClose}
            />

            {/* Modal */}
            <div className="relative bg-surface dark:bg-surface-dark border border-border dark:border-border-dark rounded-2xl shadow-2xl w-full max-w-2xl max-h-[90vh] flex flex-col overflow-hidden animate-in fade-in zoom-in duration-200">
                {/* Header */}
                <div className="flex items-center justify-between p-6 border-b border-border dark:border-border-dark bg-surface-secondary dark:bg-surface-dark-secondary">
                    <div className="flex items-center gap-4">
                        <div className={cn(
                            "p-3 rounded-xl",
                            employee.risk_category === 'High' ? 'bg-danger/10 text-danger' :
                                employee.risk_category === 'Medium' ? 'bg-warning/10 text-warning' : 'bg-success/10 text-success'
                        )}>
                            <ShieldAlert className="w-6 h-6" />
                        </div>
                        <div>
                            <h2 className="text-xl font-bold text-text-primary dark:text-text-dark-primary">
                                Risk Analysis: {employee.EmployeeID}
                            </h2>
                            <div className="flex items-center gap-3 mt-1 text-sm text-text-secondary">
                                <span className="flex items-center gap-1">
                                    <Briefcase className="w-3.5 h-3.5" />
                                    {employee.Dept || 'N/A'}
                                </span>
                                {employee.current_tenure_years !== undefined && (
                                    <span className="flex items-center gap-1">
                                        <Clock className="w-3.5 h-3.5" />
                                        {employee.current_tenure_years.toFixed(1)} years tenure
                                    </span>
                                )}
                            </div>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-surface-hover dark:hover:bg-surface-dark-hover rounded-full transition-colors"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* Content */}
                <div className="flex-1 overflow-y-auto p-6 space-y-6">
                    {/* Summary Section */}
                    <div className="grid grid-cols-2 gap-4">
                        <div className="p-4 rounded-xl bg-surface-secondary dark:bg-surface-dark-secondary border border-border/50 dark:border-border-dark/50">
                            <p className="text-xs text-text-muted uppercase tracking-wider mb-1">Departure Probability</p>
                            <div className="flex items-baseline gap-2">
                                <span className="text-3xl font-bold text-text-primary dark:text-text-dark-primary">
                                    {employee.attrition_risk_12mo !== undefined
                                        ? (employee.attrition_risk_12mo * 100).toFixed(0)
                                        : (employee.risk_score ? (employee.risk_score).toFixed(0) : 0)}%
                                </span>
                                <span className="text-sm text-text-muted">
                                    {employee.attrition_risk_12mo !== undefined ? 'over 12 months' : 'onboarding risk'}
                                </span>
                            </div>
                        </div>
                        <div className="p-4 rounded-xl bg-surface-secondary dark:bg-surface-dark-secondary border border-border/50 dark:border-border-dark/50">
                            <p className="text-xs text-text-muted uppercase tracking-wider mb-1">Risk Category</p>
                            <div className="pt-1">
                                <Badge variant={
                                    employee.risk_category === 'High' ? 'danger' :
                                        employee.risk_category === 'Medium' ? 'warning' : 'success'
                                } className="px-3 py-1 text-sm">
                                    {employee.risk_category} Priority
                                </Badge>
                            </div>
                        </div>
                    </div>

                    {/* Risk Factors Breakdown */}
                    <div>
                        <h3 className="text-sm font-semibold text-text-muted uppercase tracking-wider mb-4">Detailed Risk Drivers</h3>
                        <div className="space-y-3">
                            {employee.risk_factors && employee.risk_factors.length > 0 ? (
                                employee.risk_factors.map((factor, idx) => (
                                    <div
                                        key={idx}
                                        className="p-4 rounded-xl border border-border dark:border-border-dark bg-surface dark:bg-background-dark hover:border-accent/30 transition-colors"
                                    >
                                        <div className="flex items-start justify-between mb-2">
                                            <div className="flex items-center gap-2">
                                                {factor.direction === 'Increase Risk' ? (
                                                    <TrendingUp className="w-4 h-4 text-danger" />
                                                ) : (
                                                    <TrendingDown className="w-4 h-4 text-success" />
                                                )}
                                                <span className="font-bold text-text-primary dark:text-text-dark-primary capitalize">
                                                    {factor.factor.replace(/([A-Z])/g, ' $1').trim()}
                                                </span>
                                            </div>
                                            <Badge variant={
                                                factor.impact === 'High' ? 'danger' :
                                                    factor.impact === 'Medium' ? 'warning' : 'outline'
                                            } className="text-[10px] uppercase">
                                                {factor.impact} Impact
                                            </Badge>
                                        </div>
                                        <p className="text-sm text-text-secondary dark:text-text-dark-secondary leading-relaxed mt-1">
                                            {factor.description}
                                        </p>
                                    </div>
                                ))
                            ) : (
                                <div className="py-8 text-center text-text-muted border border-dashed border-border rounded-xl">
                                    No significant risk drivers identified
                                </div>
                            )}
                        </div>
                    </div>

                    {/* HR Recommendations */}
                    <div className="p-4 rounded-xl bg-accent/5 border border-accent/20">
                        <h4 className="text-sm font-bold text-accent mb-2">HR Strategic Action</h4>
                        <p className="text-sm text-text-secondary dark:text-text-dark-secondary italic">
                            {employee.risk_category === 'High'
                                ? "IMMEDIATE: Recommend a stay interview within 48 hours and a comprehensive review of compensation and career trajectory."
                                : employee.risk_category === 'Medium'
                                    ? "PROACTIVE: Include in next monthly check-in. Explore career development opportunities or role expansion."
                                    : "MONITOR: Continue standard engagement cycle. No immediate action required."}
                        </p>
                    </div>
                </div>

                {/* Footer */}
                <div className="p-4 border-t border-border dark:border-border-dark bg-surface-secondary dark:bg-surface-dark-secondary flex justify-end">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 bg-accent text-white rounded-lg font-medium hover:bg-accent/90 transition-colors"
                    >
                        Close Analysis
                    </button>
                </div>
            </div>
        </div>
    )
}
