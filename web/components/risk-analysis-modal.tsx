'use client'

import { useEffect } from 'react'

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

    return <div role="dialog" aria-modal="true" aria-label="Individual predictive view unavailable" className="fixed inset-0 z-50 grid place-items-center bg-black/60 p-4">
        <div className="rounded-2xl bg-surface p-6 max-w-lg space-y-4">
            <h2 className="font-semibold">Individual predictive view unavailable</h2>
            <p>No validated individual departure probability or intervention recommendation is available. Use aggregate evidence and its evaluation limits.</p>
            <button onClick={onClose} className="text-accent">Close</button>
        </div>
    </div>
}
