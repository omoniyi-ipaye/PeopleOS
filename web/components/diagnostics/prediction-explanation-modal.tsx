'use client'
import { PredictiveMathModal } from '@/components/dashboard/predictive-math-modal'

export function PredictionExplanationModal({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
    return isOpen ? <PredictiveMathModal onClose={onClose} /> : null
}
