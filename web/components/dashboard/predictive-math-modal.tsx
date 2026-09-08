'use client'
import { X } from 'lucide-react'

export function PredictiveMathModal({ onClose }: { onClose: () => void }) {
    return <div role="dialog" aria-modal="true" aria-labelledby="prediction-method-title" className="fixed inset-0 z-[100] grid place-items-center bg-black/60 p-4">
        <div className="max-w-xl rounded-2xl bg-surface p-6 space-y-4">
            <div className="flex justify-between gap-4"><h2 id="prediction-method-title" className="text-xl font-semibold">How to interpret model evidence</h2><button aria-label="Close model explanation" onClick={onClose}><X /></button></div>
            <p>The classifier is evaluated on recorded attrition outcomes. A score of 0.75 does not establish a 75% chance of leaving within a future time period.</p>
            <p>Feature importance describes the fitted model. SHAP contributions depend on the model output units and require a matching baseline and prediction.</p>
            <p>Removing protected attributes does not establish fairness: other features can remain proxies. Review measured subgroup results and local validation.</p>
            <p>Use the active model’s recorded evaluation, feature list and configured thresholds. No universal feature ranking or accuracy guarantee applies.</p>
        </div>
    </div>
}
