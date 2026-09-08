'use client'

interface EmployeeDetailModalProps { employeeId: string; onClose: () => void }

export function EmployeeDetailModal({ onClose }: EmployeeDetailModalProps) {
  return <div role="dialog" aria-modal="true" aria-label="Individual predictive view unavailable" className="fixed inset-0 z-50 grid place-items-center bg-black/60 p-4">
    <div className="max-w-lg space-y-4 rounded-2xl bg-surface p-6">
      <h2 className="font-semibold">Individual predictive view unavailable</h2>
      <p>Use aggregate Retention Signals and the model evaluation evidence. No validated individual departure probability is available.</p>
      <button onClick={onClose} className="text-accent">Close</button>
    </div>
  </div>
}
