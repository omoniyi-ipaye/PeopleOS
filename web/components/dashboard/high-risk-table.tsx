'use client'

import { RiskBadge } from '@/components/ui/badge'
import { formatCurrency } from '@/lib/utils'

export interface HighRiskEmployeeRow {
  employee_id: string
  dept: string
  tenure: number
  salary: number
  last_rating: number
  risk_score: number
  risk_category: string
}

interface HighRiskTableProps {
  employees: HighRiskEmployeeRow[]
  onEmployeeClick?: (employeeId: string) => void
}

export function HighRiskTable({ employees, onEmployeeClick }: HighRiskTableProps) {
  if (!employees || employees.length === 0) {
    return (
      <div className="text-center text-text-secondary dark:text-text-dark-secondary py-8">
        No high-risk employees found
      </div>
    )
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead>
          <tr className="border-b border-border dark:border-border-dark">
            <th className="text-left py-2 px-2 text-xs font-medium text-text-muted dark:text-text-dark-muted uppercase tracking-wider">
              Employee
            </th>
            <th className="text-left py-2 px-2 text-xs font-medium text-text-muted dark:text-text-dark-muted uppercase tracking-wider">
              Dept
            </th>
            <th className="text-right py-2 px-2 text-xs font-medium text-text-muted dark:text-text-dark-muted uppercase tracking-wider">
              Risk
            </th>
            <th className="text-right py-2 px-2 text-xs font-medium text-text-muted dark:text-text-dark-muted uppercase tracking-wider">
              Score
            </th>
          </tr>
        </thead>
        <tbody>
          {employees.slice(0, 10).map((emp) => (
            <tr
              key={emp.employee_id}
              className="border-b border-border/50 dark:border-border-dark/50 hover:bg-surface-hover dark:hover:bg-surface-dark-hover transition-colors cursor-pointer"
              onClick={() => onEmployeeClick?.(emp.employee_id)}
            >
              <td className="py-2 px-2">
                <div>
                  <div className="font-medium text-sm text-text-primary dark:text-text-dark-primary">{emp.employee_id}</div>
                  <div className="text-xs text-text-muted dark:text-text-dark-muted">
                    {emp.tenure.toFixed(1)} yrs
                  </div>
                </div>
              </td>
              <td className="py-2 px-2 text-sm text-text-secondary dark:text-text-dark-secondary">
                {emp.dept}
              </td>
              <td className="py-2 px-2 text-right">
                <RiskBadge category={emp.risk_category} />
              </td>
              <td className="py-2 px-2 text-right text-sm font-mono text-text-primary dark:text-text-dark-primary">
                {(emp.risk_score * 100).toFixed(0)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
