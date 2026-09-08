'use client'

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

export function HighRiskTable(props: HighRiskTableProps) {
  void props
  return <p role="status">Employee risk ranking is unavailable. Use aggregate Retention Signals and its evaluation limits.</p>
}
