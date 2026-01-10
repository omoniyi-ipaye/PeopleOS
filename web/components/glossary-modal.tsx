'use client'

import { useState, useEffect } from 'react'
import { X, Search, BookOpen } from 'lucide-react'
import { cn } from '@/lib/utils'

interface GlossaryTerm {
  term: string
  definition: string
  category: 'retention' | 'metrics' | 'model' | 'compensation' | 'performance'
  example?: string
}

const GLOSSARY_TERMS: GlossaryTerm[] = [
  // Retention & Risk
  {
    term: 'Flight Risk',
    definition: 'The probability that an employee will voluntarily leave the organization within the next 12 months.',
    category: 'retention',
    example: 'A flight risk of 75% means 3 in 4 chance of leaving within a year.'
  },
  {
    term: 'Retention Rate',
    definition: 'The percentage of employees who stayed with the company over a specific period.',
    category: 'retention',
    example: 'A 90% retention rate means 9 out of 10 employees stayed.'
  },
  {
    term: 'Turnover Rate',
    definition: 'The percentage of employees who left the company over a specific period, including voluntary and involuntary departures.',
    category: 'retention',
    example: 'A 15% annual turnover means 15 out of 100 employees left that year.'
  },
  {
    term: 'Watch List',
    definition: 'Employees identified as having elevated risk of departure who need attention or intervention.',
    category: 'retention'
  },
  {
    term: 'Retention Forecast',
    definition: 'A prediction of how long employees are likely to stay with the company based on various factors.',
    category: 'retention'
  },

  // Model Metrics
  {
    term: 'Risk Multiplier',
    definition: 'Shows how much a factor increases or decreases the likelihood of an employee leaving. Values above 1.0 increase risk, below 1.0 decrease risk.',
    category: 'model',
    example: 'A risk multiplier of 1.35 means 35% more likely to leave; 0.72 means 28% less likely.'
  },
  {
    term: 'Model Accuracy',
    definition: 'How well the prediction model correctly identifies employees who will leave vs those who will stay.',
    category: 'model',
    example: 'A model accuracy of 0.72 means predictions are correct 72% of the time.'
  },
  {
    term: 'Detection Rate',
    definition: 'The percentage of actual departures that the model correctly predicted in advance.',
    category: 'model',
    example: 'A detection rate of 80% means we caught 8 out of 10 departures before they happened.'
  },
  {
    term: 'Prediction Accuracy',
    definition: 'When the model flags someone as at-risk, how often that prediction is correct.',
    category: 'model',
    example: 'A prediction accuracy of 70% means 7 out of 10 flagged employees actually left.'
  },
  {
    term: 'Relationship Strength',
    definition: 'Measures how closely two factors are related. Higher values mean stronger relationships.',
    category: 'model',
    example: 'A correlation of 0.8 between tenure and salary shows they strongly move together.'
  },

  // Compensation
  {
    term: 'Compa Ratio',
    definition: 'An employee\'s salary compared to the market midpoint. 1.0 means at market rate.',
    category: 'compensation',
    example: 'A compa ratio of 0.85 means the employee earns 85% of the market rate.'
  },
  {
    term: 'Pay Equity',
    definition: 'The degree to which employees in similar roles receive similar pay regardless of demographic factors.',
    category: 'compensation'
  },
  {
    term: 'Gender Pay Gap',
    definition: 'The difference in average earnings between men and women, expressed as a percentage.',
    category: 'compensation',
    example: 'A 5% pay gap means women earn 5% less than men on average.'
  },

  // Performance & Potential
  {
    term: 'High Performer',
    definition: 'An employee who consistently exceeds expectations in their current role.',
    category: 'performance'
  },
  {
    term: 'High Potential (HiPo)',
    definition: 'An employee identified as having the ability and aspiration to advance to more senior roles.',
    category: 'performance'
  },
  {
    term: '9-Box Grid',
    definition: 'A talent assessment tool that plots employees on two axes: current performance and future potential.',
    category: 'performance',
    example: 'Top-right box = high performance + high potential = star talent.'
  },
  {
    term: 'Stagnation Index',
    definition: 'How long an employee has been in their current role compared to typical tenure, indicating potential flight risk or disengagement.',
    category: 'performance',
    example: 'A stagnation index of 1.5 means 50% longer than average in the same role.'
  },
  {
    term: 'Quality of Hire',
    definition: 'A composite measure of how well new hires perform, their retention, and their overall contribution.',
    category: 'performance'
  },

  // Additional metrics
  {
    term: 'eNPS (Employee Net Promoter Score)',
    definition: 'A measure of employee satisfaction based on how likely they are to recommend the company as a workplace. Ranges from -100 to +100.',
    category: 'metrics',
    example: 'An eNPS of +30 is good; +50 is excellent; negative scores indicate problems.'
  },
  {
    term: 'Span of Control',
    definition: 'The number of direct reports a manager has.',
    category: 'metrics',
    example: 'A span of control of 8 means the manager has 8 direct reports.'
  },
  {
    term: 'Bench Strength',
    definition: 'The readiness of internal candidates to fill key leadership positions.',
    category: 'performance'
  },
  {
    term: 'Succession Readiness',
    definition: 'How prepared the organization is to fill critical roles if current leaders leave.',
    category: 'performance'
  }
]

const categoryLabels: Record<GlossaryTerm['category'], string> = {
  retention: 'Retention & Risk',
  metrics: 'Key Metrics',
  model: 'Model & Predictions',
  compensation: 'Compensation',
  performance: 'Performance & Talent'
}

const categoryColors: Record<GlossaryTerm['category'], string> = {
  retention: 'bg-danger/10 text-danger border-danger/30',
  metrics: 'bg-accent/10 text-accent border-accent/30',
  model: 'bg-purple-500/10 text-purple-500 border-purple-500/30',
  compensation: 'bg-success/10 text-success border-success/30',
  performance: 'bg-warning/10 text-warning border-warning/30'
}

interface GlossaryModalProps {
  isOpen: boolean
  onClose: () => void
  initialSearch?: string
}

export function GlossaryModal({ isOpen, onClose, initialSearch = '' }: GlossaryModalProps) {
  const [searchTerm, setSearchTerm] = useState(initialSearch)
  const [selectedCategory, setSelectedCategory] = useState<GlossaryTerm['category'] | 'all'>('all')

  useEffect(() => {
    if (isOpen) {
      setSearchTerm(initialSearch)
    }
  }, [isOpen, initialSearch])

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

  const filteredTerms = GLOSSARY_TERMS.filter(term => {
    const matchesSearch = searchTerm === '' ||
      term.term.toLowerCase().includes(searchTerm.toLowerCase()) ||
      term.definition.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesCategory = selectedCategory === 'all' || term.category === selectedCategory
    return matchesSearch && matchesCategory
  })

  const categories = ['all', 'retention', 'metrics', 'model', 'compensation', 'performance'] as const

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative bg-surface dark:bg-surface-dark border border-border dark:border-border-dark rounded-xl shadow-2xl w-full max-w-2xl max-h-[80vh] flex flex-col mx-4">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border dark:border-border-dark">
          <div className="flex items-center gap-2">
            <BookOpen className="w-5 h-5 text-accent" />
            <h2 className="text-lg font-bold">HR Glossary</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 hover:bg-surface-hover dark:hover:bg-surface-dark-hover rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Search & Filters */}
        <div className="p-4 border-b border-border dark:border-border-dark space-y-3">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
            <input
              type="text"
              placeholder="Search terms..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-9 pr-4 py-2 bg-surface-secondary dark:bg-surface-dark-secondary border border-border dark:border-border-dark rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-accent/50"
              autoFocus
            />
          </div>

          <div className="flex flex-wrap gap-2">
            {categories.map((cat) => (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                className={cn(
                  "px-3 py-1 rounded-full text-xs font-medium transition-colors",
                  selectedCategory === cat
                    ? 'bg-accent text-white'
                    : 'bg-surface-secondary dark:bg-surface-dark-secondary text-text-secondary hover:bg-surface-hover dark:hover:bg-surface-dark-hover'
                )}
              >
                {cat === 'all' ? 'All' : categoryLabels[cat]}
              </button>
            ))}
          </div>
        </div>

        {/* Terms List */}
        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {filteredTerms.length === 0 ? (
            <div className="text-center py-8 text-text-muted">
              No terms found matching "{searchTerm}"
            </div>
          ) : (
            filteredTerms.map((item) => (
              <div
                key={item.term}
                className="p-4 bg-surface-secondary dark:bg-surface-dark-secondary rounded-lg"
              >
                <div className="flex items-start justify-between mb-2">
                  <h3 className="font-bold text-text-primary dark:text-text-dark-primary">
                    {item.term}
                  </h3>
                  <span className={cn(
                    "px-2 py-0.5 rounded-full text-[10px] font-medium border",
                    categoryColors[item.category]
                  )}>
                    {categoryLabels[item.category]}
                  </span>
                </div>
                <p className="text-sm text-text-secondary dark:text-text-dark-secondary">
                  {item.definition}
                </p>
                {item.example && (
                  <p className="text-xs text-text-muted dark:text-text-dark-muted mt-2 pt-2 border-t border-border/50 dark:border-border-dark/50">
                    <span className="font-medium">Example:</span> {item.example}
                  </p>
                )}
              </div>
            ))
          )}
        </div>

        {/* Footer */}
        <div className="p-3 border-t border-border dark:border-border-dark text-center text-xs text-text-muted">
          {filteredTerms.length} term{filteredTerms.length !== 1 ? 's' : ''} • Press ESC to close
        </div>
      </div>
    </div>
  )
}

// Floating glossary button for pages
export function GlossaryButton() {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 z-40 flex items-center gap-2 px-4 py-2 bg-accent text-white rounded-full shadow-lg hover:bg-accent/90 transition-colors"
        title="Open HR Glossary"
      >
        <BookOpen className="w-4 h-4" />
        <span className="text-sm font-medium">Glossary</span>
      </button>
      <GlossaryModal isOpen={isOpen} onClose={() => setIsOpen(false)} />
    </>
  )
}
