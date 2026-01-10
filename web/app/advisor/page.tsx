'use client'

import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { api } from '@/lib/api-client'
import { Brain, Sparkles, Send, AlertTriangle, Lightbulb, Target } from 'lucide-react'

import {
  AdvisorStatus,
  AdvisorSummary,
  AdvisorAskResponse,
} from '@/types/api'

export default function AdvisorPage() {
  const [question, setQuestion] = useState('')
  const [isRequested, setIsRequested] = useState(false)

  const { data: advisorStatus } = useQuery<AdvisorStatus>({
    queryKey: ['advisor', 'status'],
    queryFn: api.advisor.getStatus as any,
  })

  const { data: summary, isLoading: summaryLoading, refetch: generateSummary } = useQuery<AdvisorSummary>({
    queryKey: ['advisor', 'summary'],
    queryFn: api.advisor.getSummary as any,
    enabled: isRequested && (advisorStatus?.available ?? false),
    retry: 1,
  })

  const handleGenerate = () => {
    setIsRequested(true)
    generateSummary()
  }

  const askMutation = useMutation<AdvisorAskResponse, Error, string>({
    mutationFn: (q: string) => api.advisor.ask(q) as any,
  })

  const handleAsk = (e: React.FormEvent) => {
    e.preventDefault()
    if (question.trim()) {
      askMutation.mutate(question)
    }
  }

  if (!advisorStatus?.available) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-text-primary dark:text-text-dark-primary">Strategic Advisor</h1>
          <p className="text-text-secondary dark:text-text-dark-secondary mt-1">
            AI-powered insights and recommendations
          </p>
        </div>

        <div className="flex flex-col items-center justify-center h-64 gap-4 text-center">
          <AlertTriangle className="w-12 h-12 text-warning" />
          <h2 className="text-xl font-semibold text-text-primary dark:text-text-dark-primary">AI Advisor Unavailable</h2>
          <p className="text-text-secondary dark:text-text-dark-secondary max-w-md">
            {advisorStatus?.reason || 'Ensure Ollama is running locally to enable AI features.'}
          </p>
          <Badge variant="warning">
            Requires Ollama LLM
          </Badge>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Page Title */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-text-primary dark:text-text-dark-primary">Strategic Advisor</h1>
          <p className="text-text-secondary dark:text-text-dark-secondary mt-1">
            AI-powered insights powered by {advisorStatus.model}
          </p>
        </div>
        <Badge variant="success">
          <Sparkles className="w-3 h-3 mr-1" />
          AI Active
        </Badge>
      </div>

      {/* Strategic Summary */}
      <Card title="Executive Summary" className="relative overflow-hidden min-h-[400px] flex flex-col justify-center">
        {summaryLoading ? (
          <div className="flex flex-col items-center justify-center py-12 text-center animate-in fade-in zoom-in duration-500">
            <div className="relative mb-6">
              <div className="absolute inset-0 bg-accent/20 rounded-full blur-2xl animate-pulse" />
              <div className="relative bg-surface dark:bg-surface-dark p-6 rounded-full border border-accent/30 shadow-2xl">
                <Brain className="w-12 h-12 text-accent animate-bounce" />
              </div>
              <div className="absolute -top-1 -right-1">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-accent rounded-full animate-bounce [animation-delay:-0.3s]" />
                  <div className="w-2 h-2 bg-accent rounded-full animate-bounce [animation-delay:-0.15s]" />
                  <div className="w-2 h-2 bg-accent rounded-full animate-bounce" />
                </div>
              </div>
            </div>
            <h3 className="text-xl font-semibold text-text-primary dark:text-text-dark-primary mb-2">
              Analyzing Workforce Intelligence
            </h3>
            <p className="text-text-secondary dark:text-text-dark-secondary max-w-sm">
              Our Strategic Engine is synthesizing your data to generate a high-level executive briefing.
            </p>
          </div>
        ) : summary ? (
          <div className="space-y-6">
            {/* Summary */}
            <div>
              <div className="flex items-center gap-2 mb-2">
                <Brain className="w-5 h-5 text-accent" />
                <h3 className="font-medium text-text-primary dark:text-text-dark-primary">Analysis Summary</h3>
              </div>
              <p className="text-text-secondary dark:text-text-dark-secondary leading-relaxed">
                {summary.summary}
              </p>
            </div>

            {/* Key Insights */}
            {summary.key_insights && summary.key_insights.length > 0 && (
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <Lightbulb className="w-5 h-5 text-warning" />
                  <h3 className="font-medium text-text-primary dark:text-text-dark-primary">Key Insights</h3>
                </div>
                <ul className="space-y-2">
                  {summary.key_insights.map((insight: string, i: number) => (
                    <li
                      key={i}
                      className="flex items-start gap-2 text-text-secondary dark:text-text-dark-secondary"
                    >
                      <span className="w-1.5 h-1.5 rounded-full bg-warning mt-2 flex-shrink-0" />
                      {insight}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Action Items */}
            {summary.action_items && summary.action_items.length > 0 && (
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <Target className="w-5 h-5 text-success" />
                  <h3 className="font-medium text-text-primary dark:text-text-dark-primary">Recommended Actions</h3>
                </div>
                <ul className="space-y-2">
                  {summary.action_items.map((action: string, i: number) => (
                    <li
                      key={i}
                      className="flex items-start gap-2 text-text-secondary dark:text-text-dark-secondary"
                    >
                      <span className="w-1.5 h-1.5 rounded-full bg-success mt-2 flex-shrink-0" />
                      {action}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <div className="text-xs text-text-muted dark:text-text-dark-muted pt-4 border-t border-border dark:border-border-dark">
              Generated by {summary.generated_by}
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <div className="bg-surface dark:bg-surface-dark p-6 rounded-full border border-border dark:border-border-dark mb-6 shadow-sm">
              <Sparkles className="w-10 h-10 text-accent/40" />
            </div>
            <h2 className="text-xl font-semibold text-text-primary dark:text-text-dark-primary mb-2">Ready for Analysis</h2>
            <p className="text-text-secondary dark:text-text-dark-secondary max-w-md mb-8">
              Generate a comprehensive AI executive summary based on your currently loaded workforce data.
            </p>
            <Button size="lg" onClick={handleGenerate} className="px-8 shadow-lg shadow-accent/20">
              <Brain className="w-5 h-5 mr-3" />
              Generate Executive Briefing
            </Button>
          </div>
        )}
      </Card>

      {/* Ask a Question */}
      <Card title="Ask a Question" subtitle="Get AI-powered answers about your data">
        <form onSubmit={handleAsk} className="space-y-4">
          <div className="relative">
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask a question about your workforce data..."
              rows={3}
              className="w-full px-4 py-3 bg-surface dark:bg-surface-dark border border-border dark:border-border-dark rounded-lg text-text-primary dark:text-text-dark-primary placeholder-text-muted dark:placeholder-text-dark-muted focus:outline-none focus:ring-2 focus:ring-accent/50 resize-none"
            />
          </div>
          <div className="flex justify-end">
            <Button
              type="submit"
              disabled={!question.trim()}
              isLoading={askMutation.isPending}
            >
              <Send className="w-4 h-4 mr-2" />
              Ask Advisor
            </Button>
          </div>
        </form>

        {/* Answer */}
        {askMutation.data && (
          <div className="mt-6 p-4 rounded-lg bg-accent/5 border border-accent/20">
            <div className="flex items-center gap-2 mb-2">
              <Brain className="w-4 h-4 text-accent" />
              <span className="text-sm font-medium text-accent">AI Response</span>
            </div>
            <p className="text-text-secondary dark:text-text-dark-secondary leading-relaxed">
              {askMutation.data.answer}
            </p>
          </div>
        )}

        {/* Example Questions */}
        <div className="mt-6 pt-6 border-t border-border dark:border-border-dark">
          <h4 className="text-sm font-medium text-text-muted dark:text-text-dark-muted mb-3">
            Example Questions
          </h4>
          <div className="flex flex-wrap gap-2">
            {[
              'What departments have the highest attrition risk?',
              'How can we improve retention?',
              'What are our biggest HR challenges?',
              'Which employees should we prioritize for retention?',
            ].map((q) => (
              <button
                key={q}
                onClick={() => setQuestion(q)}
                className="px-3 py-1.5 text-sm bg-surface hover:bg-accent/5 dark:bg-surface-dark-hover border border-border dark:border-border-dark hover:border-accent/30 rounded-full text-text-secondary dark:text-text-dark-secondary hover:text-text-primary dark:hover:text-text-dark-primary transition-colors"
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      </Card>
    </div>
  )
}
