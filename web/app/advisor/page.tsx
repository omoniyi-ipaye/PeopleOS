'use client'

import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { GlassCard } from '@/components/ui/glass-card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { api } from '@/lib/api-client'
import { Brain, Sparkles, Send, AlertTriangle, Lightbulb, Target, MessageSquare, FlaskConical } from 'lucide-react'

import {
  AdvisorStatus,
  AdvisorSummary,
  AdvisorAskResponse,
} from '@/types/api'
import { ModelLab } from '@/components/advisor/model-lab'

export default function AdvisorPage() {
  const [question, setQuestion] = useState('')
  const [isRequested, setIsRequested] = useState(false)
  const [activeTab, setActiveTab] = useState<'briefing' | 'lab'>('briefing')

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
      <div className="space-y-6 h-[calc(100vh-100px)] flex flex-col justify-center items-center">
        <GlassCard className="max-w-md w-full text-center p-8 flex flex-col items-center gap-6">
          <div className="w-20 h-20 rounded-full bg-warning/10 flex items-center justify-center">
            <AlertTriangle className="w-10 h-10 text-warning" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-text-primary dark:text-text-dark-primary">AI Advisor Unavailable</h2>
            <p className="text-text-secondary dark:text-text-dark-secondary mt-2">
              {advisorStatus?.reason || 'Ensure Ollama is running locally to enable AI features.'}
            </p>
          </div>
          <Badge variant="warning" className="px-4 py-1.5 text-sm">
            Requires Ollama LLM
          </Badge>
        </GlassCard>
      </div>
    )
  }

  return (
    <div className="space-y-6 h-[calc(100vh-100px)] flex flex-col animate-in fade-in duration-700 slide-in-from-bottom-4">
      {/* Header with Tabs */}
      <div className="flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between flex-shrink-0">
        <div>
          <h1 className="text-4xl font-display font-bold text-gradient bg-clip-text text-transparent bg-gradient-to-r from-gray-900 to-gray-600 dark:from-white dark:to-gray-400">
            Strategic Advisor
          </h1>
          <p className="text-text-secondary dark:text-text-dark-secondary mt-2 text-lg font-light flex items-center gap-2">
            AI-powered insights powered by <span className="font-medium text-accent">{advisorStatus.model}</span>
          </p>
        </div>

        {/* Premium Tab Navigation */}
        <div className="glass p-1.5 rounded-2xl flex gap-1">
          <button
            onClick={() => setActiveTab('briefing')}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-medium transition-all duration-300 ${activeTab === 'briefing'
              ? 'bg-white dark:bg-slate-800 shadow-lg text-text-primary dark:text-white scale-105'
              : 'text-text-secondary dark:text-slate-400 hover:text-text-primary dark:hover:text-white hover:bg-white/10'
              }`}
          >
            <Sparkles className="w-4 h-4" />
            Executive Briefing
          </button>
          <button
            onClick={() => setActiveTab('lab')}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-medium transition-all duration-300 ${activeTab === 'lab'
              ? 'bg-white dark:bg-slate-800 shadow-lg text-text-primary dark:text-white scale-105'
              : 'text-text-secondary dark:text-slate-400 hover:text-text-primary dark:hover:text-white hover:bg-white/10'
              }`}
          >
            <FlaskConical className="w-4 h-4" />
            Model Lab
          </button>
        </div>
      </div>

      {/* Tab Content */}
      <div className="flex-1 min-h-0">
        {activeTab === 'briefing' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-full">
            {/* Left Column: Summary (7 cols) */}
            <div className="lg:col-span-7 h-full flex flex-col">
              <GlassCard className="h-full flex flex-col relative overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-br from-accent/5 to-transparent pointer-events-none" />

                {summaryLoading ? (
                  <div className="flex flex-col items-center justify-center h-full text-center animate-in fade-in zoom-in duration-500">
                    <div className="relative mb-8">
                      <div className="absolute inset-0 bg-accent/20 rounded-full blur-3xl animate-pulse" />
                      <div className="relative bg-white/50 dark:bg-white/5 p-8 rounded-full border border-accent/30 shadow-2xl backdrop-blur-sm">
                        <Brain className="w-16 h-16 text-accent animate-bounce" />
                      </div>
                    </div>
                    <h3 className="text-2xl font-bold text-text-primary dark:text-text-dark-primary mb-3">
                      Analyzing Workforce...
                    </h3>
                    <p className="text-text-secondary dark:text-text-dark-secondary max-w-sm text-lg">
                      Synthesizing data points to generate your executive briefing.
                    </p>
                  </div>
                ) : summary ? (
                  <div className="flex-1 overflow-y-auto pr-2 space-y-8 p-2">
                    {/* Summary Header */}
                    <div>
                      <h3 className="text-xl font-bold flex items-center gap-3 text-text-primary dark:text-white mb-4">
                        <div className="p-2 bg-accent/10 rounded-lg"><Brain className="w-6 h-6 text-accent" /></div>
                        Executive Summary
                      </h3>
                      <p className="text-lg text-text-secondary dark:text-slate-300 leading-relaxed font-light">
                        {summary.summary}
                      </p>
                    </div>

                    <div className="grid grid-cols-1 gap-6">
                      {/* Key Insights */}
                      {summary.key_insights && summary.key_insights.length > 0 && (
                        <div className="bg-warning/5 dark:bg-warning/5 rounded-2xl p-6 border border-warning/10">
                          <h4 className="font-semibold text-warning mb-4 flex items-center gap-2 text-lg">
                            <Lightbulb className="w-5 h-5" />
                            Key Insights
                          </h4>
                          <ul className="space-y-3">
                            {summary.key_insights.map((insight: string, i: number) => (
                              <li key={i} className="flex items-start gap-3 text-text-secondary dark:text-slate-300">
                                <span className="w-1.5 h-1.5 rounded-full bg-warning mt-2.5 flex-shrink-0 shadow-[0_0_8px_rgba(234,179,8,0.5)]" />
                                {insight}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {/* Action Items */}
                      {summary.action_items && summary.action_items.length > 0 && (
                        <div className="bg-success/5 dark:bg-success/5 rounded-2xl p-6 border border-success/10">
                          <h4 className="font-semibold text-success mb-4 flex items-center gap-2 text-lg">
                            <Target className="w-5 h-5" />
                            Recommended Actions
                          </h4>
                          <ul className="space-y-3">
                            {summary.action_items.map((action: string, i: number) => (
                              <li key={i} className="flex items-start gap-3 text-text-secondary dark:text-slate-300">
                                <span className="w-1.5 h-1.5 rounded-full bg-success mt-2.5 flex-shrink-0 shadow-[0_0_8px_rgba(34,197,94,0.5)]" />
                                {action}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>

                    <div className="text-xs text-text-muted text-center pt-4 opacity-50">
                      Generated by {summary.generated_by}
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center h-full text-center">
                    <div className="w-24 h-24 bg-accent/5 rounded-full flex items-center justify-center mb-6 border border-accent/10">
                      <Sparkles className="w-12 h-12 text-accent" />
                    </div>
                    <h2 className="text-3xl font-bold text-text-primary dark:text-text-dark-primary mb-3">Analysis Ready</h2>
                    <p className="text-text-secondary dark:text-text-dark-secondary max-w-md mb-8 text-lg">
                      Generate a comprehensive AI executive summary based on your currently loaded workforce data.
                    </p>
                    <Button size="lg" onClick={handleGenerate} className="px-8 py-6 text-lg rounded-xl shadow-xl shadow-accent/20 hover:scale-105 transition-transform bg-accent text-white">
                      <Brain className="w-6 h-6 mr-3" />
                      Generate Briefing
                    </Button>
                  </div>
                )}
              </GlassCard>
            </div>

            {/* Right Column: Chat (5 cols) */}
            <div className="lg:col-span-5 h-full flex flex-col">
              <GlassCard className="h-full flex flex-col">
                <div className="mb-6 pb-6 border-b border-white/10">
                  <h3 className="text-xl font-bold flex items-center gap-2 mb-1">
                    <MessageSquare className="w-5 h-5 text-accent" />
                    Strategic Chat
                  </h3>
                  <p className="text-sm text-text-muted">Ask follow-up questions about your data</p>
                </div>

                <div className="flex-1 flex flex-col overflow-hidden">
                  {/* Chat Display Area - Could be expanded to show history */}
                  <div className="flex-1 overflow-y-auto min-h-0 space-y-4 mb-4 pr-1">
                    {askMutation.data ? (
                      <div className="animate-in fade-in slide-in-from-bottom-2 duration-500">
                        <div className="flex gap-3">
                          <div className="w-8 h-8 rounded-full bg-accent flex-shrink-0 flex items-center justify-center text-white">
                            <Brain className="w-4 h-4" />
                          </div>
                          <div className="bg-accent/5 rounded-2xl rounded-tl-none p-4 border border-accent/10">
                            <p className="text-text-secondary dark:text-text-dark-secondary leading-relaxed">
                              {askMutation.data.answer}
                            </p>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="h-full flex flex-col items-center justify-center text-center opacity-40">
                        <MessageSquare className="w-12 h-12 mb-2" />
                        <p>Ask a question to start the conversation</p>
                      </div>
                    )}
                  </div>

                  {/* Quick Suggestions */}
                  <div className="flex flex-wrap gap-2 mb-4">
                    {[
                      'Risk Factors?',
                      'Retention Strategy?',
                      'Top Performers?',
                    ].map((q) => (
                      <button
                        key={q}
                        onClick={() => setQuestion(q)}
                        className="px-3 py-1.5 text-xs bg-surface/50 hover:bg-accent/10 border border-white/10 hover:border-accent/30 rounded-lg transition-colors text-text-secondary truncate max-w-[120px]"
                      >
                        {q}
                      </button>
                    ))}
                  </div>

                  <form onSubmit={handleAsk} className="relative mt-auto">
                    <textarea
                      value={question}
                      onChange={(e) => setQuestion(e.target.value)}
                      placeholder="Type your question..."
                      rows={3}
                      className="w-full pl-4 pr-4 py-3 bg-surface/50 dark:bg-black/20 border border-white/10 rounded-xl text-text-primary dark:text-text-dark-primary placeholder-text-muted focus:outline-none focus:ring-2 focus:ring-accent/50 resize-none text-sm"
                    />
                    <div className="absolute bottom-2 right-2">
                      <Button
                        type="submit"
                        size="sm"
                        disabled={!question.trim()}
                        isLoading={askMutation.isPending}
                        className="h-8 px-3 rounded-lg"
                      >
                        <Send className="w-3 h-3" />
                      </Button>
                    </div>
                  </form>
                </div>
              </GlassCard>
            </div>
          </div>
        )}

        {activeTab === 'lab' && (
          <GlassCard className="h-full overflow-hidden">
            <div className="h-full overflow-y-auto">
              <ModelLab />
            </div>
          </GlassCard>
        )}
      </div>
    </div>
  )
}
