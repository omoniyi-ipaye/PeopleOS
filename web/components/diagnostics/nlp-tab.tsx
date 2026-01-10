'use client'

import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api-client'
import { Card } from '@/components/ui/card'
import { KPICard } from '@/components/dashboard/kpi-card'
import { MessageSquare, Smile, Search, Brain, Star, AlertCircle } from 'lucide-react'
import { Badge } from '@/components/ui/badge'

export function NLPTab() {
    const { data, isLoading, error } = useQuery<any>({
        queryKey: ['nlp', 'analysis'],
        queryFn: api.nlp.getAnalysis as any,
    })

    if (isLoading) return <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {[1, 2, 3, 4].map(i => (
                <div key={i} className="h-24 bg-surface dark:bg-surface-dark rounded-xl animate-pulse" />
            ))}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="h-96 bg-surface dark:bg-surface-dark rounded-xl animate-pulse" />
            <div className="h-96 bg-surface dark:bg-surface-dark rounded-xl animate-pulse" />
        </div>
        <div className="p-4 bg-accent/5 rounded-xl border border-accent/10 flex items-center justify-center gap-3">
            <Brain className="w-5 h-5 text-accent animate-bounce" />
            <div className="text-sm font-medium text-accent">Deep analysis in progress... please wait (this can take up to 2-3 minutes)</div>
        </div>
    </div>

    if (error) return (
        <div className="p-8 text-center bg-surface dark:bg-surface-dark rounded-2xl border border-dashed border-border dark:border-border-dark">
            <AlertCircle className="w-12 h-12 text-danger mx-auto mb-4 opacity-50" />
            <h3 className="text-lg font-bold text-text-primary dark:text-text-dark-primary mb-2">Analysis Failed</h3>
            <p className="text-sm text-text-secondary dark:text-text-dark-secondary">
                {(error as Error).message || "Could not load NLP analysis"}
            </p>
        </div>
    )

    const { sentiment_summary, topics, skills, nlp_available } = data || { sentiment_summary: {}, topics: [], skills: {}, nlp_available: false }

    return (
        <div className="space-y-6">
            {!nlp_available && (
                <div className="p-4 rounded-xl bg-warning/5 border border-warning/20 flex items-center gap-3">
                    <AlertCircle className="w-5 h-5 text-warning shrink-0" />
                    <div>
                        <div className="text-sm font-bold text-warning">AI Features Offline</div>
                        <div className="text-[10px] text-warning/80">Ollama is Not Connected. AI-powered insights are currently unavailable.</div>
                    </div>
                </div>
            )}

            {/* Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <KPICard
                    title="Tone of Feedback"
                    value={sentiment_summary.avg_sentiment?.toFixed(2) || '0.00'}
                    icon={Smile}
                    subtitle="Overall sentiment (0-1)"
                    variant={sentiment_summary.avg_sentiment > 0.6 ? 'success' : sentiment_summary.avg_sentiment < 0.4 ? 'danger' : 'warning'}
                    insight="The average emotional tone of performance feedback, where higher means more positive."
                />
                <KPICard
                    title="Positive Feedback"
                    value={`${(sentiment_summary.positive_pct || 0).toFixed(0)}%`}
                    icon={Brain}
                    subtitle="Review sentiment"
                    insight="The percentage of performance reviews with a primarily positive tone."
                />
                <KPICard
                    title="Main Themes"
                    value={topics.length}
                    icon={MessageSquare}
                    subtitle="Extracted by AI"
                    insight="Common topics and discussion points identified across your workforce feedback."
                />
                <KPICard
                    title="AI Status"
                    value={nlp_available ? "Active" : "Offline"}
                    icon={Search}
                    subtitle={nlp_available ? "AI Processing Enabled" : "Action Required"}
                    variant={nlp_available ? "success" : "danger"}
                    insight="Indicates if the advanced AI engine is currently processing your workforce data."
                />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Dominant Topics */}
                <Card title="Dominant Themes" subtitle="Common patterns across performance reviews">
                    <div className="space-y-4">
                        {topics.map((topic: any) => (
                            <div key={topic.name} className="p-4 rounded-xl bg-surface-hover dark:bg-surface-dark-hover border border-border dark:border-border-dark group hover:border-accent/40 transition-colors">
                                <div className="flex justify-between items-start mb-2">
                                    <div className="font-bold text-text-primary dark:text-text-dark-primary group-hover:text-accent transition-colors">{topic.name}</div>
                                    <Badge variant="default" className="bg-accent/10 text-accent border-accent/20">
                                        {topic.prevalence}
                                    </Badge>
                                </div>
                                <p className="text-xs text-text-secondary dark:text-text-dark-secondary leading-relaxed">
                                    {topic.description}
                                </p>
                                {topic.sentiment && (
                                    <div className="mt-3 flex items-center gap-2">
                                        <div className={`w-1.5 h-1.5 rounded-full ${topic.sentiment === 'Positive' ? 'bg-success' : 'bg-danger'}`} />
                                        <span className="text-[10px] uppercase font-bold tracking-wider text-text-muted">{topic.sentiment} Sentiment</span>
                                    </div>
                                )}
                            </div>
                        ))}
                        {topics.length === 0 && (
                            <div className="h-64 flex items-center justify-center text-text-muted text-sm">
                                No themes identified yet
                            </div>
                        )}
                    </div>
                </Card>

                {/* Skill Inventory */}
                <Card title="Skill Inventory" subtitle="Technical & soft skills mentioned in reviews">
                    <div className="space-y-6">
                        <div>
                            <div className="text-xs font-bold text-text-muted uppercase tracking-widest mb-3 flex items-center gap-2">
                                <Brain className="w-3.5 h-3.5" />
                                Technical Skills
                            </div>
                            <div className="flex flex-wrap gap-2">
                                {skills.technical_skills?.map((skill: string) => (
                                    <span key={skill} className="px-3 py-1 bg-surface-hover dark:bg-surface-dark-hover border border-border dark:border-border-dark rounded-full text-[11px] text-text-primary dark:text-text-dark-primary">
                                        {skill}
                                    </span>
                                ))}
                            </div>
                        </div>

                        <div>
                            <div className="text-xs font-bold text-text-muted uppercase tracking-widest mb-3 flex items-center gap-2">
                                <Star className="w-3.5 h-3.5 text-success" />
                                Soft Skills
                            </div>
                            <div className="flex flex-wrap gap-2">
                                {skills.soft_skills?.map((skill: string) => (
                                    <span key={skill} className="px-3 py-1 bg-success/5 border border-success/10 rounded-full text-[11px] text-success">
                                        {skill}
                                    </span>
                                ))}
                            </div>
                        </div>
                    </div>
                </Card>
            </div>
        </div>
    )
}
