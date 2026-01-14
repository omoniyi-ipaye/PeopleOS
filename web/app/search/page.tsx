'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { GlassCard } from '@/components/ui/glass-card'
import { Badge } from '@/components/ui/badge'
import { api } from '@/lib/api-client'
import { cn } from '@/lib/utils'
import { Search, FileText, AlertTriangle, Sparkles, Command, ArrowRight } from 'lucide-react'

import {
  SearchResult,
  SearchStatus,
} from '@/types/api'

export default function SearchPage() {
  const [query, setQuery] = useState('')
  const [searchTerm, setSearchTerm] = useState('')
  const [isFocused, setIsFocused] = useState(false)

  const { data: searchStatus } = useQuery<SearchStatus>({
    queryKey: ['search', 'status'],
    queryFn: api.search.getStatus as any,
  })

  const { data: searchResults, isLoading } = useQuery<SearchResult>({
    queryKey: ['search', 'results', searchTerm],
    queryFn: (() => api.search.search(searchTerm, 10)) as any,
    enabled: !!searchTerm && searchTerm.length >= 3,
  })

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (query.length >= 3) {
      setSearchTerm(query)
    }
  }

  // Unavailable State
  if (searchStatus?.available === false) {
    return (
      <div className="h-[calc(100vh-100px)] flex flex-col items-center justify-center text-center p-6">
        <div className="p-6 rounded-full bg-warning/10 mb-6 relative">
          <div className="absolute inset-0 bg-warning/20 blur-xl rounded-full" />
          <AlertTriangle className="w-16 h-16 text-warning relative z-10" />
        </div>
        <h2 className="text-3xl font-display font-bold text-text-primary dark:text-white mb-3">Search Unavailable</h2>
        <p className="text-text-secondary dark:text-text-dark-secondary max-w-md text-lg leading-relaxed">
          {searchStatus?.reason || 'Upload data with a "PerformanceText" column to enable semantic search.'}
        </p>
      </div>
    )
  }

  const hasResults = searchResults?.results && searchResults.results.length > 0

  return (
    <div className="h-[calc(100vh-100px)] flex flex-col relative overflow-hidden">
      {/* Background Decor */}
      <div className="absolute top-0 right-0 w-96 h-96 bg-accent/5 rounded-full blur-3xl -z-10 animate-pulse-subtle" />
      <div className="absolute bottom-0 left-0 w-64 h-64 bg-purple-500/5 rounded-full blur-3xl -z-10" />

      {/* Main Content Container with Scroll */}
      <div className="flex-1 overflow-y-auto custom-scrollbar px-1 py-6">
        <div className={cn(
          "transition-all duration-700 ease-in-out flex flex-col items-center",
          hasResults || searchTerm ? "justify-start pt-4" : "justify-center h-full"
        )}>

          {/* Hero Section */}
          <div className={cn(
            "w-full max-w-3xl transition-all duration-700 text-center mb-8",
            hasResults || searchTerm ? "scale-95 opacity-90" : "scale-100"
          )}>
            <h1 className={cn(
              "font-display font-bold text-gradient bg-clip-text text-transparent bg-gradient-to-r from-gray-900 to-gray-500 dark:from-white dark:to-gray-400 transition-all duration-500",
              hasResults || searchTerm ? "text-3xl mb-4" : "text-5xl mb-6"
            )}>
              PeopleOS Research
            </h1>
            {!hasResults && !searchTerm && (
              <p className="text-xl text-text-secondary dark:text-text-dark-secondary font-light max-w-xl mx-auto mb-10">
                Ask questions about your workforce in natural language. Uncover hidden insights in performance reviews.
              </p>
            )}
          </div>

          {/* Search Bar */}
          <div className="w-full max-w-2xl relative z-20">
            <form onSubmit={handleSearch} className="relative">
              <div className={cn(
                "p-2 flex items-center bg-white dark:bg-slate-800 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-xl transition-all duration-300",
                isFocused ? "ring-2 ring-accent/20 border-accent" : ""
              )}>
                <div className="pl-4 pr-3 text-text-muted">
                  <Search className={cn("w-6 h-6 transition-colors", isFocused ? "text-accent" : "")} />
                </div>
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onFocus={() => setIsFocused(true)}
                  onBlur={() => setIsFocused(false)}
                  placeholder="e.g. 'Show me employees with strong leadership potential'..."
                  className="flex-1 bg-transparent border-none outline-none text-lg h-14 text-text-primary dark:text-white placeholder:text-text-muted/50"
                />
                <div className="pr-1">
                  <button
                    type="submit"
                    disabled={query.length < 3 || isLoading}
                    className="bg-accent hover:bg-accent/90 text-white p-3 rounded-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-md"
                  >
                    {isLoading ? (
                      <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    ) : (
                      <ArrowRight className="w-5 h-5" />
                    )}
                  </button>
                </div>
              </div>
            </form>

            {/* Search Stats / Hints */}
            <div className="mt-4 flex items-center justify-between text-sm px-4">
              <div className="flex items-center gap-2 text-text-secondary">
                <FileText className="w-4 h-4" />
                <span>{searchStatus?.indexed_records || 0} documents indexed</span>
              </div>
              {!hasResults && !searchTerm && (
                <div className="flex items-center gap-2 text-text-muted">
                  <Command className="w-3.5 h-3.5" />
                  <span>Try "High performer"</span>
                </div>
              )}
            </div>
          </div>

          {/* Results Grid */}
          <div className="w-full max-w-5xl mt-12 space-y-6">
            {hasResults && (
              <div className="animate-in fade-in slide-in-from-bottom-8 duration-700">
                <div className="flex items-center justify-between mb-4 px-2">
                  <h3 className="text-lg font-bold text-text-secondary">
                    Results for <span className="text-text-primary dark:text-white">"{searchTerm}"</span>
                  </h3>
                  <Badge variant="outline" className="bg-accent/5 text-accent border-accent/20">
                    {searchResults?.results.length} matches found
                  </Badge>
                </div>

                <div className="grid gap-4">
                  {searchResults.results.map((result, i) => (
                    <GlassCard key={`${result.employee_id}-${i}`} className="group hover:border-accent/30 transition-all duration-300">
                      <div className="p-5 flex gap-4">
                        {/* Score Indicator */}
                        <div className="flex flex-col items-center justify-center p-3 bg-surface-secondary/50 rounded-xl border border-white/5 h-fit min-w-[80px]">
                          <span className={cn(
                            "text-2xl font-bold font-display",
                            result.similarity_score > 0.8 ? "text-success" :
                              result.similarity_score > 0.6 ? "text-warning" : "text-text-secondary"
                          )}>
                            {(result.similarity_score * 100).toFixed(0)}%
                          </span>
                          <span className="text-[10px] text-text-muted uppercase tracking-wider">Match</span>
                        </div>

                        {/* Content */}
                        <div className="flex-1 space-y-2">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-3">
                              <h4 className="font-bold text-lg text-text-primary dark:text-white group-hover:text-accent transition-colors">
                                {result.employee_id}
                              </h4>
                              <Badge variant="outline" className="bg-surface-secondary dark:bg-white/5">
                                {result.dept}
                              </Badge>
                            </div>
                          </div>
                          <p className="text-text-secondary dark:text-text-dark-secondary leading-relaxed text-sm">
                            "{result.text}"
                          </p>
                        </div>
                      </div>
                    </GlassCard>
                  ))}
                </div>
              </div>
            )}

            {/* Empty State */}
            {searchTerm && !hasResults && !isLoading && (
              <div className="text-center py-20 animate-in fade-in zoom-in-95 duration-500">
                <div className="inline-flex p-6 rounded-full bg-surface-secondary/50 mb-6">
                  <Search className="w-12 h-12 text-text-muted" />
                </div>
                <h3 className="text-xl font-bold mb-2">No matches found</h3>
                <p className="text-text-secondary">Try adjusting your search terms or be less specific.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
