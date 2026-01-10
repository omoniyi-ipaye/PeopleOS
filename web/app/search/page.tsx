'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { api } from '@/lib/api-client'
import { Search, FileText, AlertTriangle } from 'lucide-react'

import {
  SearchResult,
  SearchStatus,
} from '@/types/api'

export default function SearchPage() {
  const [query, setQuery] = useState('')
  const [searchTerm, setSearchTerm] = useState('')

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

  if (!searchStatus?.available) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-text-primary dark:text-text-dark-primary">Semantic Search</h1>
          <p className="text-text-secondary dark:text-text-dark-secondary mt-1">
            Search performance reviews using AI
          </p>
        </div>

        <div className="flex flex-col items-center justify-center h-64 gap-4 text-center">
          <AlertTriangle className="w-12 h-12 text-warning" />
          <h2 className="text-xl font-semibold text-text-primary dark:text-text-dark-primary">Search Unavailable</h2>
          <p className="text-text-secondary dark:text-text-dark-secondary max-w-md">
            {searchStatus?.reason ||
              'Upload data with PerformanceText column to enable semantic search.'}
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Page Title */}
      <div>
        <h1 className="text-2xl font-bold text-text-primary dark:text-text-dark-primary">Semantic Search</h1>
        <p className="text-text-secondary dark:text-text-dark-secondary mt-1">
          Search performance reviews using natural language
        </p>
      </div>

      {/* Search Bar */}
      <Card padding="lg">
        <form onSubmit={handleSearch} className="flex gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-text-muted" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search performance reviews (e.g., 'leadership skills', 'needs improvement')..."
              className="w-full pl-12 pr-4 py-3 bg-surface dark:bg-surface-dark border border-border dark:border-border-dark rounded-lg text-text-primary dark:text-text-dark-primary placeholder-text-muted dark:placeholder-text-dark-muted focus:outline-none focus:ring-2 focus:ring-accent/50 shadow-inner"
            />
          </div>
          <Button type="submit" disabled={query.length < 3} isLoading={isLoading}>
            Search
          </Button>
        </form>

        <div className="flex items-center gap-2 mt-4 text-sm text-text-muted dark:text-text-dark-muted">
          <FileText className="w-4 h-4" />
          <span>{searchStatus.indexed_records} reviews indexed</span>
        </div>
      </Card>

      {/* Search Results */}
      {searchResults?.results && searchResults.results.length > 0 && (
        <Card title={`Search Results for "${searchTerm}"`}>
          <div className="space-y-4">
            {searchResults.results.map((result: any, index: number) => (
              <div
                key={`${result.employee_id}-${index}`}
                className="p-4 rounded-lg bg-surface hover:bg-surface-hover dark:bg-surface-dark hover:dark:bg-surface-dark-hover border border-border dark:border-border-dark transition-colors shadow-sm"
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-text-primary dark:text-text-dark-primary">{result.employee_id}</span>
                    <Badge variant="info" size="sm">
                      {result.dept}
                    </Badge>
                  </div>
                  <div className="text-sm font-medium text-text-muted dark:text-text-dark-muted">
                    Similarity: {(result.similarity_score * 100).toFixed(0)}%
                  </div>
                </div>
                <p className="text-sm text-text-secondary dark:text-text-dark-secondary leading-relaxed">
                  {result.text}
                </p>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* No Results */}
      {searchTerm && searchResults?.results?.length === 0 && !isLoading && (
        <div className="text-center py-12 text-text-secondary dark:text-text-dark-secondary">
          No results found for "{searchTerm}"
        </div>
      )}

      {/* Search Tips */}
      {!searchTerm && (
        <Card title="Search Tips" className="max-w-2xl">
          <ul className="space-y-2 text-sm text-text-secondary dark:text-text-dark-secondary">
            <li className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-accent shadow-sm" />
              Use natural language queries like "strong team player"
            </li>
            <li className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-accent shadow-sm" />
              Search for skills: "communication", "leadership", "technical"
            </li>
            <li className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-accent shadow-sm" />
              Find improvement areas: "needs development", "growth opportunity"
            </li>
            <li className="flex items-center gap-2 text-text-muted dark:text-text-dark-muted">
              <span className="w-1.5 h-1.5 rounded-full bg-border shadow-sm" />
              Minimum 3 characters required
            </li>
          </ul>
        </Card>
      )}
    </div>
  )
}
