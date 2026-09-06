'use client'

import { FormEvent, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api-client'
import { Button, EmptyState, Input, Page, PageHeader, SectionHeader, StateSummary, StatusBadge, Surface } from '@/components/ui'
import { ArrowRight, FileText, Search } from 'lucide-react'
import type { SearchResult, SearchStatus } from '@/types/api'

export default function SearchPage() {
  const [query, setQuery] = useState('')
  const [searchTerm, setSearchTerm] = useState('')
  const { data: status } = useQuery<SearchStatus>({ queryKey: ['search', 'status'], queryFn: () => api.search.getStatus() as Promise<SearchStatus> })
  const { data, isLoading } = useQuery<SearchResult>({ queryKey: ['search', 'results', searchTerm], queryFn: () => api.search.search(searchTerm, 10) as Promise<SearchResult>, enabled: searchTerm.length >= 3 })

  const submit = (event: FormEvent) => { event.preventDefault(); if (query.trim().length >= 3) setSearchTerm(query.trim()) }

  if (status?.available === false) return <Page><PageHeader eyebrow="Investigate · Research" title="Search is not available for this dataset" description="Semantic research only appears when the relevant text source is present and the optional indexing capability is available." /><StateSummary title="Capability unavailable" description={status.reason || 'Add a supported text source to enable semantic research.'} tone="warning" /></Page>

  const results = data?.results ?? []

  return (
    <Page>
      <PageHeader eyebrow="Investigate · Research" title="Search the evidence in workforce text" description="Use semantic retrieval to find relevant passages. Search results are evidence candidates, not conclusions or employee decisions." />

      <Surface padding="lg">
        <form onSubmit={submit} className="grid gap-4 md:grid-cols-[minmax(0,1fr)_auto] md:items-end">
          <Input label="Research query" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="e.g. leadership potential, manager support, career growth" leading={<Search className="h-4 w-4" />} />
          <Button type="submit" disabled={query.trim().length < 3} isLoading={isLoading}>Search evidence <ArrowRight className="h-4 w-4" /></Button>
        </form>
        <div className="mt-4 flex items-center gap-2 text-xs text-text-muted"><FileText className="h-3.5 w-3.5" />{status?.indexed_records ?? 0} indexed records</div>
      </Surface>

      {searchTerm ? <Surface padding="lg">
        <SectionHeader title={`Results for “${searchTerm}”`} description="Ranked by semantic similarity. Review the underlying text before drawing a conclusion." />
        <div className="mt-5 space-y-3">
          {isLoading ? <StateSummary title="Searching evidence" description="Comparing your query with the indexed workforce text." tone="info" /> : results.length ? results.map((result, index) => <article key={`${result.employee_id}-${index}`} className="rounded-2xl border border-border p-5"><div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between"><div><div className="flex flex-wrap items-center gap-2"><span className="font-semibold">Record {index + 1}</span>{result.dept ? <StatusBadge tone="neutral">{result.dept}</StatusBadge> : null}</div><p className="mt-3 text-sm leading-7 text-text-secondary">{result.text}</p></div><StatusBadge tone={result.similarity_score >= .8 ? 'success' : result.similarity_score >= .6 ? 'info' : 'neutral'}>{(result.similarity_score * 100).toFixed(0)}% match</StatusBadge></div></article>) : <EmptyState title="No relevant evidence found" description="Try a broader concept or different wording." />}
        </div>
      </Surface> : <Surface padding="lg"><EmptyState icon={Search} title="Start with a workforce question" description="Search works best for concepts and themes rather than exact employee identifiers." /></Surface>}

      <StateSummary title="Privacy and interpretation" description="Semantic search retrieves text evidence. It does not assign performance labels, recommend employment action, or convert similarity into a factual claim." tone="info" />
    </Page>
  )
}
