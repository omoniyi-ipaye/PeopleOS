'use client'

import { FormEvent, useState } from 'react'
import Link from 'next/link'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api-client'
import { Button, EmptyState, Input, Page, PageHeader, SectionHeader, StateSummary, StatusBadge, Surface } from '@/components/ui'
import { ArrowRight, Database, FileText, Search, ShieldCheck, Sparkles } from 'lucide-react'
import type { SearchResult, SearchStatus } from '@/types/api'

export default function SearchPage() {
  const [query, setQuery] = useState('')
  const [searchTerm, setSearchTerm] = useState('')
  const { data: status } = useQuery<SearchStatus>({ queryKey: ['search', 'status'], queryFn: () => api.search.getStatus() as Promise<SearchStatus> })
  const { data, isLoading, isError } = useQuery<SearchResult>({ queryKey: ['search', 'results', searchTerm], queryFn: () => api.search.search(searchTerm, 10) as Promise<SearchResult>, enabled: searchTerm.length >= 3 })

  const submit = (event: FormEvent) => { event.preventDefault(); if (query.trim().length >= 3) setSearchTerm(query.trim()) }

  if (status?.available === false) return <Page>
    <PageHeader eyebrow="Investigate · Research" title="Search is not available for this dataset" description="Semantic research appears only when the active dataset contains supported workforce text and an evidence index has been prepared." />
    <StateSummary title="Structured evidence is still available" description={status.reason || 'This dataset does not currently provide the text evidence required for semantic research.'} tone="info" />

    <div className="grid gap-6 lg:grid-cols-3">
      <Surface padding="lg">
        <div className="grid h-10 w-10 place-items-center rounded-xl bg-accent/10 text-accent"><FileText className="h-5 w-5" /></div>
        <h2 className="mt-4 font-semibold">What unlocks Research</h2>
        <p className="mt-2 text-sm leading-6 text-text-secondary">Add supported narrative fields such as performance text, survey comments or other approved workforce text sources.</p>
      </Surface>
      <Surface padding="lg">
        <div className="grid h-10 w-10 place-items-center rounded-xl bg-accent/10 text-accent"><Sparkles className="h-5 w-5" /></div>
        <h2 className="mt-4 font-semibold">What you can do now</h2>
        <p className="mt-2 text-sm leading-6 text-text-secondary">People Intelligence can still investigate structured workforce evidence such as attrition, compensation, tenure and organisational structure.</p>
      </Surface>
      <Surface padding="lg">
        <div className="grid h-10 w-10 place-items-center rounded-xl bg-accent/10 text-accent"><ShieldCheck className="h-5 w-5" /></div>
        <h2 className="mt-4 font-semibold">Why it is gated</h2>
        <p className="mt-2 text-sm leading-6 text-text-secondary">PeopleOS does not invent a search index or imply text evidence exists when the required source is missing.</p>
      </Surface>
    </div>

    <Surface padding="md" className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
      <div><div className="font-semibold">Choose the next useful path</div><div className="text-sm text-text-secondary">Add richer evidence or continue with the structured evidence already available.</div></div>
      <div className="flex flex-wrap gap-2">
        <Link href="/upload" className="inline-flex items-center gap-2 rounded-xl border border-border px-4 py-2.5 text-sm font-semibold text-text-primary transition hover:bg-background-secondary"><Database className="h-4 w-4" />Data & Sources</Link>
        <Link href="/advisor" className="inline-flex items-center gap-2 rounded-xl bg-accent px-4 py-2.5 text-sm font-semibold text-white transition hover:opacity-90">Open People Intelligence <ArrowRight className="h-4 w-4" /></Link>
      </div>
    </Surface>
  </Page>

  const results = data?.results ?? []

  return (
    <Page>
      <PageHeader eyebrow="Investigate · Research" title="Search the evidence in workforce text" description="Use semantic retrieval to find relevant passages. Search results are evidence candidates. Ranking scores order the retrieved passages; they are not probabilities or validated relevance ratings." />

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
          {isLoading ? <StateSummary title="Searching evidence" description="Comparing your query with the indexed workforce text." tone="info" /> : isError ? <StateSummary title="Search unavailable" description="The search could not complete. Retry before interpreting the results." tone="info" /> : results.length ? results.map((result, index) => <article key={`${result.employee_id}-${index}`} className="rounded-2xl border border-border p-5"><div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between"><div><div className="flex flex-wrap items-center gap-2"><span className="font-semibold">Record {index + 1}</span>{result.dept ? <StatusBadge tone="neutral">{result.dept}</StatusBadge> : null}</div><p className="mt-3 text-sm leading-7 text-text-secondary">{result.text}</p></div><StatusBadge tone="neutral">Ranking score {result.similarity_score.toFixed(3)}</StatusBadge></div></article>) : <EmptyState title="No relevant evidence found" description="Try a broader concept or different wording." />}
        </div>
      </Surface> : <Surface padding="lg"><EmptyState icon={Search} title="Start with a workforce question" description="Search works best for concepts and themes rather than exact employee identifiers." /></Surface>}

      <StateSummary title="Privacy and interpretation" description="Semantic search retrieves text evidence; it does not turn similarity into a factual claim or employment recommendation." tone="info" />
    </Page>
  )
}
