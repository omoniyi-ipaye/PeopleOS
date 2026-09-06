'use client'

import { Activity, AlertTriangle, ArrowRight, CheckCircle2, Database, Info, ShieldCheck } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { EmptyState, MetricCard, StateSummary } from '@/components/ui/data-display'
import { Input, Textarea } from '@/components/ui/field'
import { Page, PageHeader, SectionHeader } from '@/components/ui/page'
import { StatusBadge, StatusDot } from '@/components/ui/status'
import { Surface } from '@/components/ui/surface'

export default function DesignSystemPage() {
  return (
    <Page>
      <PageHeader
        eyebrow="PeopleOS Design System"
        title="Enterprise component reference"
        description="Foundations and components used to communicate workforce evidence, trust, state and action consistently. This surface is for product and engineering review."
      />

      <Card>
        <SectionHeader title="Experience-state contract" description="Every state must feel intentional. A user should always understand what happened, what remains usable and what to do next." />
        <div className="mt-5 grid gap-4 lg:grid-cols-3">
          <Surface padding="md">
            <div className="text-xs font-semibold uppercase tracking-wider text-text-muted">01 · Explain</div>
            <div className="mt-2 font-semibold">Say what this state means</div>
            <p className="mt-2 text-sm leading-6 text-text-secondary">Use business language. Distinguish unavailable, empty, partial, blocked and failed states instead of collapsing them into generic errors.</p>
          </Surface>
          <Surface padding="md">
            <div className="text-xs font-semibold uppercase tracking-wider text-text-muted">02 · Preserve value</div>
            <div className="mt-2 font-semibold">Show what still works</div>
            <p className="mt-2 text-sm leading-6 text-text-secondary">A missing model, index or optional source must never make the whole product feel broken. Surface the trustworthy capabilities that remain available.</p>
          </Surface>
          <Surface padding="md">
            <div className="text-xs font-semibold uppercase tracking-wider text-text-muted">03 · Continue</div>
            <div className="mt-2 font-semibold">Give one clear next step</div>
            <p className="mt-2 text-sm leading-6 text-text-secondary">Every state ends with a useful continuation: investigate, add evidence, review lifecycle state, retry, or return to a trusted analysis.</p>
          </Surface>
        </div>
        <div className="mt-5 flex flex-wrap items-center gap-2 text-xs text-text-muted">
          <StatusBadge tone="success">No dead ends</StatusBadge>
          <StatusBadge tone="success">No raw errors</StatusBadge>
          <StatusBadge tone="success">One primary action</StatusBadge>
          <StatusBadge tone="success">Progressive technical detail</StatusBadge>
          <span className="inline-flex items-center gap-1 font-medium text-accent">State → understanding → continuation <ArrowRight className="h-3.5 w-3.5" /></span>
        </div>
      </Card>

      <Card>
        <SectionHeader title="Semantic state language" description="State meaning is fixed across every product workflow." />
        <div className="flex flex-wrap gap-2">
          <StatusBadge>Neutral</StatusBadge>
          <StatusBadge tone="info">Information</StatusBadge>
          <StatusBadge tone="success">Verified / available</StatusBadge>
          <StatusBadge tone="warning">Attention / partial</StatusBadge>
          <StatusBadge tone="danger">Failed / blocked</StatusBadge>
          <StatusBadge tone="accent">Action / product accent</StatusBadge>
        </div>
        <div className="mt-5 flex flex-wrap gap-5">
          <StatusDot tone="success" label="Dataset active" />
          <StatusDot tone="warning" label="Model not active" />
          <StatusDot tone="info" label="Deterministic fallback" />
        </div>
      </Card>

      <Card>
        <SectionHeader title="Actions" description="Buttons own focus, disabled, loading and destructive semantics." />
        <div className="flex flex-wrap gap-3">
          <Button>Primary action</Button>
          <Button variant="secondary">Secondary</Button>
          <Button variant="outline">Outline</Button>
          <Button variant="ghost">Ghost</Button>
          <Button variant="danger">Destructive</Button>
          <Button isLoading>Working</Button>
          <Button disabled>Disabled</Button>
        </div>
      </Card>

      <Card>
        <SectionHeader title="Inputs" description="Accessible names, helper text and validation are built into the primitive." />
        <div className="grid gap-5 md:grid-cols-2">
          <Input label="Investigation name" placeholder="September retention review" description="Use a name others can understand later." />
          <Input label="Required field" required error="This field needs a value." placeholder="Required" />
          <div className="md:col-span-2"><Textarea label="Workforce question" placeholder="What changed in observed attrition this quarter?" description="PeopleOS will only answer from available aggregate evidence." /></div>
        </div>
      </Card>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="People" value="800" detail="Active dataset" icon={<Database className="h-4 w-4" />} status={<StatusBadge tone="success">Active</StatusBadge>} />
        <MetricCard label="Model" value="Not active" detail="Predictive lifecycle is optional" icon={<Activity className="h-4 w-4" />} status={<StatusBadge tone="warning">Optional</StatusBadge>} />
        <MetricCard label="Evidence coverage" value="75%" detail="Three of four planned tools contributed" icon={<ShieldCheck className="h-4 w-4" />} status={<StatusBadge tone="info">Partial</StatusBadge>} />
        <MetricCard label="Evidence quality" value="91%" detail="Heuristic quality of available investigation evidence — not probability of truth" icon={<CheckCircle2 className="h-4 w-4" />} status={<StatusBadge tone="success">Strong</StatusBadge>} />
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Surface tone="info"><div className="flex gap-3"><Info className="h-5 w-5 shrink-0 text-sky-600" /><div><div className="font-semibold">Information</div><div className="mt-1 text-sm">Context that changes how a user should interpret the result.</div></div></div></Surface>
        <Surface tone="warning"><div className="flex gap-3"><AlertTriangle className="h-5 w-5 shrink-0 text-amber-600" /><div><div className="font-semibold">Attention</div><div className="mt-1 text-sm">A capability is partial, unavailable or requires governed action.</div></div></div></Surface>
        <Surface tone="success"><div className="flex gap-3"><CheckCircle2 className="h-5 w-5 shrink-0 text-emerald-600" /><div><div className="font-semibold">Verified</div><div className="mt-1 text-sm">A deterministic check or governed capability is healthy.</div></div></div></Surface>
      </div>

      <Card title="Lifecycle state example" subtitle="State rows use the same semantic language everywhere.">
        <StateSummary label="Dataset" value="Active" tone="success" />
        <StateSummary label="Predictive model" value="Not active" tone="warning" />
        <StateSummary label="Local AI" value="Unavailable" tone="neutral" />
      </Card>

      <EmptyState
        title="No saved investigations yet"
        description="Completed investigations will appear here once a user explicitly saves one."
        icon={<Database className="h-5 w-5" />}
        action={<Button>Start investigation</Button>}
      />
    </Page>
  )
}
