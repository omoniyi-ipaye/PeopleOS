# PeopleOS Enterprise Design System

## Status
TARGET foundation implemented; product migration in progress.

## Purpose
PeopleOS must communicate workforce evidence, uncertainty, governance and action readiness consistently. The design system is therefore part of the product control surface, not a styling library.

## Architecture

```text
Foundations
  tokens · typography · color semantics · spacing · radius · elevation · motion · focus
      ↓
Primitives
  Button · Surface · Card · StatusBadge · Tooltip · form controls
      ↓
Semantic components
  MetricCard · EmptyState · StateSummary · PageHeader · SectionHeader · Feedback states
      ↓
Product patterns
  Decision briefing · Evidence ledger · Trust state · Dataset lifecycle · Model lifecycle
      ↓
Compositions
  Decision Cockpit · People Intelligence · Data & Sources · Trust Center · Retention Signals
```

## Rules
1. Product pages do not invent visual semantics. Color, radius, elevation, focus and state meaning come from the design system.
2. Green means a verified positive/available/healthy state. Amber means attention/limited/not-applicable where action may be required. Red means failed/destructive/blocked. Violet is product action/accent, never health.
3. Model absence is not success and not failure. It is `not-active` or `not-applicable` depending on context.
4. Data and model confidence are separate from evidence coverage.
5. Consequential employment actions must never be visually framed as a primary automated CTA.
6. All interactive components require keyboard focus, disabled state, loading state where asynchronous, and an accessible name.
7. Empty states explain why the state exists and provide the next valid action.
8. Loading states preserve context; avoid full-screen decorative loaders for normal navigation.
9. Technical identifiers and traces are progressive disclosure, not default product content.
10. Motion communicates state change only; no decorative animation may delay access to information.

## Foundations
### Color semantics
- neutral: structure and descriptive information
- accent: product navigation/action
- info: contextual information
- success: healthy/verified/available
- warning: partial/attention/governed action required
- danger: failure/destructive/block

### Shape
- controls: `rounded-xl`
- surfaces: `rounded-2xl`
- major panels: `rounded-3xl`
- pills/status: `rounded-full`
Arbitrary radii are not permitted in new product work.

### Elevation
- base: default bordered surface
- raised: interactive or emphasized surface
- overlay: modal/popover only
Glass effects are legacy-compatible only and must not define information hierarchy.

### Motion
- control feedback: 150ms
- surface transitions: 200ms
- respect reduced-motion preferences

## Component contracts
### Button
Owns variants, sizes, focus-visible, disabled, loading and `aria-busy`. Product pages must not hand-build button state styling.

### Surface / Card
Surface owns semantic tone, border, radius, padding and elevation. Card adds title/subtitle/action composition. `GlassCard` is a compatibility wrapper and is deprecated for new work.

### StatusBadge / StatusDot
Status components are the only default visual language for lifecycle and health state labels.

### MetricCard
Metric presentation requires: label, value, optional context, optional state. A metric card must never imply causality from correlation.

### EmptyState
Every empty state contains: condition, explanation, valid next action. It must not present a system exception as user guidance.

### PageHeader / SectionHeader
All primary screens use consistent information hierarchy and action placement.

## Product patterns
### Decision briefing
Finding → why it matters → supporting context → next investigation.

### Evidence pattern
Finding → evidence → coverage → confidence → limitations → provenance/trace (progressive disclosure).

### Trust pattern
Data state → model state → runtime state → access boundary → recovery/governance boundary.

### Lifecycle pattern
Registered → validated → active; model lifecycle is separate from dataset activation.

## Accessibility acceptance
- WCAG 2.2 AA target.
- Visible keyboard focus on all controls.
- Semantic heading order.
- Status never communicated by color alone.
- 44px preferred touch targets for primary actions; 32px minimum for compact desktop controls.
- Dialogs/overlays must manage focus and Escape behavior.
- Charts require textual finding/summary before or adjacent to visualization.
- Reduced-motion users must receive no essential information only through animation.

## Migration policy
- New components use `web/design-system/tokens.ts` and governed primitives.
- Existing `GlassCard`, Bento and legacy analytics components remain transition compatibility only.
- Primary product surfaces migrate first: shell, Decision Cockpit, People Intelligence, Data & Sources, Trust Center, Retention Signals, Saved Investigations.
- Legacy analytics screens migrate by user-journey priority, not file order.

## Readiness
**BUILD READY WITH ASSUMPTIONS**

Assumption: several legacy analytics/diagnostic screens still contain page-local styling and old UI helpers. They remain transition debt until migrated to semantic primitives. This must stay visible in design-system review and must not be described as fully TARGET-complete.
