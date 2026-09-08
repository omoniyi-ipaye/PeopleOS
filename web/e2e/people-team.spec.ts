import { test, expect, type Page } from '@playwright/test'

// Independent known-answer fixture: no imported production calculation helpers.
function workforce(name = 'A', missingOutcomes = false) {
  const fields = ['EmployeeID', 'Dept', 'Salary', 'Tenure', 'LastRating', 'Age', 'Gender', 'JobTitle', 'JobLevel', 'Location', 'HireDate', 'ManagerID', 'Attrition', 'SnapshotDate']
  // B satisfies the 50-row import minimum with 30 active and 30 unknown statuses.
  const rows = Array.from({ length: name === 'A' ? 120 : 60 }, (_, i) => [
    String(i).padStart(4, '0'), ['001', '002', ''][i % 3],
    name === 'A' ? (i < 40 ? 60000 : 90000) : 40000,
    name === 'A' ? 2 : 0, 4, 30, i % 2 ? 'Female' : 'Male', 'Analyst', 'L03',
    'Madrid', '2024-01-01', '0000', missingOutcomes ? '' : name === 'B' ? (i < 30 ? 0 : '') : (i < 80 ? 0 : i < 100 ? 1 : ''), '2026-01-01',
  ])
  return Buffer.from([fields.join(','), ...rows.map(row => row.join(','))].join('\n'))
}

async function upload(page: Page, name = 'A', missingOutcomes = false) {
  if (page.url() === 'about:blank') await page.goto('/')
  // Use the product navigation so previous analytics remain in the same QueryClient.
  await expect(page.locator('aside').first()).toBeAttached()
  const menu = page.getByRole('button', { name: 'Open navigation', exact: true })
  const mobileNavigation = await menu.isVisible()
  if (mobileNavigation) await menu.click()
  await page.getByRole('link', { name: 'Data & Sources', exact: true }).click()
  await expect(page).toHaveURL(/\/upload$/)
  if (mobileNavigation) await expect(page.getByRole('dialog', { name: 'PeopleOS navigation' })).toBeHidden()
  const response = page.waitForResponse(r => r.url().endsWith('/api/upload') && r.request().method() === 'POST')
  await page.locator('input[type=file]').setInputFiles({ name: `workforce-${name}.csv`, mimeType: 'text/csv', buffer: workforce(name, missingOutcomes) })
  const uploadResponse = await response
  expect(uploadResponse.ok(), await uploadResponse.text()).toBeTruthy()
  await expect(page.getByText('Dataset activated', { exact: true })).toBeVisible()
  // Client navigation retains React Query state, so this also checks invalidation.
  await page.getByRole('link', { name: 'Open Decision Cockpit' }).click()
  await expect(page.getByText('Active workforce', { exact: true })).toBeVisible()
}

async function metric(page: Page, label: string, value: string) {
  const card = page.getByText(label, { exact: true }).locator('../..')
  await expect(card.getByText(value, { exact: true })).toBeVisible()
}

async function investigate(page: Page, question: string) {
  await page.getByLabel('Investigation question').fill(question)
  const response = page.waitForResponse(r => r.url().endsWith('/api/intelligence/investigate') && r.request().method() === 'POST')
  await page.getByRole('button', { name: 'Investigate', exact: true }).click()
  return response
}

async function screenshot(page: Page, name: string) {
  await test.info().attach(name, { body: await page.screenshot({ fullPage: true }), contentType: 'image/png' })
}

async function noHorizontalOverflow(page: Page) {
  const dimensions = await page.evaluate(() => ({ width: document.documentElement.clientWidth, scroll: document.documentElement.scrollWidth }))
  expect(dimensions.scroll).toBeLessThanOrEqual(dimensions.width + 1)
}

test.beforeEach(async ({ request }) => {
  const reset = await request.post('/api/upload/reset')
  expect(reset.ok()).toBeTruthy()
})

test('People analyst uploads, verifies figures, investigates evidence and switches data', async ({ page }) => {
  const browserErrors: string[] = []
  page.on('pageerror', error => browserErrors.push(error.message))
  await upload(page)
  await metric(page, 'Active workforce', '80')
  await metric(page, 'Observed attrition share', '20.0%')
  await metric(page, 'Average active tenure', '2.0y')
  await noHorizontalOverflow(page)
  await screenshot(page, 'known-answer-cockpit')

  await page.getByRole('link', { name: 'Ask PeopleOS', exact: true }).click()
  const response = await investigate(page, 'How many active employees do we have and what is our recorded attrition share?')
  expect(response.ok()).toBeTruthy()
  const answer = await response.json()
  const evidence = answer.evidence.tool_results.flatMap((tool: { evidence: { metric: string; value: number }[] }) => tool.evidence)
  expect(evidence).toEqual(expect.arrayContaining([expect.objectContaining({ metric: 'headcount', value: 80 })]))
  expect(answer.model).toBeNull() // No Ollama service or external LLM required by this gate.
  await expect(page.getByText(/Deterministic synthesis.*Review the evidence/)).toBeVisible()
  await page.locator('summary').filter({ hasText: 'Evidence ledger' }).click()
  await expect(page.getByText('Current active employee count: 80', { exact: true })).toBeVisible()
  await noHorizontalOverflow(page)
  await screenshot(page, 'advisor-evidence-ledger')

  await page.getByRole('link', { name: 'Review data integrity and capability status' }).click()
  await expect(page.getByText('Snapshot verified', { exact: true })).toBeVisible()
  await expect(page.getByText(/120 source rows · 120 current employee records · 80 active · 20 unknown statuses/)).toBeVisible()
  await noHorizontalOverflow(page)
  await screenshot(page, 'trust-center')

  await upload(page, 'B')
  await metric(page, 'Active workforce', '30')
  await metric(page, 'Observed attrition share', '0.0%')
  await metric(page, 'Average active tenure', '0.0y')
  await expect(page.getByText('80', { exact: true })).toHaveCount(0)
  await page.reload()
  await metric(page, 'Active workforce', '30')
  expect(browserErrors).toEqual([])
})

test('missing outcomes remain unavailable rather than becoming zero or reassuring', async ({ page }) => {
  await upload(page, 'A', true)
  await expect(page.getByText('0 recorded active employees; 120 unknown statuses excluded from the active count.', { exact: true })).toBeVisible()
  await metric(page, 'Observed attrition share', '—')
  await expect(page.getByText('Recorded attrition evidence is unavailable', { exact: true })).toBeVisible()
  await expect(page.getByText(/below the current watch threshold/)).toHaveCount(0)
  await screenshot(page, 'unknown-outcomes')
})

test('agent blocks investigation without data and unsupported questions stay insufficient', async ({ page }) => {
  await page.goto('/advisor')
  await expect(page.getByText('Add a verified workforce dataset first', { exact: true })).toBeVisible()
  await expect(page.getByLabel('Investigation question')).toBeDisabled()
  await expect(page.getByRole('button', { name: 'Investigate', exact: true })).toBeDisabled()
  await expect(page.getByRole('link', { name: 'Add workforce data', exact: true })).toBeVisible()
  await screenshot(page, 'agent-no-data')
  await upload(page)
  await page.getByRole('link', { name: 'Ask PeopleOS', exact: true }).click()
  const unsupported = await investigate(page, 'What is the weather tomorrow in Madrid?')
  expect(unsupported.ok()).toBeTruthy()
  expect((await unsupported.json()).status).toBe('insufficient')
  await expect(page.getByText('Insufficient evidence', { exact: true }).first()).toBeVisible()
  await screenshot(page, 'unsupported-question')
})

test('an open agent answer is hidden after another tab activates a different dataset', async ({ page, context }) => {
  await upload(page)
  await page.getByRole('link', { name: 'Ask PeopleOS', exact: true }).click()
  const first = await investigate(page, 'How many active employees do we have?')
  expect(first.ok()).toBeTruthy()
  await expect(page.getByText('Investigation result', { exact: true })).toBeVisible()
  const secondTab = await context.newPage()
  await upload(secondTab, 'B')
  await page.bringToFront()
  await expect(page.getByText('The data source has changed', { exact: true })).toBeVisible({ timeout: 30_000 })
  await expect(page.getByText('Investigation result', { exact: true })).toHaveCount(0)
  const refreshed = await investigate(page, 'How many active employees do we have?')
  expect(refreshed.ok()).toBeTruthy()
  const answer = await refreshed.json()
  const evidence = answer.evidence.tool_results.flatMap((tool: { evidence: { metric: string; value: number }[] }) => tool.evidence)
  expect(evidence).toEqual(expect.arrayContaining([expect.objectContaining({ metric: 'headcount', value: 30 })]))
  await screenshot(page, 'refreshed-agent-dataset')
  await secondTab.close()
})
