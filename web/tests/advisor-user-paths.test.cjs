const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const test = require('node:test')

const source = fs.readFileSync(path.join(__dirname, '../app/advisor/page.tsx'), 'utf8')

test('starter questions use supported one-click investigation queries', () => {
  assert.match(source, /Start with workforce structure[\s\S]*Headcount by department/)
  assert.match(source, /Where is recorded attrition\?[\s\S]*Recorded attrition share by department/)
  assert.match(source, /How does pay vary\?[\s\S]*Average salary by department/)
  assert.match(source, /How are teams structured\?[\s\S]*Headcount by role/)
  assert.match(source, /onClick=\{\(\) => onInvestigate\(prompt\)\}/)
})

test('insight deep links run their prefilled question automatically', () => {
  assert.match(source, /initialQueryRun/)
  assert.match(source, /void investigate\(prompt\)/)
})
