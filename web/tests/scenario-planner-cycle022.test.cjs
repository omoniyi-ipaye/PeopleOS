const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const test = require('node:test')

const source = fs.readFileSync(path.join(__dirname, '../app/scenario-planner/page.tsx'), 'utf8')

test('scenario planner binds rendered results to the full active snapshot', () => {
  assert.match(source, /current_fingerprint/)
  assert.match(source, /resultMatchesSnapshot/)
  assert.match(source, /storedResult\.provenance\.current_fingerprint === snapshot\.current_fingerprint/)
})

test('scenario planner exposes bounded aggregate inputs and cost timing', () => {
  assert.match(source, /min=\{0\} max=\{100\}/)
  assert.match(source, /adjustmentValue >= 0/)
  assert.match(source, /Cost timing/)
  assert.match(source, /configured draw share, not an empirical probability/)
  assert.match(source, /simple payback/)
  assert.match(source, /monetary amounts in/)
  assert.match(source, /reporting_currency/)
  assert.match(source, /valueClassName="whitespace-normal break-words text-xl leading-tight"/)
})

test('scenario planner presents AI drill-down as an HR decision brief', () => {
  assert.match(source, /ScenarioDrilldownBrief/)
  assert.match(source, /local AI prioritizes what to inspect/)
  assert.match(fs.readFileSync(path.join(__dirname, '../components/scenario-drilldown-brief.tsx'), 'utf8'), /What this means for People/)
  assert.match(fs.readFileSync(path.join(__dirname, '../components/scenario-drilldown-brief.tsx'), 'utf8'), /What to validate next/)
  assert.match(fs.readFileSync(path.join(__dirname, '../components/scenario-drilldown-brief.tsx'), 'utf8'), /Show evidence explanation/)
})
