// Renderer contracts for structural terminology and retired risk semantics.
const {test} = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const glossary = fs.readFileSync(path.join(__dirname, '../components/glossary-modal.tsx'), 'utf8')

test('structural glossary describes role duration without flight-risk or promotion-readiness inference', () => {
  assert.match(glossary, /recorded years in a current role divided by recorded tenure/)
  assert.match(glossary, /do not indicate flight risk, disengagement, performance, or promotion readiness/)
  assert.match(glossary, /valid index of 0\.8/)
  assert.doesNotMatch(glossary, /stagnation index of 1\.5/)
})

test('span glossary keeps reporting links descriptive and non-causal', () => {
  assert.match(glossary, /recorded direct reports linked to a manager/)
  assert.match(glossary, /not evidence of burnout, manager effectiveness, team health, or causal organizational impact/)
})
