// Route-level edge-state checks using the real React components and controlled API fixtures.
const {test} = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const Module = require('node:module')
const ts = require('typescript')
const React = require('react')
const {renderToStaticMarkup} = require('react-dom/server')
const {QueryClient, QueryClientProvider} = require('@tanstack/react-query')

const resolve = Module._resolveFilename
Module._resolveFilename = function(request, ...args) {
  return resolve.call(this, request.startsWith('@/') ? path.join(__dirname, '..', request.slice(2)) : request, ...args)
}
for (const extension of ['.ts', '.tsx']) require.extensions[extension] = (module, filename) => {
  module._compile(ts.transpileModule(fs.readFileSync(filename, 'utf8'), {compilerOptions:{
    module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true, target: ts.ScriptTarget.ES2020,
  }}).outputText, filename)
}

function render(Component, fixtures = [], errors = []) {
  const client = new QueryClient({defaultOptions:{queries:{retry:false,retryOnMount:false,staleTime:Infinity,gcTime:0}}})
  for (const [key, value] of fixtures) client.setQueryData(key, value)
  for (const [key, message] of errors) client.getQueryCache().build(client, {queryKey:key}).setState({status:'error', fetchStatus:'idle', error:new Error(message), errorUpdatedAt:Date.now()})
  try { return renderToStaticMarkup(React.createElement(QueryClientProvider, {client}, React.createElement(Component))) }
  finally { client.clear() }
}

test('research waits for capability and exposes capability failure without claiming zero indexed records', () => {
  const Page = require('../app/search/page').default
  const failed = render(Page, [], [[['search','status'], 'offline']])
  assert.match(failed, /Research capability is unavailable/)
  assert.match(failed, /Retry capability check/)
  assert.doesNotMatch(failed, /0 indexed records|Search the evidence in workforce text/)
})

test('unmeasured experience stays clearly unavailable and never invents respondent coverage', () => {
  const Page = require('../app/employee-experience/page').default
  const html = render(Page, [[['experience','analysis'], {
    experience_index:{available:false,reason:'Survey fields missing'},
    segments:{available:false},drivers:{available:false},lifecycle:{available:false},
    signals:{has_enps:false,has_pulse:false,total_signals:0},summary:{health_indicator:'Unknown',total_employees:120,at_risk_count:0},warnings:[],recommendations:[],
  }]])
  assert.match(html, /No measured experience data yet/)
  assert.match(html, /will not guess engagement/)
  assert.doesNotMatch(html, /0\.0% response coverage|0 measured respondents/)
})

test('empty department payload remains unavailable and supporting query failures stay visible', () => {
  const Page = require('../app/workforce-health/page').default
  const html = render(Page, [[['analytics','departments'], {departments:[],total_departments:0}]], [
    [['analytics','correlations'], 'correlation failure'], [['analytics','high-risk-departments'], 'risk failure'],
  ])
  assert.match(html, /No workforce insight is available yet/)
  assert.match(html, /Some supporting analysis is temporarily unavailable/)
  assert.match(html, /Active people[\s\S]*Unavailable/)
})

test('platform failure cannot masquerade as an inactive predictive model', () => {
  const Page = require('../app/flight-risk/page').default
  const html = render(Page, [], [[['platform','status'], 'offline']])
  assert.match(html, /Predictive capability state is unavailable/)
  assert.doesNotMatch(html, /Predictive retention signals are not active/)
})

test('data controls fail closed while source state cannot be verified', () => {
  const Page = require('../app/upload/page').default
  const html = render(Page, [], [[['upload','status'], 'offline']])
  assert.match(html, /Your data source could not be checked/)
  assert.match(html, />Retry<\/button>/)
  assert.match(html, /Choose file<\/button>/)
  assert.match(html, /disabled[^>]*>[\s\S]*Explore with sample data/)
})

test('scenario inputs expose bounded numeric contracts and bound select labels', () => {
  const Page = require('../app/scenario-planner/page').default
  const html = render(Page, [
    [['analytics','departments'], {departments:[{dept:'Very long department name used to validate wrapping behavior'}]}],
    [['platform','status'], {integrity:{snapshot:{generation:'g1'}}}],
  ])
  assert.match(html, /min="-100" max="100" step="0\.1"/)
  assert.match(html, /for="scenario-scope"/)
  assert.match(html, /id="scenario-scope"/)
})

test('quality-of-hire omitted summary fields do not become measured zero', () => {
  const Page = require('../app/quality-of-hire/page').default
  const html = render(Page, [[['quality-of-hire','analysis'], {summary:{},source_effectiveness:[],correlations:{correlations:[]},warnings:[],recommendations:[]} ]])
  assert.match(html, /People represented[\s\S]*Unavailable/)
  assert.match(html, /Source cohorts[\s\S]*Unavailable/)
  assert.match(html, /Pre-hire measures[\s\S]*Unavailable/)
  assert.match(html, /role="tablist"/)
  assert.match(html, /role="tab" aria-selected="true"/)
  assert.match(html, /role="tabpanel"/)
})
