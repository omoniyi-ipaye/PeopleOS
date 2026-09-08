// Actual React rendering with controlled API evidence; not browser or live-LLM validation.
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
    module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop:true, target:ts.ScriptTarget.ES2020,
  }}).outputText, filename)
}
function render(Component, props = {}, fixtures = []) {
  const client = new QueryClient({defaultOptions:{queries:{retry:false,staleTime:Infinity,gcTime:0}}})
  for (const [key, value] of fixtures) client.setQueryData(key, value)
  try { return renderToStaticMarkup(React.createElement(QueryClientProvider,{client},React.createElement(Component,props))) }
  finally { client.clear() }
}
const Page = require('../app/page').default
test('currency formatting never assumes dollars and rejects non-finite amounts', () => {
  const {formatCurrency} = require('../lib/utils')
  assert.equal(formatCurrency(60000, 'eur'), 'EUR 60,000')
  assert.equal(formatCurrency(60000), '60,000')
  assert.equal(formatCurrency(60000, 'EURO'), '60,000')
  assert.equal(formatCurrency(Infinity, 'EUR'), 'Unavailable')
})
function cockpit(summary) {
  return render(Page,{},[
    [['platform','status'],{data:{loaded:true}}],
    [['analytics','summary'],{active_count:12,tenure_mean:0,...summary}],
    [['analytics','departments'],{departments:[]}],
  ])
}
for (const share of [0, 0.149, 0.15, 0.8, 1]) test(`observed share ${share} stays descriptive without a validated comparison`, () => {
  const html = cockpit({observed_attrition_share:share})
  assert.match(html,/Recorded employee outcomes/)
  assert.match(html,/no comparison benchmark has been established/)
  assert.ok(html.includes(`${(share*100).toFixed(1)}%`))
  assert.doesNotMatch(html,/watch threshold|deserves investigation|bg-amber-100/)
})
test('an explicit null observation cannot fall back to stale legacy turnover', () => {
  const html = cockpit({observed_attrition_share:null,turnover_rate:0.75})
  assert.match(html,/Recorded attrition evidence is unavailable/)
  assert.doesNotMatch(html,/75\.0%/)
})
test('invalid observed fractions fail closed rather than displaying impossible percentages', () => {
  for (const value of [-0.1,1.5,NaN,Infinity]) {
    const html = cockpit({observed_attrition_share:value})
    assert.match(html,/Recorded attrition evidence is unavailable/)
    assert.doesNotMatch(html,/NaN|Infinity|150\.0%|-10\.0%/)
  }
})
test('department chart distinguishes missing measurements from measured zero', () => {
  const {DepartmentBarChart} = require('../components/charts/department-bar-chart')
  assert.match(render(DepartmentBarChart,{data:[{dept:'A',headcount:1,avg_salary:null}],dataKey:'avg_salary'}),/measurements are unavailable/)
  assert.doesNotMatch(render(DepartmentBarChart,{data:[{dept:'A',headcount:0}]}),/measurements are unavailable/)
})
test('department tooltip labels observed shares accurately and never invents zero', () => {
  const {DepartmentBarChart} = require('../components/charts/department-bar-chart')
  const tree = DepartmentBarChart({data:[{dept:'A',headcount:4,turnover_rate:0.25}],dataKey:'turnover_rate'})
  const children = React.Children.toArray(tree.props.children.props.children)
  const tooltip = children.find(child => typeof child.props.formatter === 'function')
  const axis = children.find(child => typeof child.props.tickFormatter === 'function')
  assert.deepEqual(tooltip.props.formatter(0.25),['25.0%','Observed attrition share (not period turnover)'])
  assert.deepEqual(tooltip.props.formatter(null),['Unavailable','Observed attrition share (not period turnover)'])
  assert.equal(axis.props.tickFormatter(0.25),'25.0%')
})
test('salary labels do not assert conversion or annualization that was not performed', () => {
  const {CompensationTab} = require('../components/diagnostics/compensation-tab')
  const html = render(CompensationTab,{},[[['compensation','analysis'],{summary:{avg_salary:null,total_payroll:null,headcount:0},equity_scores:[],warnings:['Mixed pay periods cannot be aggregated.']}]])
  assert.match(html,/Unavailable/)
  assert.match(html,/Mixed pay periods cannot be aggregated/)
  assert.doesNotMatch(html,/follow the source contract/)
})
