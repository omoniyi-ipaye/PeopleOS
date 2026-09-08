// Render real React components with controlled query evidence; no model/server calls.
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
  const compiled = ts.transpileModule(fs.readFileSync(filename,'utf8'), {compilerOptions: {
    module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true, target: ts.ScriptTarget.ES2020,
  }})
  module._compile(compiled.outputText, filename)
}
function render(Component, props={}, fixtures=[]) {
  const client = new QueryClient({defaultOptions:{queries:{retry:false,staleTime:Infinity,gcTime:0}}})
  for (const [key,value] of fixtures) client.setQueryData(key,value)
  try { return renderToStaticMarkup(React.createElement(QueryClientProvider,{client},React.createElement(Component,props))) }
  finally {client.clear()}
}
// Correctness contracts converted from the independent specialist reproductions.
const Page = require('../app/page').default
const status = {data:{loaded:true,row_count:42},workspace:{active_model:false}}
test('missing outcomes stay unavailable without false reassurance', () => {
  const html = render(Page,{},[
    [['platform','status'],status],
    [['analytics','summary'],{observed_attrition_share:null,turnover_rate:null,active_count:2,tenure_mean:3}],
    [['analytics','departments'],{departments:[]}],
  ])
  assert.doesNotMatch(html,/0\.0%/)
  assert.doesNotMatch(html,/below the current watch threshold/)
  assert.match(html,/Recorded attrition evidence is unavailable/)
})
test('loading summary cannot invent an active population or attrition share', () => {
  const html = render(Page,{},[[['platform','status'],status]])
  assert.doesNotMatch(html,/>42<\/div>/)
  assert.doesNotMatch(html,/0\.0%/)
  assert.match(html,/Loading workforce evidence/)
})
test('measured zero tenure is preserved', () => {
  const html = render(Page,{},[
    [['platform','status'],status],
    [['analytics','summary'],{observed_attrition_share:0,active_count:2,tenure_mean:0}],
  ])
  assert.match(html,/Average active tenure/)
  assert.match(html,/0\.0y/)
})

test('legacy individual components cannot expose supplied employee predictions', () => {
  const {EmployeeDetailModal} = require('../components/dashboard/employee-detail-modal')
  const {HighRiskTable} = require('../components/dashboard/high-risk-table')
  const modal=render(EmployeeDetailModal,{employeeId:'PRIVATE-ID',onClose:()=>{}})
  const table=render(HighRiskTable,{employees:[{employee_id:'PRIVATE-ID',risk_score:.95,risk_category:'High'}]})
  assert.match(modal,/Individual predictive view unavailable/)
  assert.match(table,/Employee risk ranking is unavailable/)
  assert.doesNotMatch(modal+table,/PRIVATE-ID|95%/)
})

test('homepage summary failure renders an explicit unavailable state', () => {
  const client=new QueryClient({defaultOptions:{queries:{retry:false,retryOnMount:false,gcTime:0}}})
  client.setQueryData(['platform','status'],status)
  client.getQueryCache().build(client,{queryKey:['analytics','summary']}).setState({
    status:'error',fetchStatus:'idle',error:new Error('Unavailable data'),errorUpdatedAt:Date.now(),
  })
  try {
    const html=renderToStaticMarkup(React.createElement(QueryClientProvider,{client},React.createElement(Page)))
    assert.match(html,/Workforce evidence is unavailable/)
    assert.doesNotMatch(html,/0\.0%|below the current watch threshold/)
  } finally {client.clear()}
})
