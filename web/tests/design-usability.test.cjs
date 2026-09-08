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
const {InvestigationResult, evidenceClaim} = require('../components/advisor/investigation-result')
const response = {
  request_id:'request-example',question:'What does workforce data show?',answer:'There are 80 active employees. [e-active]',
  status:'partial',confidence:.7,tools_used:['workforce.summary','experience'],model:null,warnings:['Sample data only.'],
  evidence:{unknowns:['No direct experience survey was supplied.'],contradictions:['Two recorded measures disagree.'],verification_notes:['Employee IDs reconciled.'],tool_results:[
    {result_id:'run1',tool_id:'workforce.summary',status:'success',summary:'Recorded workforce evidence.',warnings:[],evidence:[{evidence_id:'e-active',kind:'measurement',claim:'Active employees: 80',source_tool:'workforce.summary',metric:'active_count',value:80,confidence:.9}]},
    {result_id:'run2',tool_id:'experience',status:'blocked',summary:'Survey missing.',warnings:['Experience is unavailable.'],evidence:[]},
  ]},
}
test('investigation exposes actual tool outcomes, missing evidence and conflicts',()=>{
 const html=render(InvestigationResult,{result:response,source:'workforce-a.csv · version 1'})
 assert.match(html,/Partial evidence/);assert.match(html,/No direct experience survey/);assert.match(html,/Two recorded measures disagree/)
 assert.match(html,/blocked/);assert.match(html,/e-active/);assert.match(html,/workforce-a.csv/);assert.match(html,/Employee IDs reconciled/)
 assert.match(html,/Tool evidence coverage<\/dt><dd[^>]*>Unavailable/)
 assert.doesNotMatch(html,/60%|Complete coverage/)
 assert.match(html,/<details><summary/)
})
test('missing evidence values cannot turn into measured zero',()=>{
 for(const value of [null,undefined,'',false]) assert.equal(evidenceClaim({claim:'Recorded attrition: unavailable',metric:'observed_attrition_share',value}), 'Recorded attrition: unavailable')
 assert.equal(evidenceClaim({claim:'Recorded attrition: 0',metric:'observed_attrition_share',value:0}), 'Recorded attrition: 0.0%')
})
test('invalid evidence scores remain unavailable rather than clamped to certainty',()=>{
 const html=render(InvestigationResult,{result:{...response,confidence:2,evidence:{...response.evidence,coverage_score:NaN}},source:'Test'})
 assert.doesNotMatch(html,/100%|NaN/)
 assert.match(html,/Unavailable/)
})
test('homepage uses captured live source identity and runtime model readiness',()=>{
 const live=JSON.parse(fs.readFileSync(path.join(__dirname,'../../docs/validation/live-dummy-data/responses.json'),'utf8'))
 const status={...live['status-a'].body,workspace:{active_model:true},capabilities:{predictive_model:false}}
 const html=render(require('../app/page').default,{},[[['platform','status'],status],[['analytics','summary'],live['summary-a'].body],[['analytics','departments'],live['departments-a'].body]])
 assert.match(html,/workforce-a.csv/);assert.match(html,/Dataset snapshot verified/);assert.match(html,/>80<\/div>/)
 assert.doesNotMatch(html,/Predictive model active|Experimental predictive model available/)
})
test('advisor requires a verified source before accepting a question',()=>{
 const html=render(require('../app/advisor/page').default,{},[[['platform','status'],{data:{loaded:false},integrity:{status:'unavailable'}}]])
 assert.match(html,/Add a verified workforce dataset first/);assert.match(html,/Add workforce data/)
 assert.match(html,/<textarea[^>]*disabled/)
})
test('homepage connection failure is distinguishable from an empty installation',()=>{
 const client=new QueryClient({defaultOptions:{queries:{retry:false,retryOnMount:false,gcTime:0}}})
 client.getQueryCache().build(client,{queryKey:['platform','status']}).setState({status:'error',fetchStatus:'idle',error:new Error('Offline'),errorUpdatedAt:Date.now()})
 try {
  const html=renderToStaticMarkup(React.createElement(QueryClientProvider,{client},React.createElement(require('../app/page').default)))
  assert.match(html,/PeopleOS connection unavailable/);assert.match(html,/Retry connection/)
  assert.doesNotMatch(html,/Explore with sample data/)
 } finally {client.clear()}
})

test('evidence ledger distinguishes tool execution coverage from measured population',()=>{
 const measured={...response,evidence:{...response.evidence,coverage_score:1,tool_results:[{...response.evidence.tool_results[0],evidence:[{...response.evidence.tool_results[0].evidence[0],metadata:{measured_count:1,eligible_count:100,excluded_count:99,population:'active_employees'}}]}]}}
 const html=render(InvestigationResult,{result:measured,source:'Incomplete measures'})
 assert.match(html,/Tool evidence coverage/);assert.match(html,/Measured: 1/);assert.match(html,/Eligible population: 100/);assert.match(html,/Excluded or missing: 99/)
})

test('navigation offers a native mobile dialog and only existing application destinations',()=>{
 const html=render(require('../components/sidebar').Sidebar)
 assert.match(html,/aria-label="Open navigation"/)
 assert.match(html,/<dialog[^>]*aria-label="PeopleOS navigation"/)
 assert.match(html,/aria-label="Close navigation"/)
 assert.match(html,/md:flex/)
 for(const match of html.matchAll(/href="([^"]+)"/g)) {
  const destination=match[1]
  assert.ok(fs.existsSync(path.join(__dirname,'../app',destination==='/' ? '' : destination,'page.tsx')), `Missing destination ${destination}`)
 }
 assert.doesNotMatch(html,/Saved Investigations|\/sessions/)
})
test('header distinguishes historical source records from active employees and model metadata',()=>{
 const html=render(require('../components/header').Header,{},[[['platform','status'],{
  data:{loaded:true,row_count:240},integrity:{snapshot:{source_rows:240,active_rows:80}},workspace:{active_model:true},capabilities:{predictive_model:false},
 }]])
 assert.match(html,/240 source records · 80 active employees/)
 assert.doesNotMatch(html,/240 people|predictive model active|experimental model available/)
})
