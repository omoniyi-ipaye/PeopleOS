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
test('compensation shows source data and dispersion without hardcoded fairness or a false empty outlier conclusion', () => {
  const {CompensationTab} = require('../components/diagnostics/compensation-tab')
  const html = render(CompensationTab,{},[[['compensation','analysis'],{summary:{avg_salary:12345,total_payroll:24690,headcount:2},equity_scores:[{dept:'Engineering',headcount:2,equity_score:.72}],warnings:[]}]])
  assert.match(html,/12,345/);assert.match(html,/72/);assert.match(html,/Salary observations/)
  assert.doesNotMatch(html,/Overall fairness|No significant outliers|\$12,345/)
})
test('SHAP rejects missing or nonadditive baselines and keeps measured zero', () => {
  const {ShapWaterfallChart} = require('../components/charts/shap-waterfall-chart')
  assert.match(render(ShapWaterfallChart,{features:[],baseValue:NaN,prediction:.5}),/Explanation unavailable/)
  assert.match(render(ShapWaterfallChart,{features:[{feature:'x',value:1,contribution:.2}],baseValue:.1,prediction:.9}),/do not reconcile/)
  const html=render(ShapWaterfallChart,{features:[{feature:'x',value:0,contribution:.2}],baseValue:0,prediction:.2})
  assert.match(html,/0.000/);assert.doesNotMatch(html,/Explanation unavailable|Increases Risk/)
})
test('correlations are coefficients on a bounded scale, never percent causal effects', () => {
  const {CorrelationHeatmap} = require('../components/charts/correlation-heatmap')
  const html=render(CorrelationHeatmap,{data:[{feature:'Salary',correlation:.8,abs_correlation:.8}]})
  assert.match(html,/0.80/);assert.match(html,/width:80%/);assert.doesNotMatch(html,/240%|80%<|Increases Risk/)
})
test('NLP missing inference stays unavailable and generated themes have no fabricated prevalence', () => {
  const {NLPTab}=require('../components/diagnostics/nlp-tab')
  const html=render(NLPTab,{},[[['nlp','analysis'],{nlp_available:false,sentiment_summary:{avg_sentiment:null,positive_pct:0},topics:[{name:'Support',description:'Team support',prevalence:null}],skills:{}}]])
  assert.match(html,/Unavailable/);assert.match(html,/Prevalence not measured/);assert.doesNotMatch(html,/>0%/)
})
test('model lab missing backtest does not become zero accuracy or an optimization recommendation', () => {
  const {ModelLab}=require('../components/advisor/model-lab')
  const html=render(ModelLab,{},[[['model-lab','validation'],{status:'warning',metrics:null,message:'Mature outcomes required',interpretation:'No prospective evidence'}],[['model-lab','sensitivity'],[]]])
  assert.match(html,/Prospective accuracy metrics: Unavailable/);assert.doesNotMatch(html,/One-Click|0%|Excellent/)
})
test('experience renderer tolerates unmeasured segment averages', () => {
  const Page=require('../app/employee-experience/page').default
  const html=render(Page,{},[[['experience','analysis'],{experience_index:{available:true,overall_exi:50,respondent_count:1,response_coverage:.5},segments:{segments:[{segment:'Unavailable',count:1,percentage:50,avg_exi:null}]},drivers:{drivers:[]},lifecycle:{stages:[]},signals:{total_signals:1,has_pulse:true},summary:{},warnings:[],recommendations:[]}]])
  assert.doesNotMatch(html,/NaN|undefined/)
})
test('methodology modal explains limits without hardcoded feature ranks or fairness guarantees', () => {
  const {PredictiveMathModal}=require('../components/dashboard/predictive-math-modal')
  const html=render(PredictiveMathModal,{onClose:()=>{}})
  assert.match(html,/does not establish a 75% chance/)
  assert.match(html,/does not establish fairness/)
  assert.doesNotMatch(html,/High Impact|ensure unbiased outcomes/)
})
test('retired individual risk renderer cannot turn a heuristic into a departure probability', () => {
  const {RiskAnalysisModal}=require('../components/risk-analysis-modal')
  const html=render(RiskAnalysisModal,{isOpen:true,onClose:()=>{},employee:{EmployeeID:'E0',risk_category:'High',risk_score:75}})
  assert.match(html,/Individual predictive view unavailable/)
  assert.doesNotMatch(html,/75%|stay interview within 48 hours/)
})
