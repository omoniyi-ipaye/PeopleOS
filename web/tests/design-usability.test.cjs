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
const {InvestigationResult, AgentRunPanel, evidenceClaim} = require('../components/advisor/investigation-result')
const {AnswerDrawer} = require('../components/advisor/answer-drawer')
const {RelationshipInsight} = require('../components/relationship-insight')
const {ScenarioDrilldownBrief} = require('../components/scenario-drilldown-brief')
const appShellSource = fs.readFileSync(path.join(__dirname, '../components/app-shell.tsx'), 'utf8')
const WorkforceHealthPage = require('../app/workforce-health/page').default
const response = {
  request_id:'request-example',question:'What does workforce data show?',answer:'There are 80 active employees. [e-active]',
  status:'partial',confidence:.7,tools_used:['workforce.summary','experience'],model:null,warnings:['Sample data only.'],
  evidence:{unknowns:['No direct experience survey was supplied.'],contradictions:['Two recorded measures disagree.'],verification_notes:['Employee IDs reconciled.'],tool_results:[
    {result_id:'run1',tool_id:'workforce.summary',status:'success',summary:'Recorded workforce evidence.',warnings:[],evidence:[{evidence_id:'e-active',kind:'measurement',claim:'Active employees: 80',source_tool:'workforce.summary',metric:'active_count',value:80,confidence:.9}]},
    {result_id:'run2',tool_id:'experience',status:'blocked',summary:'Survey missing.',warnings:['Experience is unavailable.'],evidence:[]},
  ]},
}
test('investigation leads with a human answer while preserving inspectable evidence, gaps and tool outcomes',()=>{
 const html=render(InvestigationResult,{result:response,source:'workforce-a.csv · version 1'})
 assert.match(html,/PeopleOS answer/);assert.match(html,/Some evidence missing/);assert.match(html,/Two recorded measures disagree/)
 assert.match(html,/Why you can trust this answer/);assert.match(html,/Important limitations/)
 assert.match(html,/blocked/);assert.match(html,/e-active/);assert.match(html,/workforce-a.csv/)
 assert.match(html,/Evidence quality/);assert.match(html,/Coverage/);assert.match(html,/Tools with evidence/)
 assert.doesNotMatch(html,/Complete coverage/)
 assert.match(html,/<details[^>]*>/)
})
test('investigation exposes the governed agent path and approved next questions',()=>{
 const traced={...response,model:'local-model',agent_steps:[
  {id:'understand',label:'Understand the question',status:'complete',detail:'Mapped the question to a safe evidence plan.'},
  {id:'evidence',label:'Gather evidence',status:'complete',detail:'Ran an approved workforce check.',tools:['workforce.summary']},
  {id:'verify',label:'Check the evidence',status:'complete',detail:'The evidence supported a complete answer.'},
  {id:'explain',label:'Explain what it means',status:'complete',detail:'AI selected verified evidence; values stayed deterministic.'},
  {id:'next',label:'Suggest what to explore next',status:'complete',detail:'Prepared a bounded follow-up question.'},
 ],next_actions:[{label:'See workforce structure',question:'Headcount by department',reason:'Put the answer in workforce context.'}]}
 const html=render(AgentRunPanel,{result:traced})
 assert.match(html,/PeopleOS agent/);assert.match(html,/AI assisted/);assert.match(html,/Understand the question/);assert.match(html,/bounded follow-up question/)
})
test('strategic investigations translate evidence into practical people signals',()=>{
 const item=(metric,value,claim,metadata={})=>({evidence_id:`strategic-${metric}`,kind:'derived',claim,source_tool:'workforce.summary',metric,value,confidence:1,metadata})
 const strategic={...response,question:'What should I be paying attention to in this workforce?',status:'partial',tools_used:['workforce.summary','workforce.compensation_equity','workforce.fairness','workforce.employee_experience','workforce.organization_structure'],evidence:{...response.evidence,unknowns:[],tool_results:[{result_id:'strategic-run',tool_id:'workforce.summary',status:'success',summary:'Strategic workforce evidence.',warnings:[],evidence:[
  item('active_count',662,'Current active employee count: 662'),
  item('record_count',800,'Current employee record count: 800'),
  item('department_count',8,'Active department count: 8'),
  item('observed_attrition_share',.173,'Observed attrition share: 0.173',{measured_count:800}),
  item('unadjusted_gender_pay_gap_pct',2.2,'Unadjusted gender pay gap is 2.2%'),
  item('employee_experience_index',49.1,'Configured Employee Experience Index: 49.1/100'),
  item('attrition_outcome_disparity',.129,'Operations group differs by 12.9%',{attribute:'Department',group:'Operations',group_size:42}),
  item('critical_stagnation_count',27,'Configured critical role-duration threshold count: 27'),
 ]}]}}
 const html=render(InvestigationResult,{result:strategic,source:'sample_hr_data.csv · version 6',reportingCurrency:'USD'})
 const visibleAnswer=html.split('Technical details')[0]
 const visibleText=visibleAnswer.replace(/<[^>]+>/g,'')
 assert.match(visibleAnswer,/practical signals/)
 assert.match(visibleAnswer,/662 people are currently recorded as active across 8 departments/)
 assert.match(visibleAnswer,/17\.3% of 800 records with a known outcome are marked as departed/)
 assert.match(visibleAnswer,/unadjusted gender pay gap is 2\.2%/)
 assert.match(visibleAnswer,/configured index is 49\.1\/100/)
 assert.match(visibleAnswer,/Operations department differs from the overall recorded departure rate by 12\.9 percentage points/)
 assert.match(visibleAnswer,/27 people meet the configured role-duration threshold/)
 assert.doesNotMatch(visibleText,/r=|p=/)
 assert.match(visibleText,/do not explain cause/)
})
test('grouped investigations lead with plain People language and a visual breakdown',()=>{
 const grouped={...response,question:'Headcount by department',answer:'Headcount by department — Engineering: 220 (n=220); Sales: 143 (n=143).',status:'complete',confidence:.915,tools_used:['workforce.derived_analysis'],warnings:[],evidence:{coverage_score:1,unknowns:[],contradictions:[],verification_notes:[],tool_results:[{result_id:'grouped-run',tool_id:'workforce.derived_analysis',status:'success',summary:'Governed downstream aggregate analysis completed.',warnings:[],evidence:[{evidence_id:'grouped-evidence',kind:'derived',claim:'Governed downstream aggregate analysis completed.',source_tool:'workforce.derived_analysis',metric:'derived_analysis',value:{groups:[{group:'Engineering',value:220,measured_count:220,eligible_count:220,excluded_count:0},{group:'Sales',value:143,measured_count:143,eligible_count:143,excluded_count:0}],suppressed_groups:0},metadata:{analysis_spec:{operation:'group_summary',population:'active',group_by:'Dept',statistic:'count'},population_count:662}}]}]}}
 const html=render(InvestigationResult,{result:grouped,source:'sample_hr_data.csv · version 6',reportingCurrency:'USD'})
 assert.match(html,/There are 662 people/);assert.match(html,/People by department/);assert.match(html,/<table/)
 const visibleAnswer=html.split('Technical details')[0]
 assert.doesNotMatch(visibleAnswer,/n=220/);assert.doesNotMatch(visibleAnswer,/Headcount by department —/)
})
test('answer drawer keeps Ask PeopleOS conversational and exposes drill-down controls',()=>{
 const html=render(AnswerDrawer,{result:{answer:response,source:'workforce-a.csv · version 1'},loading:false,error:null,followUps:[{label:'By department',question:'Headcount by department',reason:'Put the answer in workforce context.'}],onClose:()=>{},onCancel:()=>{},onInvestigate:()=>{}})
 assert.match(html,/role="dialog"/)
 assert.match(html,/aria-modal="true"/)
 assert.match(html,/Close answer details/)
 assert.match(html,/Your answer/)
 assert.match(html,/Keep exploring/)
 assert.match(html,/By department/)
 assert.match(html,/conversation stays open behind this panel/)
})
test('investigation surfaces the grounded AI narrative above the verified evidence',()=>{
 const ai={...response,model:'local-narrator',synthesis_mode:'grounded_llm',answer:'The active workforce gives the People team a useful starting point. [e-active]'}
 const html=render(InvestigationResult,{result:ai,source:'workforce-a.csv · version 1'})
 assert.match(html,/AI explanation/)
 assert.match(html,/useful starting point/)
 assert.match(html,/AI composed from completed analysis/)
})
test('missing evidence values cannot turn into measured zero',()=>{
 for(const value of [null,undefined,'',false]) assert.equal(evidenceClaim({claim:'Recorded attrition: unavailable',metric:'observed_attrition_share',value}), 'Recorded attrition: unavailable')
 assert.equal(evidenceClaim({claim:'Recorded attrition: 0',metric:'observed_attrition_share',value:0}), 'Recorded attrition: 0.0%')
})
test('related patterns lead with a practical people meaning and keep statistics secondary',()=>{
 const html=render(RelationshipInsight,{signal:'Interview Score',outcome:'recorded departures',correlation:.13,observations:500,pValue:.027,context:'hiring',nextStep:'Compare roles before changing the interview process.'})
 const primary=html.split('<details')[0]
 assert.match(primary,/What this means in practice/)
 assert.match(primary,/people with higher Interview Score tended to have more recorded departures/)
 assert.match(primary,/Weak relationship/)
 assert.match(primary,/Based on 500 people with both measures recorded/)
 assert.match(primary,/Useful next check/)
 assert.match(primary,/Relationship score 0\.13/)
 assert.doesNotMatch(primary,/r=0\.13|p=0\.027/)
 assert.match(html,/Technical check: relationship score r=0\.13 · sample test p=0\.027/)
 assert.match(html,/does not tell us that one factor caused the other/)
})
test('workforce insights keep the department table separate from a responsive related-pattern board',()=>{
 const html=render(WorkforceHealthPage,{},[
  [['analytics','departments'],{departments:[{dept:'Engineering',headcount:220,turnover_rate:.18,avg_tenure:4.6,avg_rating:3.54}]}],
  [['analytics','summary'],{observed_attrition_share:.173,attrition_known_count:800,active_count:662}],
  [['analytics','correlations'],{correlations:[{feature:'InterviewScore',correlation:.13,abs_correlation:.13,observations:800,p_value:.027},{feature:'Salary',correlation:-.07,abs_correlation:.07,observations:800,p_value:.1}]}],
 ])
 assert.ok(html.indexOf('Department picture') < html.indexOf('Related patterns'))
 assert.match(html,/>2 patterns</)
 assert.match(html,/md:grid-cols-2 xl:grid-cols-3/)
 assert.match(html,/Relationship signal/)
 assert.match(html,/Interview Score/)
})
test('scenario drill-down translates model output into an HR decision brief',()=>{
 const html=render(ScenarioDrilldownBrief,{drilldown:{
  status:'complete',
  answer:'Bottom line: The second situation is modeled to leave USD 2M less net value than the first.',
  focus_label:'Financial trade-off',
  headline:'The second situation is modeled to leave USD 2M less net value than the first.',
  people_takeaway:'The model shows no difference in its modeled turnover-change result.',
  use_for:['Frame the trade-off with People and Finance owners.'],
  validate_next:['Confirm the assumptions with Finance.'],
  decision_boundary:'This is exploratory planning evidence, not a forecast.',
  selected_evidence:['impact','outcome'],
  warnings:[],
 }})
 assert.match(html,/People takeaway/)
 assert.match(html,/What this means for People/)
 assert.match(html,/How to use this/)
 assert.match(html,/What to validate next/)
 assert.match(html,/Financial trade-off/)
 assert.match(html,/Evidence used:[\s\S]*Modeled financial impact/)
 assert.match(html,/Show evidence explanation/)
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
test('advisor requires verified workforce data before accepting a question',()=>{
 const html=render(require('../app/advisor/page').default,{},[[['platform','status'],{data:{loaded:false},integrity:{status:'unavailable'}}]])
 assert.match(html,/Add workforce data first/);assert.match(html,/Add workforce data/)
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

test('evidence ledger separates investigation coverage from measured population coverage',()=>{
 const measured={...response,evidence:{...response.evidence,coverage_score:1,tool_results:[{...response.evidence.tool_results[0],evidence:[{...response.evidence.tool_results[0].evidence[0],metadata:{measured_count:1,eligible_count:100,excluded_count:99,population:'active_employees'}}]}]}}
 const html=render(InvestigationResult,{result:measured,source:'Incomplete measures'})
 assert.match(html,/Coverage/);assert.match(html,/Measured: 1/);assert.match(html,/Eligible: 100/);assert.match(html,/Excluded or missing: 99/)
})

test('navigation offers a native mobile dialog and only existing primary destinations',()=>{
 const html=render(require('../components/sidebar').Sidebar)
 assert.match(html,/aria-label="Open navigation"/)
 assert.match(html,/<dialog[^>]*aria-label="PeopleOS navigation"/)
 assert.match(html,/aria-label="Close navigation"/)
 assert.match(html,/md:flex/)
 assert.match(html,/>Home<|aria-label="Home"/)
 assert.match(html,/Ask PeopleOS/);assert.match(html,/Insights/);assert.match(html,/Plan/);assert.match(html,/Data/);assert.match(html,/Trust &amp; Privacy/)
 for(const match of html.matchAll(/href="([^"]+)"/g)) {
  const destination=match[1]
  assert.ok(fs.existsSync(path.join(__dirname,'../app',destination==='/' ? '' : destination,'page.tsx')), `Missing destination ${destination}`)
 }
 assert.doesNotMatch(html,/>Saved Investigations</)
})
test('empty root route uses a focused setup shell without application navigation',()=>{
 const {isSetupRoute}=require('../components/app-shell')
 assert.equal(isSetupRoute('/',undefined),true)
 assert.equal(isSetupRoute('/',{data:{loaded:false}}),true)
 assert.equal(isSetupRoute('/',{data:{loaded:true}}),false)
 assert.equal(isSetupRoute('/upload',{data:{loaded:false}}),false)
})
test('the shared app shell returns each navigated page to its top-level reading position',()=>{
 assert.match(appShellSource,/useEffect\(\(\) => \{[\s\S]*scrollTo\(\{ top: 0, left: 0, behavior: 'auto' \}\)[\s\S]*\}, \[pathname\]\)/)
 assert.match(appShellSource,/data-peopleos-scroll-container/)
})
test('header distinguishes historical source records from active employees and model metadata',()=>{
 const html=render(require('../components/header').Header,{},[[['platform','status'],{
  data:{loaded:true,row_count:240},integrity:{snapshot:{source_rows:240,active_rows:80}},workspace:{active_model:true},capabilities:{predictive_model:false},
 }]])
 assert.match(html,/240 source records · 80 active employees/)
 assert.doesNotMatch(html,/240 people|predictive model active|experimental model available/)
})

test('active workforce card explains all-unknown and partly-known employment populations',()=>{
 for(const [active,unknown] of [[0,120],[80,20]]) {
  const html=render(require('../app/page').default,{},[
   [['platform','status'],{data:{loaded:true},integrity:{status:'verified',snapshot:{source_name:'status-fixture.csv',current_rows:120,unknown_status_rows:unknown}}}],
   [['analytics','summary'],{active_count:active,observed_attrition_share:active===0 ? null : .2,tenure_mean:null}],
   [['analytics','departments'],{departments:[]}],
  ])
  assert.match(html,new RegExp(`>${active}<\\/div>`))
  assert.ok(html.includes(`${active} recorded active employees; ${unknown} unknown statuses excluded from the active count.`))
 }
})
