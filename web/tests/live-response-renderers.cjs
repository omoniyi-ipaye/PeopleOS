// Render real application components from responses captured over the live API.
// This verifies server-rendered content, not hydration or browser interactions.
const fs = require('node:fs')
const path = require('node:path')
const assert = require('node:assert/strict')
const Module = require('node:module')
const ts = require('typescript')
const React = require('react')
const {renderToStaticMarkup} = require('react-dom/server')
const {QueryClient, QueryClientProvider} = require('@tanstack/react-query')
const directory = path.resolve(process.argv[2] || '../docs/validation/live-dummy-data')
const responses = JSON.parse(fs.readFileSync(path.join(directory,'responses.json'),'utf8'))
const resolve = Module._resolveFilename
Module._resolveFilename = function(request,...args) {
  return resolve.call(this,request.startsWith('@/') ? path.join(__dirname,'..',request.slice(2)) : request,...args)
}
for(const extension of ['.ts','.tsx']) require.extensions[extension] = (module,filename) => {
  module._compile(ts.transpileModule(fs.readFileSync(filename,'utf8'), {compilerOptions:{
    module:ts.ModuleKind.CommonJS,jsx:ts.JsxEmit.ReactJSX,esModuleInterop:true,target:ts.ScriptTarget.ES2020,
  }}).outputText,filename)
}
function render(component,fixtures) {
  const client=new QueryClient({defaultOptions:{queries:{retry:false,staleTime:Infinity,gcTime:0}}})
  for(const [key,name] of fixtures) client.setQueryData(key,responses[name].body)
  try {return renderToStaticMarkup(React.createElement(QueryClientProvider,{client},React.createElement(component)))}
  finally {client.clear()}
}
const Page=require('../app/page').default
const overview=render(Page,[[['platform','status'],'status-a'],[['analytics','summary'],'summary-a'],[['analytics','departments'],'departments-a']])
assert.match(overview,/Active workforce/);assert.match(overview,/>80<\/div>/)
assert.match(overview,/20\.0%/);assert.match(overview,/2\.0y/)
const missing=render(Page,[[['platform','status'],'status-b'],[['analytics','summary'],'summary-b'],[['analytics','departments'],'departments-b']])
assert.match(missing,/Recorded attrition evidence is unavailable/)
assert.doesNotMatch(missing,/0\.0%|below the current watch threshold/)
const compensation=render(require('../components/diagnostics/compensation-tab').CompensationTab,[[['compensation','analysis'],'compensation-a']])
assert.match(compensation,/6,000,000/);assert.match(compensation,/75,000/)
assert.doesNotMatch(compensation,/Overall fairness|NaN/)
const experience=render(require('../app/employee-experience/page').default,[[['experience','analysis'],'experience-a']])
assert.doesNotMatch(experience,/NaN|undefined/)
const sections=[['Known workforce',overview],['Unknown outcomes',missing],['Recorded compensation',compensation],['Missing experience evidence',experience]]
const checks=sections.map(([name])=>({name,passed:true}))
fs.writeFileSync(path.join(directory,'renderer-checks.json'),JSON.stringify({method:'React server rendering using captured live API responses; no browser hydration',checks},null,2)+'\n')
const cssRoot=path.join(__dirname,'../.next/static')
const css=fs.existsSync(cssRoot) ? fs.readdirSync(cssRoot,{recursive:true}).filter(p=>p.endsWith('.css')).map(p=>fs.readFileSync(path.join(cssRoot,p),'utf8')).join('\n') : ''
fs.writeFileSync(path.join(directory,'rendered-output.html'),'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PeopleOS dummy-data output validation</title><style>'+css+'</style></head><body style="padding:32px"><h1 style="font-size:28px;margin-bottom:24px">PeopleOS: captured dummy-data outputs</h1><p style="margin-bottom:24px">Static evidence from the real React components. Controls in this captured report are not interactive.</p>'+sections.map(([name,html])=>'<section style="margin-bottom:48px"><h2 style="font-size:22px;margin-bottom:16px">'+name+'</h2>'+html+'</section>').join('')+'</body></html>')
console.log('PASS: 4 real-component render cases using live API responses')
