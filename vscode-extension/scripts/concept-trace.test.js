"use strict";
const test=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs'),path=require('node:path');
const {conceptTraceHtml,registerConceptTrace}=require('../concept-trace');
const html=conceptTraceHtml({cspSource:'test-source'});
const script=html.match(/<script nonce="[^"]+">([\s\S]*?)<\/script>/)[1];

test('Concept Trace has nonce CSP, explicit target selection, shared cosine scale, and a yellow icon',()=>{
  new vm.Script(script);
  assert.match(html,/script-src 'nonce-/);
  assert.match(html,/Shared scale: −1 … 0 … \+1/);
  assert.match(html,/Missing fitted map/);
  assert.match(html,/Stop is exclusive/);
  assert.match(html,/1024/);
  assert.doesNotMatch(html,/<input id="target"[^>]*value=/);
  assert.doesNotMatch(script,/innerHTML|eval\(/);
  assert.match(fs.readFileSync(path.join(__dirname,'../assets/concept-trace.svg'),'utf8'),/#ffcc33/);
  const manifest=JSON.parse(fs.readFileSync(path.join(__dirname,'../package.json'),'utf8'));
  assert.ok(manifest.files.includes('concept-trace.js'));
  assert.ok(manifest.contributes.commands.some(c=>c.command==='llmdbg.showConceptTrace'));
});

function browser(){
  const sent=[],events={},elements={};
  function element(){return {style:{},children:[],value:'',hidden:false,disabled:false,clientWidth:800,clientHeight:400,scrollLeft:0,scrollTop:0,attrs:{},
    append(...items){this.children.push(...items);},replaceChildren(...items){this.children=items;},setAttribute(k,v){this.attrs[k]=v;},
    querySelector(){return this.body||(this.body=element());},getBoundingClientRect(){return {left:0,top:0};}};}
  for(const id of ['colorScale','scaleLegend','lattice','viewport','surface','status','run','zoom','threshold','thresholdValue','context','generation','generatedText','hover','inspector','selection','readout','edges','edgeTable','edgeNote','edgeCount','form','target','piece','pieceLabel','start','stop','lens'])elements[id]=element();
  elements.colorScale.value='1';
  for(const id of ['patchPreview','patchImage','patchRegion','patchCaption'])elements[id]=element();
  elements.candidates=element();
  elements.colorScaleValue=element();
  elements.zoom.value='22';elements.threshold.value='.05';elements.edgeCount.value='12';elements.start.value='0';
  elements.lattice.getContext=()=>new Proxy({},{get:(o,k)=>k in o?o[k]:()=>{},set:(o,k,v)=>(o[k]=v,true)});
  vm.runInNewContext(script,{document:{getElementById:id=>elements[id],createElement:element,body:{}},
    setTimeout,clearTimeout,window:{addEventListener:(type,fn)=>events[type]=fn,devicePixelRatio:1},
    getComputedStyle:()=>({getPropertyValue:key=>key.includes('background')?'#1e1e1e':'#ccc'}),
    acquireVsCodeApi:()=>({postMessage:m=>sent.push(m)})});
  function reply(message,result,error){events.message({data:{type:'conceptReply',requestId:message.requestId,result,error}});}
  return {sent,events,elements,reply};
}
const tick=()=>new Promise(resolve=>setImmediate(resolve));

test('typing suggests exact vocabulary IDs and preserves leading spaces on selection',async()=>{
  const b=browser(),e=b.elements;b.sent.shift();
  e.target.value='fly';e.target.oninput();
  await new Promise(resolve=>setTimeout(resolve,220));
  const query=b.sent.shift();assert.equal(query.action,'candidates');
  b.reply(query,{tokens:[{id:42,text:' fly'},{id:43,text:'fly'}]});await tick();
  assert.equal(e.candidates.children[0].textContent,'#42 "·fly"');
  e.candidates.children[0].onclick();assert.equal(e.target.value,' fly');
  const submit=e.form.onsubmit({preventDefault(){}});
  const run=b.sent.shift();assert.equal(run.action,'run');assert.equal(run.tokenId,42);
  b.reply(run,null,'test stopped');await submit;
});

test('browser replays exact token text, expands heads, inspects safely, and requests target edges on demand',async()=>{
  const b=browser(),e=b.elements;
  b.reply(b.sent.shift(),{model:'test/model',tokenCount:4,promptTokenCount:4,maxNewTokens:20,generationLimit:24,currentToken:3,lenses:[{id:'logit',method:'logit',label:'Logit Lens'},{id:'lens:j',method:'j',label:'J-Lens'}]});await tick();
  assert.equal(e.lens.value,'lens:j');
  assert.equal(e.stop.placeholder,'24');assert.match(e.context.textContent,/max_new_tokens=20/);
  e.target.value=' cat';
  const submitted=e.form.onsubmit({preventDefault(){}});
  const tokenize=b.sent.shift();assert.equal(tokenize.action,'tokenize');assert.equal(tokenize.target,' cat');
  b.reply(tokenize,{tokens:[{id:3,text:' cat'}]});await tick();
  const run=b.sent.shift();assert.equal(run.action,'run');assert.equal(run.tokenId,3);
  const base={snapshotId:'snapshot',model:'test/model',target:' cat',targetId:3,lensLabel:'J-Lens',mode:'lens:j',start:0,stop:4,tokens:['a','b','c','d'],promptTokenCount:3,maxNewTokens:20,generatedTokenCount:1,generatedText:'<script>',stoppedAtEos:true,metric:'cosine',warning:'Saved snapshot',rows:[
    {id:'embed',layer:-1,label:'Embedding',scores:[null,null,null,null],reason:'No map'},
    {id:'post:0',layer:0,label:'Layer 0 · Post',scores:[.1,.2,.3,.4]}]};
  base.images=[{source:'cat.jpg',preview:'data:image/jpeg;base64,AA',startToken:0,endToken:3,gridRows:2,gridColumns:2,regionMapping:'merged-row-major'}];
  b.reply(run,base);await submitted;
  assert.match(e.context.textContent,/#3/);
  assert.match(e.context.textContent,/generated 1 \/ max_new_tokens=20 · stopped at EOS/);
  assert.equal(e.generatedText.textContent,'<script>');assert.equal(e.generation.hidden,false);
  assert.match(e.lattice.attrs['aria-label'],/2 rows/);
  e.colorScale.value='0.2';e.colorScale.oninput();
  assert.equal(e.colorScaleValue.textContent,'0.20');
  assert.match(e.scaleLegend.textContent,/−0.2 … 0 … \+0.2/);
  assert.equal(b.sent.length,0,'changing color scale does not replay');
  // Expanding is a separate request, not part of the initial residual replay.
  e.lattice.onclick({clientX:8,clientY:69});
  const heads=b.sent.shift();assert.equal(heads.action,'heads');assert.equal(heads.layer,0);
  const outputs=[{id:'value:0:0',layer:0,head:0,kind:'value',label:'Head 0 · Value',scores:[.2,.3,.4,.5]},
                 {id:'output:0:0',layer:0,head:0,kind:'output',label:'Head 0 · Output',scores:[.1,.2,.3,.4]}];
  b.reply(heads,{snapshotId:'snapshot',rows:outputs});await tick();
  assert.match(e.lattice.attrs['aria-label'],/4 rows/);
  e.lattice.onclick({clientX:161,clientY:91});
  assert.equal(e.patchPreview.hidden,false);
  assert.equal(e.patchRegion.style.left,'0%');
  assert.equal(e.patchRegion.style.width,'50%');
  assert.match(e.patchCaption.textContent,/Token 0 · row 1 \/ 2 · column 1 \/ 2/);
  const inspect=b.sent.shift();assert.equal(inspect.action,'inspect');assert.equal(inspect.rowId,'output:0:0');
  b.reply(inspect,{norm:2,tokens:[{id:3,text:'<script>',score:2}]});await tick();
  assert.equal(e.readout.children[0].textContent,'<script>');assert.equal(e.edges.disabled,false);
  const requestEdges=e.edges.onclick();const edges=b.sent.shift();assert.equal(edges.action,'edges');assert.equal(edges.head,0);
  b.reply(edges,{snapshotId:'snapshot',sourceRowId:'value:0:0',rows:[{position:0,token:'a',support:.1,magnitude:1}],metric:'signed support',supportSum:.1,normalizationBiasSupport:0,targetScore:.1,reconstructionError:0,warning:'Not causal'});
  await requestEdges;assert.equal(e.edgeTable.hidden,false);assert.match(e.edgeNote.textContent,/Not causal/);
  b.events.message({data:{type:'invalidated',message:'Session ended'}});assert.equal(e.run.disabled,true);
});

test('multi-token input requires an explicit token choice before replay',async()=>{
  const b=browser(),e=b.elements;
  b.reply(b.sent.shift(),{model:'m',tokenCount:4,currentToken:0,lenses:[{id:'logit',method:'logit',label:'Logit Lens'}]});await tick();
  e.target.value='two words';const submit=e.form.onsubmit({preventDefault(){}});
  b.reply(b.sent.shift(),{tokens:[{id:1,text:'two'},{id:2,text:' words'}]});await submit;
  assert.equal(e.pieceLabel.hidden,false);assert.equal(b.sent.length,0);assert.match(e.status.textContent,/Choose one token/);
});

test('module registers independent commands without opening a panel automatically',()=>{
  const commands=new Map(),context={subscriptions:[]};
  const vscode={commands:{registerCommand:(name,fn)=>(commands.set(name,fn),{})},debug:{onDidTerminateDebugSession:()=>({})}};
  registerConceptTrace(context,vscode);
  assert.ok(commands.has('llmdbg.showConceptTrace'));assert.ok(commands.has('llmdbg.closeConceptTrace'));
});
