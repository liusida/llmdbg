"use strict";
const test=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs'),path=require('node:path');
const {conceptTraceHtml,registerConceptTrace}=require('../concept-trace');
const html=conceptTraceHtml({cspSource:'test-source'});
const script=html.match(/<script nonce="[^"]+">([\s\S]*?)<\/script>/)[1];

test('Concept Trace has nonce CSP, explicit target selection, shared cosine scale, and a yellow icon',()=>{
  new vm.Script(script);
  assert.match(html,/script-src 'nonce-/);
  assert.match(html,/Shared scale: −0\.5 … 0 … \+0\.5/);
  assert.match(html,/id="colorScale"[^>]*value="0\.5"/);
  assert.match(html,/<output id="colorScaleValue">0\.50<\/output>/);
  assert.match(html,/Missing fitted map/);
  assert.doesNotMatch(html,/<input id="(?:start|stop)"/);
  assert.match(html,/class="sliders"/);
  assert.match(html,/class="info-popover" role="tooltip"><div id="context">/);
  assert.match(html,/id="targetField"><input id="target"[^>]+><div id="candidates"/);
  assert.match(html,/#candidates\{[^}]*z-index:1000/);
  assert.doesNotMatch(html,/Embedding at top|id="hover"/);
  assert.match(html,/#readout\{[^}]*height:42px;[^}]*flex-wrap:nowrap;[^}]*overflow-x:auto/);
  assert.match(html,/#selection\{height:1\.6em;white-space:nowrap/);
  assert.match(html,/id="threshold"[^>]*value="0\.00"/);
  assert.match(html,/<output id="thresholdValue">0\.00<\/output>/);
  assert.match(html,/id="zoom" type="range" min="4" max="24" step="1" value="12"/);
  assert.doesNotMatch(html,/<input id="target"[^>]*value=/);
  assert.doesNotMatch(script,/innerHTML|eval\(/);
  assert.match(fs.readFileSync(path.join(__dirname,'../assets/concept-trace.svg'),'utf8'),/#ffcc33/);
  const manifest=JSON.parse(fs.readFileSync(path.join(__dirname,'../package.json'),'utf8'));
  assert.ok(manifest.files.includes('concept-trace.js'));
  assert.ok(manifest.contributes.commands.some(c=>c.command==='llmdbg.showConceptTrace'));
});

function browser(){
  const sent=[],events={},elements={},painted=[];
  function element(){return {style:{},children:[],value:'',hidden:false,disabled:false,clientWidth:800,clientHeight:400,scrollLeft:0,scrollTop:0,attrs:{},
    focus(){this.focused=true;},
    append(...items){this.children.push(...items);},replaceChildren(...items){this.children=items;},setAttribute(k,v){this.attrs[k]=v;},removeAttribute(k){delete this.attrs[k];},
    querySelector(){return this.body||(this.body=element());},getBoundingClientRect(){return {left:0,top:0};}};}
  for(const id of ['colorScale','scaleLegend','lattice','viewport','surface','status','run','zoom','threshold','thresholdValue','context','generation','generatedText','hover','inspector','selection','readout','edges','edgeTable','edgeNote','edgeCount','form','target','piece','pieceLabel','start','stop','lens'])elements[id]=element();
  elements.colorScale.value='0.5';
  for(const id of ['patchPreview','patchImage','patchRegion','patchCaption','generatedCaption'])elements[id]=element();
  elements.candidates=element();
  elements.targetField=element();
  elements.targetField.contains=node=>node===elements.target||node===elements.candidates||elements.candidates.children.includes(node);
  elements.regenerate=element();
  elements.colorScaleValue=element();
  elements.zoomValue=element();
  elements.zoom.value='22';elements.threshold.value='0';elements.edgeCount.value='12';elements.start.value='0';
  elements.lattice.getContext=()=>new Proxy({fillRect(...args){painted.push({kind:'rect',args,color:this.fillStyle});},
    fillText(...args){painted.push({kind:'text',args});}},{get:(o,k)=>k in o?o[k]:()=>{},set:(o,k,v)=>(o[k]=v,true)});
  vm.runInNewContext(script,{document:{getElementById:id=>elements[id],createElement:element,body:{}},
    setTimeout,clearTimeout,window:{addEventListener:(type,fn)=>events[type]=fn,devicePixelRatio:1},
    getComputedStyle:()=>({getPropertyValue:key=>key.includes('background')?'#1e1e1e':'#ccc'}),
    acquireVsCodeApi:()=>({postMessage:m=>sent.push(m)})});
  function reply(message,result,error){events.message({data:{type:'conceptReply',requestId:message.requestId,result,error}});}
  return {sent,events,elements,reply,painted};
}
const tick=()=>new Promise(resolve=>setImmediate(resolve));

test('typing suggests exact vocabulary IDs and preserves leading spaces on selection',async()=>{
  const b=browser(),e=b.elements;b.sent.shift();
  e.target.value='fly';e.target.oninput();
  await new Promise(resolve=>setTimeout(resolve,220));
  const query=b.sent.shift();assert.equal(query.action,'candidates');
  b.reply(query,{tokens:[{id:42,text:' fly'},{id:43,text:'fly'}]});await tick();
  assert.equal(e.candidates.children[0].textContent,'#42 "·fly"');
  assert.equal(e.target.attrs['aria-expanded'],'true');
  b.events.pointerdown({target:e.target});assert.equal(e.candidates.hidden,false);
  e.candidates.children[0].onclick();assert.equal(e.target.value,' fly');
  assert.equal(e.target.attrs['aria-expanded'],'false');assert.equal(e.candidates.hidden,true);
  const submit=e.form.onsubmit({preventDefault(){}});
  const run=b.sent.shift();assert.equal(run.action,'run');assert.equal(run.tokenId,42);
  b.reply(run,null,'test stopped');await submit;
});

test('browser replays exact token text, expands heads, inspects safely, and requests target edges on demand',async()=>{
  const b=browser(),e=b.elements;
  b.reply(b.sent.shift(),{model:'test/model',tokenCount:4,promptTokenCount:4,maxNewTokens:20,generationLimit:24,currentToken:3,lenses:[{id:'logit',method:'logit',label:'Logit Lens'},{id:'lens:j',method:'j',label:'J-Lens'}]});await tick();
  assert.equal(e.lens.value,'lens:j');
  assert.match(e.context.textContent,/max_new_tokens=20/);
  e.start.value='99';e.stop.value='100';
  e.target.value=' cat';
  const submitted=e.form.onsubmit({preventDefault(){}});
  const tokenize=b.sent.shift();assert.equal(tokenize.action,'tokenize');assert.equal(tokenize.target,' cat');
  b.reply(tokenize,{tokens:[{id:3,text:' cat'}]});await tick();
  const run=b.sent.shift();assert.equal(run.action,'run');assert.equal(run.tokenId,3);
  assert.equal(run.start,0);assert.equal(Object.hasOwn(run,'stop'),false);
  const base={snapshotId:'snapshot',model:'test/model',target:' cat',targetId:3,lensLabel:'J-Lens',mode:'lens:j',start:0,stop:4,tokens:['a','b','c','d'],promptTokenCount:3,maxNewTokens:20,generatedTokenCount:1,generatedText:'<script>',generatedTokenStart:3,generatedTokens:['<script>'],generationCache:'disk',reusedGeneratedTokens:1,persistentCacheEnabled:true,stoppedAtEos:true,metric:'cosine',warning:'Saved snapshot',rows:[
    {id:'embed',layer:-1,label:'Embedding',scores:[null,null,null,null],reason:'No map'},
    {id:'post:0',layer:0,label:'Layer 0 · Post',scores:[.1,.2,.3,.4]}]};
  base.images=[{source:'cat.jpg',preview:'data:image/jpeg;base64,AA',startToken:0,endToken:3,gridRows:2,gridColumns:2,regionMapping:'merged-row-major'}];
  b.reply(run,base);await submitted;
  assert.equal(e.inspector.hidden,false);assert.equal(e.readout.children.length,0);
  assert.match(e.context.textContent,/#3/);
  assert.match(e.context.textContent,/generated 1 \/ max_new_tokens=20 · stopped at EOS/);
  assert.match(e.context.textContent,/Disk cache: reused 1 generated token IDs/);
  assert.equal(e.status.textContent,'');
  assert.equal(e.generatedText.children[0].textContent,'<script>');assert.equal(e.generation.hidden,false);
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

test('Regenerate explicitly bypasses caches and disables both controls while busy',async()=>{
  const b=browser(),e=b.elements;b.sent.shift();e.target.value=' sky';
  e.regenerate.onclick();
  assert.equal(e.run.disabled,true);assert.equal(e.regenerate.disabled,true);
  const confirm=b.sent.shift();assert.equal(confirm.action,'confirmRegenerate');
  assert.equal(b.sent.length,0);
  b.reply(confirm,true);await tick();
  b.reply(b.sent.shift(),{tokens:[{id:3,text:' sky'}]});await tick();
  const run=b.sent.shift();assert.equal(run.action,'run');assert.equal(run.regenerate,true);
  b.reply(run,null,'test replay error');await tick();
  assert.equal(e.run.disabled,false);assert.equal(e.regenerate.disabled,false);
});

test('cancelling Regenerate does not tokenize, replay, or bypass caches',async()=>{
  const b=browser(),e=b.elements;b.sent.shift();e.target.value=' sky';
  const clicked=e.regenerate.onclick();
  const confirm=b.sent.shift();assert.equal(confirm.action,'confirmRegenerate');
  b.reply(confirm,false);await clicked;
  assert.equal(b.sent.length,0);assert.equal(e.run.disabled,false);assert.equal(e.regenerate.disabled,false);
  const replay=e.form.onsubmit({preventDefault(){}});
  b.reply(b.sent.shift(),{tokens:[{id:3,text:' sky'}]});await tick();
  const run=b.sent.shift();assert.equal(run.regenerate,undefined);
  b.reply(run,null,'test stopped');await replay;
});

test('generated text is a visible preview with exact token highlighting and range offsets',async()=>{
  assert.match(html,/<section id="generation"/);
  assert.match(html,/All prompt and generated tokens/);
  assert.doesNotMatch(html,/<details id="generation"/);
  const b=browser(),e=b.elements;b.sent.shift();e.target.value=' sky';
  const submitted=e.form.onsubmit({preventDefault(){}});
  b.reply(b.sent.shift(),{tokens:[{id:3,text:' sky'}]});await tick();
  const run=b.sent.shift();
  b.reply(run,{snapshotId:'text',model:'test/model',target:' sky',targetId:3,lensLabel:'Logit Lens',mode:'logit',
    start:4,stop:7,tokens:['hello','\n','<script>'],promptTokenCount:3,maxNewTokens:10,generatedTokenCount:4,
    generatedTokenStart:3,generatedTokens:['earlier ','hello','\n','<script>'],generatedText:'earlier hello\n<script>',
    generatedTokenHints:[null,null,null,'Tokens 3–6 decoded together: 🏀'],
    rows:[{id:'post:0',layer:0,label:'Layer 0',scores:[.1,.2,.3]}],images:[],metric:'cosine',warning:'Saved'});
  await submitted;
  const spans=e.generatedText.children;
  assert.equal(e.generation.hidden,false);
  assert.equal(spans.map(s=>s.textContent).join(''),'earlier hello\n<script>');
  assert.equal(spans[0].attrs['data-token-position'],'3');
  assert.match(spans[3].title,/decoded together: 🏀/);
  e.lattice.onclick({clientX:161,clientY:47});
  assert.equal(e.lattice.focused,true);
  assert.equal(spans[1].className,'generated-token selected');
  assert.equal(e.generatedCaption.textContent,'Token 4 · "hello"');
  b.reply(b.sent.shift(),{norm:1,tokens:[]});await tick();
  spans[3].offsetTop=1000;spans[3].offsetHeight=20;
  e.lattice.onclick({clientX:205,clientY:47});
  assert.equal(spans[1].className,'generated-token');
  assert.equal(spans[3].className,'generated-token selected');
  assert.equal(e.generatedText.scrollTop,800);
  assert.equal(e.generatedCaption.textContent,'Token 6 · "<script>"');
  b.reply(b.sent.shift(),{norm:1,tokens:[]});await tick();
  e.lattice.onkeydown({key:'ArrowLeft',preventDefault(){}});
  assert.equal(spans[3].className,'generated-token');
  assert.equal(spans[2].className,'generated-token selected');
  assert.equal(e.generatedCaption.textContent,'Token 5 · "\\n"');
  b.reply(b.sent.shift(),{norm:1,tokens:[]});await tick();
  spans[3].onclick();
  const prediction=b.sent.shift();
  assert.equal(prediction.action,'inspect');
  assert.equal(prediction.position,5);
  assert.equal(prediction.rowId,'post:0');
  assert.match(e.selection.textContent,/predicts token 6/);
  assert.equal(spans[3].className,'generated-token selected');
  assert.match(e.generatedCaption.textContent,/Predicted from token 5/);
  b.reply(prediction,{norm:1,tokens:[]});await tick();
  assert.match(e.selection.textContent,/predicts token 6/);
  spans[0].onclick();
  assert.match(e.status.textContent,/outside this snapshot/);
  assert.equal(b.sent.length,0);
});

test('lattice wraps at panel width and hit testing follows resize and band boundaries',async()=>{
  const b=browser(),e=b.elements;b.sent.shift();e.target.value=' cat';
  const submitted=e.form.onsubmit({preventDefault(){}});
  b.reply(b.sent.shift(),{tokens:[{id:3,text:' cat'}]});await tick();
  b.reply(b.sent.shift(),{snapshotId:'wrapped',model:'test/model',target:' cat',targetId:3,lensLabel:'Logit Lens',mode:'logit',start:0,stop:65,
    tokens:Array.from({length:65},(_,i)=>String(i)),rows:[{id:'post:0',layer:0,label:'Layer 0',scores:Array(65).fill(.1)}],images:[],metric:'cosine',warning:'Saved'});
  await submitted;
  assert.equal(e.surface.style.width,'100%');
  // 800px panel: floor((800 - 150 - 12) / 22) = 29 columns per band.
  assert.equal(e.surface.style.height,'228px');
  e.lattice.onclick({clientX:161,clientY:123});
  let inspect=b.sent.shift();assert.equal(inspect.position,29);
  b.reply(inspect,{norm:1,tokens:[]});await tick();
  e.lattice.onkeydown({key:'ArrowLeft',preventDefault(){}});
  inspect=b.sent.shift();assert.equal(inspect.position,28);
  b.reply(inspect,{norm:1,tokens:[]});await tick();
  e.viewport.clientWidth=580;b.events.resize();
  assert.equal(e.surface.style.height,'304px');
  e.lattice.onclick({clientX:161,clientY:123});
  inspect=b.sent.shift();assert.equal(inspect.position,19);
  b.reply(inspect,{norm:1,tokens:[]});await tick();
  e.viewport.scrollTop=76;
  e.lattice.onclick({clientX:161,clientY:47});
  inspect=b.sent.shift();assert.equal(inspect.position,19);
  b.reply(inspect,{norm:1,tokens:[]});await tick();
  e.zoom.value='12';e.zoom.oninput();
  assert.equal(e.zoomValue.textContent,'12px');
  assert.equal(e.surface.style.height,'132px');
});

test('arrow keys move between tokens and residual layers after selection',async()=>{
  const b=browser(),e=b.elements;b.sent.shift();e.target.value=' cat';
  const submitted=e.form.onsubmit({preventDefault(){}});
  b.reply(b.sent.shift(),{tokens:[{id:3,text:' cat'}]});await tick();
  b.reply(b.sent.shift(),{snapshotId:'keys',model:'test/model',target:' cat',targetId:3,lensLabel:'Logit Lens',mode:'logit',start:0,stop:2,tokens:['a','b'],
    rows:[{id:'embed',layer:-1,label:'Embedding',scores:[.1,.2]}, {id:'post:0',layer:0,label:'Layer 0',scores:[.1,.2]}],images:[],metric:'cosine',warning:'Saved'});
  await submitted;
  e.lattice.onclick({clientX:161,clientY:69});
  b.reply(b.sent.shift(),{norm:1,tokens:[]});await tick();
  for(const [key,rowId,position] of [['ArrowUp','embed',0],['ArrowRight','embed',1],['ArrowDown','post:0',1],['ArrowLeft','post:0',0]]){
    e.lattice.onkeydown({key,preventDefault(){}});
    const inspect=b.sent.shift();assert.equal(inspect.rowId,rowId);assert.equal(inspect.position,position);
    b.reply(inspect,{norm:1,tokens:[]});await tick();
  }
  assert.equal(e.lattice.focused,true);
});

test('PageUp/PageDown jump to first/last layer and Home/End to snapshot token boundaries',async()=>{
  const b=browser(),e=b.elements;b.sent.shift();e.target.value=' cat';
  const submitted=e.form.onsubmit({preventDefault(){}});
  b.reply(b.sent.shift(),{tokens:[{id:3,text:' cat'}]});await tick();
  b.reply(b.sent.shift(),{snapshotId:'jumps',model:'test/model',target:' cat',targetId:3,lensLabel:'Logit Lens',mode:'logit',start:5,stop:8,tokens:['a','b','c'],
    rows:[{id:'embed',layer:-1,label:'Embedding',scores:[.1,.2,.3]},
      ...[0,1,2].map(layer=>({id:'post:'+layer,layer,label:'Layer '+layer,scores:[.1,.2,.3]}))],
    images:[],metric:'cosine',warning:'Saved'});
  await submitted;
  for(const [key,rowId,position] of [['PageDown','post:2',5],['End','post:2',7],
    ['PageUp','post:0',7],['Home','post:0',5],['ArrowUp','embed',5],
    ['PageUp','post:0',5],['PageDown','post:2',5],['End','post:2',7],['End','post:2',7]]){
    let prevented=false;
    e.lattice.onkeydown({key,preventDefault(){prevented=true;}});
    assert.equal(prevented,true);
    const inspect=b.sent.shift();assert.equal(inspect.rowId,rowId);assert.equal(inspect.position,position);
    b.reply(inspect,{norm:1,tokens:[]});await tick();
  }
  assert.equal(e.lattice.focused,true);
});

test('persistent token strip includes prompt, multiple images, and generation with synchronized selection',async()=>{
  const b=browser(),e=b.elements;b.sent.shift();e.target.value=' cat';
  const submitted=e.form.onsubmit({preventDefault(){}});
  b.reply(b.sent.shift(),{tokens:[{id:3,text:' cat'}]});await tick();
  b.reply(b.sent.shift(),{snapshotId:'strip',model:'test/model',target:' cat',targetId:3,lensLabel:'Logit Lens',mode:'logit',start:2,stop:7,tokens:['<image>','ask','<image>','answer','!'],
    promptTokenCount:5,promptTokens:['system','<image>','<image>','ask','<image>'],generatedTokenStart:5,generatedTokens:['answer','!'],
    rows:[{id:'post:0',layer:0,label:'Layer 0',scores:[.1,.2,.3,.4,.5]}],
    images:[{startToken:1,endToken:2,gridRows:1,gridColumns:2,regionMapping:'merged-row-major',source:'one.png'},
            {startToken:4,endToken:4,gridRows:1,gridColumns:1,regionMapping:'merged-row-major',source:'two.png'}],metric:'cosine',warning:'Saved'});
  await submitted;
  function all(node){return [node,...node.children.flatMap(all)];}
  const nodes=all(e.generatedText),tokens=nodes.filter(n=>n.attrs['data-token-position']!==undefined);
  assert.deepEqual(tokens.map(n=>Number(n.attrs['data-token-position'])),[0,1,2,3,4,5,6]);
  const cards=nodes.filter(n=>n.className==='token-image');assert.equal(cards.length,2);
  const details=cards[0].children[1],region=cards[0].children[0].children[1];
  assert.equal(details.open,undefined);
  e.lattice.onclick({clientX:161,clientY:47});
  assert.equal(details.open,true);assert.equal(region.hidden,false);assert.equal(region.style.left,'50%');
  assert.equal(tokens[2].className,'generated-token selected');
  b.reply(b.sent.shift(),{norm:1,tokens:[]});await tick();
  e.lattice.onkeydown({key:'ArrowRight',preventDefault(){}});
  assert.equal(region.hidden,true);assert.equal(cards[0].className,'token-image');
  assert.equal(tokens[3].className,'generated-token selected');assert.equal(e.generation.hidden,false);
  b.reply(b.sent.shift(),{norm:1,tokens:[]});await tick();
  tokens[6].onclick();let inspect=b.sent.shift();assert.equal(inspect.position,5);
  assert.equal(tokens[6].className,'generated-token selected');
  b.reply(inspect,{norm:1,tokens:[]});await tick();
  tokens[3].onclick();inspect=b.sent.shift();assert.equal(inspect.position,3);
  b.reply(inspect,{norm:1,tokens:[]});await tick();
  tokens[0].onclick();assert.equal(b.sent.length,0);assert.match(e.status.textContent,/outside this snapshot/);
});

test('panel reconnects in place, waits for model readiness, and ignores old-session replies',async()=>{
  const commands=new Map(),handlers={},messages=[],calls=[];
  let receiver,created=0,disposed=0,revealed=0,resolveOld;
  const metadata={model:'new/model',tokenCount:2,currentToken:0,lenses:[{id:'logit',method:'logit',label:'Logit Lens'}]};
  const old={id:'old',type:'llmdbg',customRequest:()=>new Promise(resolve=>resolveOld=resolve)};
  let ready=false;
  const fresh={id:'new',type:'llmdbg',customRequest:async(command,args)=>{calls.push(args);if(!ready)throw new Error('not loaded');return metadata;}};
  const panel={webview:{cspSource:'test',postMessage:async m=>messages.push(m),onDidReceiveMessage:fn=>(receiver=fn,{})},
    onDidDispose:()=>({}),reveal:()=>revealed++,dispose:()=>disposed++};
  const vscode={ViewColumn:{Beside:2},commands:{registerCommand:(name,fn)=>(commands.set(name,fn),{}),executeCommand:()=>{}},
    window:{createWebviewPanel:()=>(created++,panel)},debug:{activeDebugSession:old,
      onDidStartDebugSession:fn=>(handlers.start=fn,{}),onDidChangeActiveDebugSession:fn=>(handlers.active=fn,{}),
      onDidReceiveDebugSessionCustomEvent:fn=>(handlers.custom=fn,{}),onDidTerminateDebugSession:fn=>(handlers.end=fn,{})}};
  registerConceptTrace({subscriptions:[]},vscode);commands.get('llmdbg.showConceptTrace')();
  const pending=receiver({type:'conceptTrace',requestId:99,action:'inspect'});
  handlers.end(old);handlers.start(fresh);await tick();
  assert.ok(messages.some(m=>m.type==='sessionChanged'));assert.ok(!messages.some(m=>m.type==='sessionReady'));
  ready=true;handlers.custom({session:fresh,event:'llmdbg/vectorChanged'});await tick();
  assert.equal(messages.find(m=>m.type==='sessionReady').metadata.model,'new/model');
  resolveOld({old:true});await pending;
  assert.ok(!messages.some(m=>m.requestId===99));
  handlers.end(old);assert.equal(messages.at(-1).type,'sessionReady');
  vscode.debug.activeDebugSession=fresh;commands.get('llmdbg.showConceptTrace')();
  assert.equal(created,1);assert.equal(disposed,0);assert.equal(revealed,1);
});

test('browser reconnect preserves controls and enables Replay without inspecting stale snapshots',async()=>{
  const b=browser(),e=b.elements;b.sent.shift();e.target.value=' basketball';e.lens.value='lens:r';e.zoom.value='12';e.colorScale.value='.27';
  b.events.message({data:{type:'invalidated',message:'ended'}});
  b.events.message({data:{type:'sessionChanged'}});
  assert.equal(e.run.disabled,true);
  b.events.message({data:{type:'sessionReady',metadata:{model:'new/model',tokenCount:2,currentToken:0,lenses:[{id:'lens:r',method:'r',label:'R-Lens'},{id:'lens:j',method:'j',label:'J-Lens'}]}}});
  assert.equal(e.run.disabled,false);assert.equal(e.regenerate.disabled,false);
  assert.equal(e.target.value,' basketball');assert.equal(e.lens.value,'lens:r');assert.equal(e.zoom.value,'12');assert.equal(e.colorScale.value,'.27');
  assert.match(e.status.textContent,/Connected to the new debug session/);
});

test('turn header backgrounds clip to wrapped token bands and identify turns on hover',async()=>{
  const b=browser(),e=b.elements;b.sent.shift();e.target.value=' cat';
  const submitted=e.form.onsubmit({preventDefault(){}});
  b.reply(b.sent.shift(),{tokens:[{id:3,text:' cat'}]});await tick();
  b.reply(b.sent.shift(),{snapshotId:'turns',model:'m',target:' cat',targetId:3,lensLabel:'Logit Lens',mode:'logit',
    start:10,stop:45,tokens:Array(35).fill('x'),rows:[{id:'post:0',layer:0,label:'Layer 0',scores:Array(35).fill(.1)}],
    turns:[{start:0,end:15,role:'user'},{start:15,end:45,role:'assistant'}],images:[],metric:'cosine',warning:'Saved'});
  await submitted;
  const rects=b.painted.filter(p=>p.kind==='rect');
  assert.ok(rects.some(p=>p.color==='rgba(106,164,218,.20)'&&p.args.join(',')==='150,0,110,36'));
  assert.ok(rects.some(p=>p.color==='rgba(174,137,218,.20)'&&p.args.join(',')==='260,0,528,36'));
  assert.ok(rects.some(p=>p.color==='rgba(174,137,218,.20)'&&p.args.join(',')==='150,76,132,36'));
  assert.ok(b.painted.some(p=>p.kind==='text'&&p.args[0]==='User'));
  assert.ok(b.painted.some(p=>p.kind==='text'&&p.args[0]==='Assistant'));
  e.lattice.onmousemove({clientX:161,clientY:12});assert.match(e.lattice.title,/user turn · tokens 0–14/);
  e.lattice.onmousemove({clientX:300,clientY:12});assert.match(e.lattice.title,/assistant turn · tokens 15–44/);
});

test('partial generation renders the lattice and token strip with a nonfatal notice',async()=>{
  const b=browser(),e=b.elements;b.sent.shift();e.target.value=' cat';
  const submitted=e.form.onsubmit({preventDefault(){}});
  b.reply(b.sent.shift(),{tokens:[{id:3,text:' cat'}]});await tick();
  b.reply(b.sent.shift(),{snapshotId:'partial',model:'test/model',target:' cat',targetId:3,lensLabel:'Logit Lens',mode:'logit',
    start:0,stop:3,tokens:['question','prefix','word'],promptTokens:['question','prefix'],promptTokenCount:2,
    maxNewTokens:1,generatedTokenCount:1,generatedTokenStart:2,generatedTokens:['word'],
    incomplete:true,incompleteReason:'Partial AUTO_REPLY: reached max_new_tokens=1 before EOS. Later turns have not run.',
    rows:[{id:'post:0',layer:0,label:'Layer 0',scores:[.1,.2,.3]}],images:[],metric:'cosine',warning:'Saved'});
  await submitted;
  assert.equal(e.status.className,'partial');assert.match(e.status.textContent,/Partial AUTO_REPLY/);
  assert.equal(e.generation.hidden,false);assert.equal(e.run.disabled,false);
  assert.match(e.lattice.attrs['aria-label'],/positions 0 to 2/);
  e.lattice.onclick({clientX:205,clientY:47});
  const inspect=b.sent.shift();assert.equal(inspect.action,'inspect');assert.equal(inspect.position,2);
  b.reply(inspect,{norm:1,tokens:[]});await tick();
  assert.equal(e.inspector.hidden,false);
});

test('module registers independent commands without opening a panel automatically',()=>{
  const commands=new Map(),context={subscriptions:[]};
  const vscode={commands:{registerCommand:(name,fn)=>(commands.set(name,fn),{})},debug:{onDidTerminateDebugSession:()=>({}),onDidStartDebugSession:()=>({}),onDidChangeActiveDebugSession:()=>({}),onDidReceiveDebugSessionCustomEvent:()=>({})}};
  registerConceptTrace(context,vscode);
  assert.ok(commands.has('llmdbg.showConceptTrace'));assert.ok(commands.has('llmdbg.closeConceptTrace'));
});
