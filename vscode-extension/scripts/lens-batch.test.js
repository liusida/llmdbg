"use strict";
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const source=fs.readFileSync(path.join(__dirname,'../extension.js'),'utf8');
const views=['residual-pre','attention-head-values','attention-heads','attention-output','residual-mid','neurons','mlp-output','residual-post'];
function harness(){
  const calls=[],posts=[];
  const active={async customRequest(command,args){calls.push({command,args});
    if(command==='llmdbg/icaSupport')return {supported:false};
    if(command==='llmdbg/residualToVocabularyLensSupport')return {lenses:[{id:'r',label:'R-Lens',method:'r'}]};
    assert.equal(command,'llmdbg/neuronLensBatch');
    return {items:args.requests.map(({mode,view})=>({mode,view,item:{view,label:view}}))};
  }};
  const context={neuronLensPanel:{webview:{async postMessage(value){posts.push(value)}}},neuronLensRevision:1,
    neuronLensRefreshRequest:0,neuronLensExecutionReady:true,neuronLensModes:['lens:r'],neuronLensViews:views,
    neuronLensItemCache:new Map(),neuronLensIcaInitialized:false,
    pinnedCosineReference:undefined,pinnedTokenReference:undefined,pinnedPatchVector:undefined,
    vscode:{debug:{activeDebugSession:active},workspace:{getConfiguration:()=>({get:()=> 'dark'})}},
    mergeLensSnapshots:entries=>entries.find(e=>e.item)?.item};
  active.type='llmdbg';
  vm.runInNewContext(source.slice(source.indexOf('  async function refreshNeuronLens('),source.indexOf('  async function showAttentionTokenContributions(')),context);
  return {context,calls,posts,active,run:()=>context.refreshNeuronLens()};
}
test('one panel refresh batches all eight views and reuses per-section cache',async()=>{
  const h=harness();await h.run();
  const batches=h.calls.filter(c=>c.command==='llmdbg/neuronLensBatch');
  assert.equal(batches.length,1);assert.equal(batches[0].args.requests.length,8);
  assert.deepEqual(Array.from(batches[0].args.requests,r=>r.view),views);
  assert.equal(h.posts.at(-1).items.length,8);
  await h.run();assert.equal(h.calls.filter(c=>c.command==='llmdbg/neuronLensBatch').length,1);
  await h.context.refreshNeuronLens(['lens:r'],views,true);
  assert.equal(h.calls.filter(c=>c.command==='llmdbg/neuronLensBatch').length,2);
});
test('a failed section does not suppress healthy batched sections',async()=>{
  const h=harness(),original=h.active.customRequest;
  h.active.customRequest=async(command,args)=>{const result=await original(command,args);
    if(command==='llmdbg/neuronLensBatch')result.items[7]={mode:'lens:r',view:'residual-post',error:'failed section'};
    return result;};
  await h.run();const items=h.posts.at(-1).items;
  assert.equal(items.length,8);assert.equal(items[0].view,'residual-pre');assert.equal(items[7].error,'failed section');
});
test('old-state batch replies are not cached or displayed after a debugger step',async()=>{
  const h=harness(),original=h.active.customRequest;
  h.active.customRequest=async(command,args)=>{const result=await original(command,args);
    if(command==='llmdbg/neuronLensBatch')h.context.neuronLensRevision++;
    return result;};
  await h.run();assert.equal(h.context.neuronLensItemCache.size,0);assert.equal(h.posts.filter(p=>p.type==='lens').length,0);
});
test('a newer refresh supersedes older same-state requests',async()=>{
  const h=harness(),original=h.active.customRequest;let release,started;
  const waiting=new Promise(resolve=>started=resolve);
  let count=0;
  h.active.customRequest=async(command,args)=>{const result=await original(command,args);
    if(command==='llmdbg/neuronLensBatch'&&++count===1){started();await new Promise(resolve=>release=resolve);}
    return result;};
  const old=h.run();await waiting;
  await h.context.refreshNeuronLens(['lens:r'],['residual-post']);
  release();await old;
  const displays=h.posts.filter(p=>p.type==='lens');assert.equal(displays.length,1);
  assert.equal(displays[0].items.length,1);assert.equal(displays[0].items[0].view,'residual-post');
});
