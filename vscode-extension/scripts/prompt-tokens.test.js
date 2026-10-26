"use strict";
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const source=fs.readFileSync(path.join(__dirname,'../extension.js'),'utf8');
const context={escapeHtml:value=>String(value).replaceAll('<','&lt;').replaceAll('>','&gt;')};
vm.runInNewContext(source.slice(source.indexOf('function promptTokensHtml('),source.indexOf('async function promptImagePreviews(')),context);
const image={source:'cat.jpg',startToken:1,endToken:4,tokenCount:4,regionMapping:'merged-row-major',gridFrames:1,gridRows:2,gridColumns:2};
function page(images=[image]){return context.promptTokensHtml({cspSource:'test'},{currentToken:2,tokens:['hello',...Array(4).fill('<|image_pad|>'),'end'],images,prompt:'hello'});}
test('image token groups start collapsed and expose individual accessible patch tokens',()=>{
  const html=page();
  assert.match(html,/data-group="0" role="button" tabindex="0" aria-expanded="false"/);
  assert.equal((html.match(/data-patch-group="0"/g)||[]).length,4);
  assert.equal((html.match(/Click to locate its image patch" hidden/g)||[]).length,4);
  assert.match(html,/class="patch-image-frame"/);
  new vm.Script(html.match(/<script nonce="[^"]+">([\s\S]*?)<\/script>/)[1]);
});
test('expand/collapse, filtering, and patch highlighting do not move debugger token',()=>{
  const html=page(),script=html.match(/<script nonce="[^"]+">([\s\S]*?)<\/script>/)[1];
  function token(dataset){return {dataset,hidden:dataset.patchGroup!==undefined,attrs:{},classList:{toggle(){}},setAttribute(key,value){this.attrs[key]=value}};}
  const group=token({group:'0',search:'1 4 image'}),patch=token({patchGroup:'0',position:'4',search:'4 image_pad'}),items=[group,patch];
  const region={hidden:true,style:{}},label={textContent:''},card={querySelector:s=>s==='.patch-label'?label:region,scrollIntoView(){}};
  const filter={value:''},listeners={};
  const document={getElementById:()=>filter,querySelectorAll:s=>s==='.token'?items:s==='.patch-region'?[region]:[],querySelector:s=>s==='.tokens'?{addEventListener:(type,fn)=>listeners[type]=fn}:s.startsWith('[data-image-index')?card:null};
  vm.runInNewContext(script,{document});
  const click=item=>listeners.click({target:{closest:()=>item}});
  click(group);assert.equal(patch.hidden,false);assert.equal(group.attrs['aria-expanded'],'true');
  click(patch);assert.equal(region.hidden,false);assert.deepEqual(region.style,{left:'50%',top:'50%',width:'50%',height:'50%'});
  assert.equal(label.textContent,'Token 4 · row 2 / 2 · column 2 / 2');
  click(group);assert.equal(patch.hidden,true);
  filter.value='4';filter.oninput();assert.equal(patch.hidden,false);
  filter.value='';filter.oninput();assert.equal(patch.hidden,true);
  assert.doesNotMatch(script,/postMessage/);
});
test('multiple images preserve original indices even when a span is unavailable',()=>{
  assert.match(page([{source:'unknown.jpg'},image]),/data-patch-group="1"/);
});
