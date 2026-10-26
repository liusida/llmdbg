"use strict";
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const source=fs.readFileSync(path.join(__dirname,'../extension.js'),'utf8');
const context={};
vm.runInNewContext(source.slice(source.indexOf('function escapeHtml('),source.indexOf('const LENS_READOUT_CSS =')),context);
test('memory contributions reuse token UI without presenting attention weights or raw values',()=>{
  const data={kind:'delta-memory',head:12,headDimension:128,transformerLayer:17,tokenPosition:1,token:'cat',
    rows:[{position:0,token:'<script>',contribution:.8,magnitude:1,beta:.2,decay:.9},
          {position:1,token:'cat',contribution:-.1,magnitude:.2,beta:.5,decay:.8}],
    sum:.7,positiveSum:.8,negativeSum:-.1,outputNorm:.7,reconstructionError:.0001,
    images:[{startToken:0,endToken:1,gridRows:1,gridColumns:2,gridFrames:1,regionMapping:'merged-row-major',source:'cat.jpg'}]};
  const html=context.attentionTokenContributionsHtml({cspSource:'test'},data);
  assert.match(html,/Head 12 · DeltaNet Memory Contributions/);
  assert.match(html,/saved snapshot/);
  assert.match(html,/not an attention weight or causal deletion effect/);
  assert.match(html,/held fixed/);
  assert.match(html,/Signed support/);
  assert.match(html,/Projected contribution magnitude/);
  assert.doesNotMatch(html,/<th>Weight<\/th>|Weight × value|Value magnitude|Actual mixed\[|undefined|NaN/);
  assert.match(html,/&lt;script&gt;/);
  assert.match(html,/id="filter"/);
  assert.match(html,/id="order"/);
  assert.match(html,/class="image-region"/);
  assert.match(html,/data-position="0"/);
  for(const script of html.matchAll(/<script nonce="[^"]+">([\s\S]*?)<\/script>/g))new vm.Script(script[1]);
});
test('full-attention contribution labels remain unchanged',()=>{
  const html=context.attentionTokenContributionsHtml({cspSource:'test'},{channel:5,queryHead:0,kvHead:0,dimension:5,
    headDimension:128,tokenPosition:0,token:'x',rows:[{position:0,token:'x',weight:.5,value:2,contribution:1}],sum:1,
    positiveSum:1,negativeSum:0,mixedValue:1,transformerLayer:3,images:[]});
  assert.match(html,/Channel 5 · Token contributions/);
  assert.match(html,/<th>Weight<\/th>/);
  assert.match(html,/Σ weight × value/);
});
