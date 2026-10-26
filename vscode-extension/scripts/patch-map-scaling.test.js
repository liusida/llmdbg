"use strict";
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('fs'),path=require('path'),vm=require('vm');
const source=fs.readFileSync(path.join(__dirname,'../extension.js'),'utf8');
const context={escapeHtml:value=>String(value)};
vm.runInNewContext(source.slice(source.indexOf('function contributionImageOverlaysHtml('),source.indexOf('function attentionImageMapsHtml(')),context);
const image={startToken:25,gridRows:1,gridColumns:2,gridFrames:1,regionMapping:'merged-row-major',source:'cat.jpg'};

test('patch maps default to all sources, with a local image-only switch for each metric',()=>{
  const html=context.contributionImageOverlaysHtml([image],[
    {position:0,contribution:1,value:100},
    {position:25,contribution:.000022,value:2},
    {position:26,contribution:-.000011,value:-1},
  ]);
  const opacities=[...html.matchAll(/--all-source-opacity:([^;]+);--image-only-opacity:([^\"]+)/g)].map(match=>[Number(match[1]),Number(match[2])]);
  for(const [index,expected] of [[0,[.82*.000022,.82]],[1,[.86*.000011,.43]],[2,[.82*.02,.82]],[3,[.86*.01,.43]]]){
    expected.forEach((value,which)=>assert.ok(Math.abs(opacities[index][which]-value)<1e-12));
  }
  assert.equal((html.match(/class="image-only-toggle"/g)||[]).length,2);
  for(const control of html.matchAll(/class="image-only-toggle"[^>]*><input([^>]*)>/g))assert.doesNotMatch(control[1],/checked/);
  assert.match(html,/including text and special tokens/);
  assert.match(html,/fill-opacity:var\(--all-source-opacity\)/);
  assert.match(html,/:has\(\.image-only-toggle input:checked\).*fill-opacity:var\(--image-only-opacity\)/);
  assert.match(html,/:has\(\.original-image-toggle input:not\(:checked\)\).*visibility:hidden/);
});

test('zero and missing patches remain transparent without NaN; other images do not set the local maximum',()=>{
  const empty=context.contributionImageOverlaysHtml([image],[{position:25,contribution:0,value:0}]);
  assert.doesNotMatch(empty,/NaN|Infinity/);
  assert.equal((empty.match(/--all-source-opacity:0;--image-only-opacity:0/g)||[]).length,4);
  const html=context.contributionImageOverlaysHtml([image,{...image,startToken:30}],[{position:25,contribution:1,value:1},{position:30,contribution:10,value:10}]);
  const first=[...html.matchAll(/--all-source-opacity:([^;]+);--image-only-opacity:([^\"]+)/g)][0];
  assert.ok(Math.abs(Number(first[1])-.082)<1e-12);
  assert.equal(Number(first[2]),.82);
});
