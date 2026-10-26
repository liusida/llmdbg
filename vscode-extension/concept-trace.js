"use strict";

const crypto = require('node:crypto');

function conceptTraceHtml(webview) {
  const nonce = crypto.randomBytes(18).toString('base64');
  return `<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data: https:; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';">
<style>
*{box-sizing:border-box}body{margin:0;padding:16px;font:13px/1.45 var(--vscode-font-family);color:var(--vscode-foreground);background:var(--vscode-editor-background)}
h1{font-size:18px;margin:0 0 12px}form,.controls{display:flex;align-items:center;flex-wrap:wrap;gap:8px 14px;margin-bottom:10px}label{display:inline-flex;align-items:center;gap:6px}input,select,button{font:inherit;color:var(--vscode-input-foreground);background:var(--vscode-input-background);border:1px solid var(--vscode-panel-border);padding:5px 8px}input:focus,select:focus,button:focus-visible{outline:1px solid var(--vscode-focusBorder)}button{cursor:pointer}button.primary{background:var(--vscode-button-background);color:var(--vscode-button-foreground)}button:disabled{opacity:.5;cursor:default}#target{width:180px}input[type=number]{width:76px}input[type=range]{width:110px;padding:0;accent-color:#e5b63b}#status,#context,.hint{color:var(--vscode-descriptionForeground)}#status.error{color:var(--vscode-errorForeground)}#context{white-space:pre-wrap;margin:8px 0}.legend{display:flex;flex-wrap:wrap;gap:8px 18px;margin:8px 0;font-size:12px;color:var(--vscode-descriptionForeground)}.positive{color:#e5b63b}.negative{color:#6aa4da}#viewport{position:relative;overflow:auto;height:56vh;min-height:280px;max-height:680px;border:1px solid var(--vscode-panel-border);margin:10px 0;background:var(--vscode-editor-background)}#surface{position:relative;min-width:100%;min-height:100%}canvas{position:sticky;top:0;left:0;display:block;outline:none}canvas:focus-visible{box-shadow:inset 0 0 0 1px var(--vscode-focusBorder)}#hover{white-space:pre-wrap;min-height:36px;color:var(--vscode-descriptionForeground);font-size:12px}#inspector{border:1px solid var(--vscode-panel-border);padding:12px;margin-top:8px}#selection{font-weight:600;white-space:pre-wrap}.readout{display:flex;align-items:center;flex-wrap:wrap;gap:5px;margin:10px 0}.chip{font-family:var(--vscode-editor-font-family);white-space:pre-wrap;padding:3px 7px;border:1px solid rgba(164,111,211,.55);border-radius:3px;background:rgba(55,35,78,.5)}.readout.logit .chip{border-color:rgba(82,148,214,.55);background:rgba(24,54,86,.5)}.readout.r .chip{border-color:rgba(63,164,139,.55);background:rgba(20,64,54,.5)}.readout.jpp .chip{border-color:rgba(222,176,65,.55);background:rgba(76,57,18,.5)}#edgeNote{white-space:pre-wrap;margin:8px 0;font-size:12px;color:var(--vscode-descriptionForeground)}table{border-collapse:collapse;width:100%;font-size:12px}td,th{border-bottom:1px solid var(--vscode-panel-border);padding:4px 7px;text-align:left}td.num{font-variant-numeric:tabular-nums}td button{padding:1px 5px}.hint{font-size:12px}[hidden]{display:none!important}
#viewport{min-height:100px;max-height:none;overflow-x:auto;overflow-y:hidden}
.positive{color:#00ff41}
</style></head><body>
<h1>Concept Trace</h1>
<form id="form"><label>Target <input id="target" placeholder="Exact token text, e.g.  cat" autocomplete="off" spellcheck="false"></label><label>Lens <select id="lens"></select></label><label id="pieceLabel" hidden>Token <select id="piece"></select></label><button id="run" class="primary" type="submit">Replay</button></form>
<div class="controls"><label>Start <input id="start" type="number" min="0" value="0"></label><label>Stop <input id="stop" type="number" min="1" placeholder="all"></label><span class="hint">Stop is exclusive · up to 1024 displayed positions</span><label>Brightness threshold <input id="threshold" type="range" min="0" max="0.95" step="0.01" value="0.05"><output id="thresholdValue">0.05</output></label><label>Zoom <input id="zoom" type="range" min="2" max="40" step="2" value="22"></label></div>
<div id="candidates" hidden aria-label="Candidate vocabulary tokens" style="display:flex;flex-wrap:wrap;gap:5px;margin:8px 0"></div>
<div id="status">Start and pause a model session to prepare a snapshot.</div><div id="context"></div><details id="generation" hidden><summary>Generated continuation</summary><pre id="generatedText"></pre></details>
<div class="controls"><label>Full-brightness cosine ± <input id="colorScale" type="range" min="0.01" max="1" step="0.01" value="1"><output id="colorScaleValue">1.00</output></label><span class="hint">Set 0.2 to saturate colors at |cosine| ≥ 0.2. Applies to all points; no replay needed.</span></div>
<div class="legend"><span class="positive">● Positive cosine</span><span class="negative">● Negative cosine</span><span>○ Missing fitted map</span><span id="scaleLegend">Shared scale: −1 … 0 … +1 · not probabilities</span></div>
<div class="hint">Embedding at top; token positions run left to right. Click a layer label to expand Value/Output heads. Click a point to inspect it. Arrow keys move the selected point.</div>
<div id="viewport"><div id="surface"><canvas id="lattice" tabindex="0" role="img" aria-label="Concept Trace lattice. No snapshot yet."></canvas></div></div><div id="hover"></div>
<section id="inspector" hidden><div id="selection"></div><div id="readout" class="readout"></div><button id="edges" type="button" disabled>Trace target contributions</button><label>Show edges <select id="edgeCount"><option value="12">Top 12</option><option value="30">Top 30</option><option value="10000">All</option></select></label><div id="edgeNote"></div><table id="edgeTable" hidden><thead><tr><th>Source / write position</th><th>Token</th><th>Signed target support</th><th>Projected magnitude</th></tr></thead><tbody></tbody></table></section>
<figure id="patchPreview" hidden style="margin:16px 0"><div style="position:relative;display:inline-block;max-width:100%"><img id="patchImage" alt="Selected image token location" style="display:block;max-width:100%;max-height:65vh"><span id="patchRegion" style="position:absolute;border:2px solid #ffcc33;background:rgba(255,204,51,.22);pointer-events:none"></span></div><figcaption id="patchCaption"></figcaption></figure>
<p class="hint">This is a saved replay snapshot. Lens alignment is not proof of semantic content or causal use. Edges are computed only on request; gates, queries and normalization statistics are held fixed. Image-token positions are spatial anchors, not exclusive pixel origins.</p>
<script nonce="${nonce}">(${browserMain.toString()})();</script></body></html>`;
}

function browserMain() {
  const vscode = acquireVsCodeApi();
  const $ = id => document.getElementById(id);
  const canvas=$('lattice'), viewport=$('viewport'), surface=$('surface'), ctx=canvas.getContext('2d');
  let snapshot=null, expanded=null, headRows=new Map(), displayed=[], selected=null, edgeData=null;
  let requestId=0, pending=new Map(), inspectionVersion=0, encodedTarget=null, busy=false, ended=false;
  const labelWidth=150, headerHeight=36;
  let candidate=null,candidateTimer,candidateVersion=0;
  $('target').oninput=()=>{
    candidate=null;encodedTarget=null;$('pieceLabel').hidden=true;
    const version=++candidateVersion,text=$('target').value;
    clearTimeout(candidateTimer);$('candidates').replaceChildren();$('candidates').hidden=true;
    if(!text||ended)return;
    candidateTimer=setTimeout(async()=>{
      try{
        const result=await request('candidates',{target:text});
        if(version!==candidateVersion)return;
        $('candidates').replaceChildren();$('candidates').hidden=false;
        for(const token of result.tokens){
          const button=document.createElement('button');button.type='button';
          button.textContent='#'+token.id+' '+JSON.stringify(token.text).replace(/ /g,'·');
          button.title='Exact token: '+JSON.stringify(token.text);
          button.onclick=()=>{candidate=token;$('target').value=token.text;++candidateVersion;$('candidates').hidden=true;status('Selected token #'+token.id+' '+JSON.stringify(token.text)+'. Click Replay.');};
          $('candidates').append(button);
        }
        if(!result.tokens.length)$('candidates').textContent='No matching vocabulary tokens.';
      }catch(error){if(version===candidateVersion)status(error.message,true);}
    },180);
  };
  function status(text,error=false){$('status').textContent=text;$('status').className=error?'error':'';}
  function setBusy(value){busy=value;$('run').disabled=value||ended;}
  function request(action,args={}){
    const id=++requestId;
    return new Promise((resolve,reject)=>{pending.set(id,{resolve,reject});vscode.postMessage({type:'conceptTrace',action,...args,requestId:id});});
  }
  function snapshotArgs(){return {snapshotId:snapshot.snapshotId};}
  function pitch(){return Number($('zoom').value);}
  function rows(){
    displayed=[];
    if(!snapshot)return;
    for(const row of snapshot.rows){
      if(row.layer===expanded&&headRows.has(row.layer))displayed.push(...headRows.get(row.layer));
      displayed.push(row);
    }
  }
  function layout(){
    if(snapshot)viewport.style.height=Math.max(100,headerHeight+displayed.length*pitch()+14)+'px';
    // Clear the old canvas footprint before measuring: a newly visible vertical
    // scrollbar must not leave a viewport-wide canvas overflowing horizontally.
    canvas.style.width='0px';canvas.style.height='0px';
    surface.style.width=snapshot?(labelWidth+snapshot.tokens.length*pitch()+12)+'px':'100%';
    surface.style.height=(headerHeight+displayed.length*pitch()+12)+'px';
    if(snapshot){
      const chrome=Math.max(2,(viewport.offsetHeight-viewport.clientHeight)||2);
      viewport.style.height=Math.max(100,headerHeight+displayed.length*pitch()+12+chrome)+'px';
    }
    const width=viewport.clientWidth,height=viewport.clientHeight,dpr=window.devicePixelRatio||1;
    canvas.style.width=width+'px';canvas.style.height=height+'px';canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);
    ctx.setTransform(dpr,0,0,dpr,0,0);
    draw();
  }
  function cell(rowIndex,column){return {x:labelWidth+(column+.5)*pitch()-viewport.scrollLeft,y:headerHeight+(rowIndex+.5)*pitch()-viewport.scrollTop};}
  function draw(){
    const width=viewport.clientWidth,height=viewport.clientHeight,p=pitch();
    const css=getComputedStyle(document.body),bg=css.getPropertyValue('--vscode-editor-background').trim()||'#1e1e1e',fg=css.getPropertyValue('--vscode-foreground').trim()||'#ccc';
    ctx.clearRect(0,0,width,height);ctx.fillStyle=bg;ctx.fillRect(0,0,width,height);
    if(!snapshot)return;
    ctx.font='11px sans-serif';
    const threshold=Number($('threshold').value);
    ctx.save();ctx.beginPath();ctx.rect(labelWidth,headerHeight,width-labelWidth,height-headerHeight);ctx.clip();
    if(Number.isInteger(snapshot.promptTokenCount)&&snapshot.stop>snapshot.promptTokenCount){
      const boundary=labelWidth+(snapshot.promptTokenCount-snapshot.start)*p-viewport.scrollLeft;
      ctx.strokeStyle='rgba(128,128,128,.6)';ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(boundary,headerHeight);ctx.lineTo(boundary,height);ctx.stroke();ctx.setLineDash([]);
    }
    for(let r=Math.max(0,Math.floor(viewport.scrollTop/p)-1);r<displayed.length;r++){
      const row=displayed[r],y=cell(r,0).y;if(y>height)break;
      if(row.layer===expanded){ctx.fillStyle='rgba(128,128,128,.045)';ctx.fillRect(labelWidth,y-p/2,width-labelWidth,p);}
      for(let c=Math.max(0,Math.floor(viewport.scrollLeft/p)-1);c<snapshot.tokens.length;c++){
        const point=cell(r,c);if(point.x>width)break;
        const value=row.scores[c];ctx.beginPath();ctx.arc(point.x,point.y,Math.min(p*.4,Math.max(2,p*.16)),0,Math.PI*2);
        if(value===null){ctx.strokeStyle='rgba(130,130,130,.5)';ctx.stroke();}
        else{
          const scale=Number($('colorScale').value);
          const ceiling=Number.isFinite(scale)&&scale>0?Math.min(1,scale):1;
          const strength=Math.min(1,Math.max(0,(Math.abs(value)-threshold)/Math.max(1e-6,ceiling-threshold)));
          ctx.fillStyle=Math.abs(value)<=threshold?'rgba(128,128,128,.12)':value>=0?'rgba(0,255,65,'+(.08+.92*strength)+')':'rgba(106,164,218,'+(.08+.92*strength)+')';ctx.fill();
        }
      }
    }
    if(edgeData&&selected){
      const r=displayed.findIndex(row=>row.id===selected.row.id),sourceRow=displayed.findIndex(row=>row.id===edgeData.sourceRowId),dest=cell(r,selected.column);
      const edges=[...edgeData.rows].sort((a,b)=>Math.abs(b.support)-Math.abs(a.support)).slice(0,Number($('edgeCount').value));
      const max=Math.max(...edges.map(edge=>Math.abs(edge.support)),1e-30);
      for(const edge of edges){
        const c=edge.position-snapshot.start;if(sourceRow<0||c<0||c>=snapshot.tokens.length)continue;
        const from=cell(sourceRow,c);ctx.beginPath();ctx.strokeStyle=edge.support>=0?'rgba(0,255,65,.75)':'rgba(106,164,218,.75)';ctx.lineWidth=.5+2*Math.abs(edge.support)/max;
        const lift=Math.min(30,Math.max(8,Math.abs(dest.x-from.x)*.1));
        ctx.moveTo(from.x,from.y);ctx.bezierCurveTo(from.x,from.y-lift,dest.x,dest.y-lift,dest.x,dest.y);ctx.stroke();
        ctx.fillStyle=ctx.strokeStyle;ctx.beginPath();ctx.arc(from.x,from.y,3,0,Math.PI*2);ctx.fill();
        ctx.beginPath();ctx.moveTo(dest.x,dest.y-1);ctx.lineTo(dest.x-3,dest.y-7);ctx.lineTo(dest.x+3,dest.y-7);ctx.closePath();ctx.fill();
      }
      ctx.lineWidth=1;
    }
    if(selected){const r=displayed.findIndex(row=>row.id===selected.row.id);if(r>=0){const point=cell(r,selected.column);ctx.strokeStyle=fg;ctx.lineWidth=2;ctx.strokeRect(point.x-p*.35,point.y-p*.35,p*.7,p*.7);ctx.lineWidth=1;}}
    ctx.restore();
    ctx.fillStyle=bg;ctx.fillRect(0,0,labelWidth,height);ctx.fillRect(0,0,width,headerHeight);
    ctx.fillStyle=fg;ctx.font='11px sans-serif';ctx.fillText('Layer / token position →',8,18);
    if(Number.isInteger(snapshot.promptTokenCount)&&snapshot.stop>snapshot.promptTokenCount){
      const boundary=labelWidth+(snapshot.promptTokenCount-snapshot.start)*p-viewport.scrollLeft;
      if(boundary<width)ctx.fillText('Generated',Math.max(labelWidth+4,boundary+4),12);
    }
    const rowFont=Math.min(11,p*.8);
    ctx.font=rowFont+'px sans-serif';
    for(let r=0;r<displayed.length;r++){
      const y=cell(r,0).y;if(y<headerHeight||y>height)continue;
      const row=displayed[r];ctx.fillStyle=fg;ctx.fillText((row.id.startsWith('post:')?(expanded===row.layer?'− ':'+ '):'')+row.label,8,y+rowFont*.35,labelWidth-16);
    }
    ctx.font='11px sans-serif';
    const step=p<20?10:5;
    for(let c=0;c<snapshot.tokens.length;c++){
      const x=cell(0,c).x;if(x<labelWidth||x>width)continue;
      if((snapshot.start+c)%step===0){ctx.fillStyle=fg;ctx.fillText(String(snapshot.start+c),x-7,28);}
    }
    canvas.setAttribute('aria-label','Concept Trace for '+JSON.stringify(snapshot.target)+', '+snapshot.lensLabel+', '+displayed.length+' rows, positions '+snapshot.start+' to '+(snapshot.stop-1)+'.');
  }
  function hit(event){
    const rect=canvas.getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top;
    if(y<headerHeight)return null;
    const r=Math.floor((y+viewport.scrollTop-headerHeight)/pitch());
    const c=Math.floor((x+viewport.scrollLeft-labelWidth)/pitch());
    if(r<0||r>=displayed.length)return null;
    return {row:displayed[r],column:c,label:x<labelWidth};
  }
  function describe(point){const score=point.row.scores[point.column];return point.row.label+' · token '+(snapshot.start+point.column)+' '+JSON.stringify(snapshot.tokens[point.column])+' · '+(score===null?'Missing fitted lens map':'cosine '+score.toFixed(4));}
  async function expand(layer){
    if(busy)return;
    const snapshotId=snapshot.snapshotId;
    if(expanded===layer){expanded=null;rows();layout();return;}
    expanded=layer;
    try{
      status('Computing projected head readouts…');
      if(!headRows.has(layer)){const result=await request('heads',{...snapshotArgs(),layer});if(snapshot?.snapshotId!==snapshotId||result.snapshotId!==snapshotId)return;headRows.set(layer,result.rows);}
      rows();layout();status('Saved snapshot. Click an output-head point to trace source contributions.');
    }catch(error){expanded=null;status(error.message,true);}
  }
  async function select(point){
    if(!snapshot||point.column<0||point.column>=snapshot.tokens.length)return;
    const version=++inspectionVersion;selected=point;edgeData=null;draw();
    const position=snapshot.start+point.column;
    const image=(snapshot.images||[]).find(image=>Number.isInteger(image.startToken)&&position>=image.startToken&&position<=image.endToken);
    $('patchPreview').hidden=!image;
    if(image){
      $('patchImage').src=image.preview||image.source;
      const rows=image.gridRows,columns=image.gridColumns,offset=position-image.startToken;
      const mapped=image.regionMapping==='merged-row-major'&&rows>0&&columns>0;
      $('patchRegion').hidden=!mapped;
      const row=Math.floor(offset/columns)%rows,column=offset%columns;
      if(mapped)Object.assign($('patchRegion').style,{left:(100*column/columns)+'%',top:(100*row/rows)+'%',width:(100/columns)+'%',height:(100/rows)+'%'});
      $('patchCaption').textContent='Token '+position+(mapped?' · row '+(row+1)+' / '+rows+' · column '+(column+1)+' / '+columns:' · Patch location unavailable')+' · '+image.source;
    }
    $('inspector').hidden=false;$('selection').textContent=describe(point);$('readout').replaceChildren();$('edgeTable').hidden=true;$('edgeNote').textContent='';
    $('edges').disabled=true;
    if(point.row.scores[point.column]===null){$('edgeNote').textContent=point.row.reason;return;}
    try{
      const result=await request('inspect',{...snapshotArgs(),rowId:point.row.id,position:snapshot.start+point.column});
      if(version!==inspectionVersion)return;
      $('selection').textContent=describe(point)+' · norm '+result.norm.toFixed(2);
      $('readout').className='readout '+(snapshot.mode==='logit'?'logit':point.row.map?.method||'');
      for(const token of result.tokens){const chip=document.createElement('span');chip.className='chip';chip.textContent=token.text;chip.title='Token '+token.id+' · Lens score '+token.score.toFixed(3);$('readout').append(chip);}
      $('edges').disabled=point.row.kind!=='output'||!!snapshot.interventions;
    }catch(error){if(version===inspectionVersion)$('edgeNote').textContent=error.message;}
  }
  $('edges').onclick=async()=>{
    if(!selected||selected.row.kind!=='output')return;
    const version=inspectionVersion,point=selected;$('edges').disabled=true;$('edgeNote').textContent='Replaying selected head to compute target contributions…';
    try{
      const result=await request('edges',{...snapshotArgs(),layer:point.row.layer,head:point.row.head,position:snapshot.start+point.column});
      if(version!==inspectionVersion)return;
      edgeData=result;
      const outside=result.rows.filter(row=>row.position<snapshot.start).length;
      const sourceAvailable=displayed.some(row=>row.id===result.sourceRowId);
      $('edgeNote').textContent=result.metric+'\nΣ source support '+result.supportSum.toFixed(4)+' · normalization bias '+result.normalizationBiasSupport.toFixed(4)+' · reconstructed target cosine '+result.targetScore.toFixed(4)+' · head reconstruction error '+result.reconstructionError.toPrecision(3)+(outside?' · '+outside+' sources outside the visible range':'')+'\n'+(sourceAvailable?'Arrows connect source Value points to the selected Output point. ':'Source Value row unavailable; contributions are shown in the table only. ')+result.warning;
      const tbody=$('edgeTable').querySelector('tbody');tbody.replaceChildren();
      for(const edge of [...result.rows].sort((a,b)=>Math.abs(b.support)-Math.abs(a.support)).slice(0,30)){
        const tr=document.createElement('tr'),td=document.createElement('td'),button=document.createElement('button');button.textContent=String(edge.position);button.onclick=()=>select({row:point.row,column:edge.position-snapshot.start});td.append(button);tr.append(td);
        for(const value of [JSON.stringify(edge.token),edge.support.toPrecision(4),edge.magnitude.toPrecision(4)]){const cell=document.createElement('td');cell.textContent=value;tr.append(cell);}tbody.append(tr);
      }
      $('edgeTable').hidden=false;draw();
    }catch(error){if(version===inspectionVersion)$('edgeNote').textContent=error.message;}
    finally{if(version===inspectionVersion)$('edges').disabled=false;}
  };
  $('form').onsubmit=async event=>{
    event.preventDefault();if(busy)return;
    const target=$('target').value;if(!target){status('Enter a target token. Leading spaces are significant.',true);return;}
    setBusy(true);
    try{
      const pieces=candidate&&candidate.text===target?{tokens:[candidate]}:await request('tokenize',{target});
      if(pieces.tokens.length>1&&encodedTarget!==target){
        $('piece').replaceChildren();for(const token of pieces.tokens){const option=document.createElement('option');option.value=token.id;option.textContent=token.id+' '+JSON.stringify(token.text);$('piece').append(option);}
        $('pieceLabel').hidden=false;encodedTarget=target;status('The text splits into multiple tokens. Choose one token, then Replay.');return;
      }
      $('pieceLabel').hidden=pieces.tokens.length===1;
      status('Replaying the prompt into an independent cache and scoring the target direction…');
      const result=await request('run',{target,mode:$('lens').value,tokenId:pieces.tokens.length===1?pieces.tokens[0].id:Number($('piece').value),start:Number($('start').value||0),...($('stop').value?{stop:Number($('stop').value)}:{})});
      snapshot=result;expanded=null;headRows.clear();selected=null;edgeData=null;inspectionVersion++;$('inspector').hidden=true;
      rows();layout();$('context').textContent='Saved snapshot · '+result.model+' · target '+JSON.stringify(result.target)+' (#'+result.targetId+') · '+result.lensLabel+' · positions '+result.start+'–'+(result.stop-1)+'\n'+result.metric+(Number.isInteger(result.maxNewTokens)?'\nPrompt '+result.promptTokenCount+' tokens · generated '+result.generatedTokenCount+' / max_new_tokens='+result.maxNewTokens+(result.stoppedAtEos?' · stopped at EOS':'')+' · private greedy replay':'');
      $('generation').hidden=!result.generatedText;$('generatedText').textContent=result.generatedText||'';
      status(result.warning);
    }catch(error){status(error.message,true);}finally{setBusy(false);}
  };
  canvas.onmousemove=event=>{const point=hit(event);$('hover').textContent=point&&!point.label&&point.column>=0&&snapshot&&point.column<snapshot.tokens.length?describe(point):'';};
  canvas.onclick=event=>{const point=hit(event);if(busy||ended||!point||!snapshot)return;if(point.label){if(point.row.id.startsWith('post:'))void expand(point.row.layer);return;}void select(point);};
  canvas.onkeydown=event=>{
    if(!snapshot||!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key))return;event.preventDefault();
    let r=selected?displayed.findIndex(row=>row.id===selected.row.id):0,c=selected?selected.column:0;
    if(event.key==='ArrowLeft')c--;if(event.key==='ArrowRight')c++;if(event.key==='ArrowUp')r--;if(event.key==='ArrowDown')r++;
    r=Math.max(0,Math.min(displayed.length-1,r));c=Math.max(0,Math.min(snapshot.tokens.length-1,c));
    const point=cell(r,c);
    if(point.x<labelWidth+10)viewport.scrollLeft=Math.max(0,viewport.scrollLeft+point.x-labelWidth-10);
    if(point.x>viewport.clientWidth-10)viewport.scrollLeft+=point.x-viewport.clientWidth+10;
    if(point.y<headerHeight+10)viewport.scrollTop=Math.max(0,viewport.scrollTop+point.y-headerHeight-10);
    if(point.y>viewport.clientHeight-10)viewport.scrollTop+=point.y-viewport.clientHeight+10;
    void select({row:displayed[r],column:c});
  };
  $('threshold').oninput=()=>{$('thresholdValue').textContent=Number($('threshold').value).toFixed(2);draw();};
  $('colorScale').oninput=()=>{const scale=Number($('colorScale').value);if(!Number.isFinite(scale)||scale<=0||scale>1)return;$('colorScaleValue').textContent=scale.toFixed(2);$('scaleLegend').textContent='Shared scale: −'+scale+' … 0 … +'+scale+' · saturated beyond endpoints · not probabilities';draw();};
  $('zoom').oninput=layout;$('edgeCount').onchange=draw;viewport.onscroll=draw;
  window.addEventListener('resize',layout);
  window.addEventListener('message',event=>{
    const data=event.data;
    if(data.type==='conceptReply'){const item=pending.get(data.requestId);if(item){pending.delete(data.requestId);if(data.error)item.reject(new Error(data.error));else item.resolve(data.result);}return;}
    if(data.type==='invalidated'){ended=true;for(const item of pending.values())item.reject(new Error(data.message));pending.clear();setBusy(false);$('edges').disabled=true;status(data.message,true);}
  });
  void request('metadata').then(data=>{
    $('lens').replaceChildren();for(const lens of data.lenses){const option=document.createElement('option');option.value=lens.id;option.textContent=lens.label;$('lens').append(option);}
    const preferred=data.lenses.find(lens=>lens.method==='j');if(preferred)$('lens').value=preferred.id;
    $('stop').placeholder=String(data.generationLimit??data.tokenCount);$('stop').max=String(data.generationLimit??data.tokenCount);$('context').textContent=data.model+' · '+data.tokenCount+' current tokens · debugger at '+data.currentToken+(Number.isInteger(data.maxNewTokens)?'\nPrompt '+data.promptTokenCount+' tokens + up to '+data.maxNewTokens+' generated tokens, from model.chat(max_new_tokens='+data.maxNewTokens+').':'');status('Enter a target token and Replay.');
  }).catch(error=>status(error.message,true));
  layout();
}

function registerConceptTrace(context, vscode) {
  let panel, session;
  function invalidate(message) { if(panel)void panel.webview.postMessage({type:'invalidated',message}); }
  context.subscriptions.push(
    vscode.commands.registerCommand('llmdbg.showConceptTrace',()=>{
      const active=vscode.debug.activeDebugSession;
      if(!active||active.type!=='llmdbg'){void vscode.window.showWarningMessage('Start and pause an llmdbg session before opening Concept Trace.');return;}
      if(panel&&session?.id===active.id){panel.reveal(panel.viewColumn);return;}
      if(panel)panel.dispose();
      session=active;
      panel=vscode.window.createWebviewPanel('llmdbg.conceptTrace','llmdbg Concept Trace',{viewColumn:vscode.ViewColumn.Beside,preserveFocus:false},{enableScripts:true,retainContextWhenHidden:true});
      const own=panel;
      void vscode.commands.executeCommand('setContext','llmdbg.conceptTraceOpen',true);
      panel.webview.onDidReceiveMessage(async message=>{
        if(message.type!=='conceptTrace')return;
        const {requestId,...args}=message;delete args.type;
        try{
          if(!session)throw new Error('The debug session has ended. Reopen Concept Trace in a paused session.');
          const result=await session.customRequest('llmdbg/conceptTrace',args);
          if(args.action==='run'&&result.images){
            const fs=require('node:fs/promises'),path=require('node:path');
            result.images=await Promise.all(result.images.map(async image=>{
              if(/^https:\/\//i.test(image.source)||/^data:image\//i.test(image.source))return {...image,preview:image.source};
              try{
                const source=path.resolve(result.sourcePath?path.dirname(result.sourcePath):session.configuration.cwd||'.',image.source);
                const mime={'.jpg':'image/jpeg','.jpeg':'image/jpeg','.png':'image/png','.webp':'image/webp','.gif':'image/gif'}[path.extname(source).toLowerCase()];
                if(!mime)return image;
                const bytes=await fs.readFile(source);return {...image,preview:'data:'+mime+';base64,'+bytes.toString('base64')};
              }catch{return image;}
            }));
          }
          if(panel===own)await own.webview.postMessage({type:'conceptReply',requestId,result});
        }catch(error){if(panel===own)await own.webview.postMessage({type:'conceptReply',requestId,error:error.message});}
      },null,context.subscriptions);
      panel.onDidDispose(()=>{if(panel===own){panel=undefined;session=undefined;void vscode.commands.executeCommand('setContext','llmdbg.conceptTraceOpen',false);}},null,context.subscriptions);
      panel.webview.html=conceptTraceHtml(panel.webview);
    }),
    vscode.commands.registerCommand('llmdbg.closeConceptTrace',()=>panel?.dispose()),
    vscode.debug.onDidTerminateDebugSession(ended=>{if(ended.id===session?.id){session=undefined;invalidate('The debug session ended. This graph is a saved snapshot; reopen Concept Trace in a new session to inspect or replay.');}}),
    {dispose(){panel?.dispose();}}
  );
}

module.exports={conceptTraceHtml,registerConceptTrace};
