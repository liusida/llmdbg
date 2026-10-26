"""Find cat-family J-Lens readouts in a layer's incoming residual stream."""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import torch

from llmdbg.config import load_debug_config
from llmdbg.dap import LLMDebugAdapter, ProtocolStream
from llmdbg.model_runtime import prepare_model_execution
from llmdbg.workspace_lens import resolve_view_site


def cat_family(text):
    return '猫' in text or '貓' in text or text.strip().lower() in {'cat', 'cats'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[2]/'llmdbg-usecase/Qwen35-2B-Vision.llmdbg.py')
    parser.add_argument('--layer',type=int,default=17)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output,config_path=args.output.resolve(),args.config.resolve()
    config=load_debug_config(config_path)
    os.chdir(config_path.parent)
    execution=prepare_model_execution(config['model'],revision=config.get('revision','main'),
        messages=config['messages'],device=config.get('device','auto'),dtype=config.get('dtype','auto'),
        progress=lambda msg: print(msg,end='',flush=True))
    adapter=LLMDebugAdapter(ProtocolStream(io.BytesIO(),io.BytesIO()))
    adapter.execution,adapter.tokens,adapter.messages=execution,execution.tokens,config['messages']
    images=adapter._prompt_images()
    captured=[]
    def capture(_module,inputs):
        captured.append(inputs[0].detach().float().cpu()[0])
    handle=execution.runtime.layers[args.layer].register_forward_pre_hook(capture)
    try:
        execution._prefill(execution.token_count)
    finally:
        handle.remove()
    vectors=torch.cat(captured)
    assert vectors.shape[0]==execution.token_count
    site,source_layer=resolve_view_site('residual-pre',args.layer)
    layer_name=f'model.layers.{args.layer}.gated_linear_attention'
    def decode(batch):
        results=execution._vectors_lens(layer_name,tuple(tuple(row) for row in batch.tolist()),
            source_site=site,source_layer=source_layer,token_count=10,
            include_logit_lens=False,include_j_lens=True,include_r_lens=False)
        if any(not result.j_lens for result in results):raise RuntimeError('J-Lens map unavailable')
        return results
    def describe(position,projection):
        tokens=[{'text':t.text,'id':t.token_id,'score':t.score} for t in projection.j_lens]
        matches=[rank for rank,t in enumerate(tokens,1) if cat_family(t['text'])]
        row={'position':position,'token':execution.tokens[position],'norm':projection.norm,
             'cat_rank':min(matches) if matches else None,'tokens':tokens}
        for image in images:
            if image.get('startToken',-1)<=position<=image.get('endToken',-2):
                offset=position-image['startToken']
                row.update(row_1based=offset//image['gridColumns']+1,column_1based=offset%image['gridColumns']+1)
        return row
    rows=[]
    for start in range(0,len(vectors),16):
        rows.extend(describe(start+i,result) for i,result in enumerate(decode(vectors[start:start+16])))
    matches=[r for r in rows if r['cat_rank'] is not None]
    top1=[r for r in matches if r['cat_rank']==1]
    # Validate boundaries and strongest matches using the exact paused-token path.
    positions={38,94,95}
    if matches:positions.update(range(max(0,matches[0]['position']-1),matches[0]['position']+2))
    positions.update(r['position'] for r in top1[:8])
    checks=[]
    for position in sorted(p for p in positions if p<execution.token_count):
        execution.cached_through,execution.past_key_values=-1,None
        state=next(s for s in execution.trace_token(position).states if s.layer.name==layer_name)
        projection=decode(torch.tensor([state.input_vector]))[0]
        checks.append(describe(position,projection))
        print('SPOT_CHECK',position,'rank',checks[-1]['cat_rank'],[t['text'] for t in checks[-1]['tokens']],flush=True)
    report={'model':config['model'],'revision':config.get('revision'),'view':'residual-pre','layer':args.layer,
            'source_site':site,'source_layer':source_layer,'lens_metadata':projection.j_lens_metadata,
            'prompt':execution.rendered_prompt,'images':images,'rows':rows,'paused_token_checks':checks,
            'matching_rule':'Chinese 猫/貓 or exact stripped case-insensitive cat/cats, within top 10'}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('CAT_TOP10',[(r['position'],r['cat_rank']) for r in matches],flush=True)
    print('CAT_TOP1',[r['position'] for r in top1],flush=True)
    print('WROTE',output,flush=True)


if __name__=='__main__':main()
