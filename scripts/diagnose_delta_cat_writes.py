"""Trace DeltaNet write content and a fixed final-query read through prompt history."""
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


def feline(text):
    return '猫' in text or text.strip().lower() in {'cat', 'cats'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[2]/'llmdbg-usecase/Qwen35-2B-Vision.llmdbg.py')
    parser.add_argument('--layer', type=int, default=17)
    parser.add_argument('--head', type=int, default=12)
    parser.add_argument('--query', type=int, default=281)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--spot-check', type=int, nargs='+', help='Check selected positions with independent debugger-style prefills')
    args = parser.parse_args()
    output, config_path = args.output.resolve(), args.config.resolve()
    config = load_debug_config(config_path)
    os.chdir(config_path.parent)
    execution = prepare_model_execution(config['model'],revision=config.get('revision','main'),
        messages=config['messages'],device=config.get('device','auto'),dtype=config.get('dtype','auto'),
        progress=lambda message: print(message,end='',flush=True) if not message.startswith(('Preparing token','Resident model lock','Captured ')) else None)
    execution.capture_delta_core = True
    if args.query >= execution.token_count:
        raise ValueError(f'Query {args.query} outside prompt of {execution.token_count} tokens')
    adapter = LLMDebugAdapter(ProtocolStream(io.BytesIO(),io.BytesIO()))
    adapter.execution, adapter.tokens, adapter.messages = execution, execution.tokens, config['messages']
    images = adapter._prompt_images()
    layer_name = f'model.layers.{args.layer}.gated_linear_attention'
    mixer = execution.runtime.layers[args.layer].linear_attn
    width = mixer.head_v_dim
    weight = mixer.out_proj.weight.detach().T[args.head*width:(args.head+1)*width].float()
    if args.spot_check:
        final = next(s for s in execution.trace_token(args.query).states if s.layer.name==layer_name)
        q = final.tensors['delta_query'][0,0,args.head].float()
        gate = final.tensors['delta_gate'][0,0,args.head]
        records=[]
        for position in args.spot_check:
            execution.cached_through,execution.past_key_values=-1,None
            state=next(s for s in execution.trace_token(position).states if s.layer.name==layer_name)
            t=state.tensors
            correction=adapter._delta_memory_write(state).reshape(mixer.num_v_heads,width)[args.head]
            read=q @ t['delta_updated_memory'][0,args.head].float()
            with torch.inference_mode():
                projected=mixer.norm(read.to(gate.dtype),gate).float() @ weight
                vectors=(tuple((correction@weight).cpu().tolist()),tuple(projected.cpu().tolist()))
            decoded=execution._vectors_lens(layer_name,vectors,source_site='mid',source_layer=args.layer,
                token_count=5,include_logit_lens=False,include_j_lens=True,include_r_lens=False)
            row={'position':position,'token':execution.tokens[position]}
            for kind,result in zip(('write','fixed_query_read'),decoded):
                row[kind]={'projected_norm':result.norm,'tokens':[{'text':item.text,'score':item.score} for item in result.j_lens]}
            records.append(row)
            print('SPOT_CHECK',json.dumps(row,ensure_ascii=False),flush=True)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps({'model':config['model'],'revision':config.get('revision'),
            'layer':args.layer,'head':args.head,'query':args.query,'mode':'independent debugger prefills',
            'images':images,'records':records},ensure_ascii=False,indent=2)+'\n')
        return
    rows, writes, values, memories, keys, decays, gates = [], [], [], [], [], [], []
    for position in range(args.query+1):
        trace = execution.trace_token(position)
        state = next(s for s in trace.states if s.layer.name==layer_name)
        t = state.tensors
        correction = adapter._delta_memory_write(state).reshape(mixer.num_v_heads,width)[args.head]
        writes.append(correction.cpu())
        values.append(t['delta_value'][0,0,args.head].float().cpu())
        memories.append(t['delta_updated_memory'][0,args.head].float().cpu())
        keys.append(t['delta_key'][0,0,args.head].float().cpu())
        decays.append(float(t['delta_decay'][0,0,args.head]))
        gates.append(t['delta_gate'][0,0,args.head].detach().cpu())
        row = {'position':position,'token':execution.tokens[position],
               'beta':float(t['delta_beta'][0,0,args.head]),'decay':decays[-1],
               'write_norm':float(correction.norm())}
        for image_index,image in enumerate(images):
            if image.get('startToken',-1)<=position<=image.get('endToken',-2):
                offset=position-image['startToken']
                row.update(image=image_index+1,row_1based=offset//image['gridColumns']+1,
                           column_1based=offset%image['gridColumns']+1)
        rows.append(row)
        if position%32==0: print('CAPTURE',position,flush=True)
    query = t['delta_query'][0,0,args.head].float().cpu()
    readout = t['delta_readout'][0,0,args.head].float().cpu()
    gate = gates[-1].to(weight.device)
    memory_batch = torch.stack(memories)
    fixed_reads = torch.einsum('k,skv->sv',query,memory_batch)
    torch.testing.assert_close(fixed_reads[-1],readout,atol=1e-4,rtol=1e-3)
    with torch.inference_mode():
        normalized = mixer.norm(fixed_reads.to(device=weight.device,dtype=mixer.out_proj.weight.dtype),
                                gate.expand(len(rows),-1)).float()
        final_mixed = t['gated_delta_mixing'].flatten()[args.head*width:(args.head+1)*width].float()
        torch.testing.assert_close(normalized[-1],final_mixed,atol=.015,rtol=.03)
        batches = {'write':torch.stack(writes).to(weight.device)@weight,
                   'value':torch.stack(values).to(weight.device)@weight,
                   'fixed_query_read':normalized@weight}
    # Exact additive reconstruction with observed innovations held fixed.
    # Not a causal write-deletion experiment: future corrections depend on history.
    decay_after = torch.ones(len(rows))
    for s in range(len(rows)-2,-1,-1):decay_after[s]=decay_after[s+1]*decays[s+1]
    retrieval = torch.stack(keys)@query
    contributions = torch.stack(writes)*((decay_after*retrieval)[:,None])
    torch.testing.assert_close(contributions.sum(0),readout,atol=1e-4,rtol=2e-3)
    final_projected = batches['fixed_query_read'][-1]
    target_direction=final_projected/final_projected.norm().clamp_min(1e-30)
    projected_contributions=contributions.to(weight.device)@weight
    # Readout-space alignment is descriptive; nonlinear output norm/gate are excluded.
    for i,row in enumerate(rows):
        row['fixed_query_key_dot']=float(retrieval[i])
        row['decay_survival']=float(decay_after[i])
        row['surviving_readout_contribution_norm']=float(contributions[i].norm())
        row['projected_readout_contribution_norm']=float(projected_contributions[i].norm())
        row['alignment_to_final_head_output']=float(projected_contributions[i]@target_direction)
    lens_metadata=None
    for kind,vectors in batches.items():
        for start in range(0,len(rows),16):
            decoded=execution._vectors_lens(layer_name,tuple(tuple(v) for v in vectors[start:start+16].cpu().tolist()),
                source_site='mid',source_layer=args.layer,token_count=10,include_logit_lens=False,
                include_j_lens=True,include_r_lens=False)
            for row,projection in zip(rows[start:start+16],decoded):
                if not projection.j_lens:raise RuntimeError('J-Lens unavailable for requested layer/site')
                lens_metadata=projection.j_lens_metadata
                tokens=[{'id':item.token_id,'text':item.text,'score':item.score} for item in projection.j_lens]
                row[kind]={'projected_norm':projection.norm,'tokens':tokens,
                           'feline_top1':feline(tokens[0]['text']),
                           'feline_top5':any(feline(item['text']) for item in tokens[:5])}
        print('DECODED',kind,flush=True)
    report={'model':config['model'],'revision':config.get('revision'),'layer':args.layer,'head':args.head,
            'query':args.query,'query_token':execution.tokens[args.query], 'prompt':execution.rendered_prompt,
            'images':images,'lens_metadata':lens_metadata,'rows':rows,
            'checks':{'readout_reconstruction':True,'native_gated_head_reproduction':True,
                      'fixed_query_probe':'Final query and gate held fixed while substituting each earlier memory snapshot',
                      'contributions':'Observed innovations, with later decays and final query; exact sum but not causal deletion effects'}}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('FINAL_HEAD',rows[-1]['fixed_query_read']['tokens'][:5],flush=True)
    for kind in ('write','value','fixed_query_read'):
        matches=[r for r in rows if r[kind]['feline_top1']]
        print('FELINE_TOP1',kind,[r['position'] for r in matches],flush=True)
        strongest=sorted(matches,key=lambda r:r[kind]['projected_norm'],reverse=True)[:12]
        print('STRONGEST',kind,[(r['position'],r.get('row_1based'),r.get('column_1based'),round(r[kind]['projected_norm'],4),[t['text'] for t in r[kind]['tokens'][:5]]) for r in strongest],flush=True)
    print('WROTE',output,flush=True)


if __name__=='__main__':main()
