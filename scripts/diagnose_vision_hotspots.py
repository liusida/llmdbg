"""Reproduce image-patch hotspots independently of the debugger UI.

Uses the authored example's prompt/revision, actual cached Q/K/V, and teacher-
forced input-embedding ablations. No debugger/runtime code is modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import io
import json
import os
from pathlib import Path

import torch

from llmdbg.config import load_debug_config
from llmdbg.dap import LLMDebugAdapter, ProtocolStream
from llmdbg.model_runtime import prepare_model_execution


def ranked(values, positions, count=8):
    order = values.abs().argsort(descending=True)[:count].tolist()
    return [{"position": positions[i], "value": float(values[i])} for i in order]


def image_description(execution, messages):
    adapter = LLMDebugAdapter(ProtocolStream(io.BytesIO(), io.BytesIO()))
    adapter.execution, adapter.tokens, adapter.messages = execution, execution.tokens, messages
    return adapter._prompt_images()[0]


def summarize(execution, trace, query, image, targets, layer, channel):
    start, end = image['startToken'], image['endToken']
    positions = list(range(start, end+1))
    records = []
    for state in trace.states:
        if not state.layer.name.endswith('.full_attention'):
            continue
        index = int(state.layer.name.split('.')[2])
        weights = state.tensors['attention_weights'][0, :, -1].detach().float().cpu()
        values = state.tensors['attention_cached_values'][0].detach().float().cpu()
        keys = state.tensors['attention_cached_keys'][0].detach().float().cpu()
        width, kv_count = values.shape[-1], values.shape[0]
        for head in range(weights.shape[0]):
            kv = head // (weights.shape[0]//kv_count)
            visible = [p for p in positions if p < weights.shape[-1]]
            if not visible:
                continue
            dim = channel % width
            alpha, val = weights[head, visible], values[kv, visible, dim]
            product = alpha * val
            key_norm, value_norm = keys[kv, visible].norm(dim=-1), values[kv, visible].norm(dim=-1)
            by_product = product.abs().argsort(descending=True).tolist()
            by_attention = alpha.argsort(descending=True).tolist()
            total = float(weights[head].sum())
            item = {'query': query, 'query_text': execution.tokens[query], 'layer': index,
                    'head': head, 'kv_head': kv, 'dimension': dim,
                    'image_attention_mass': float(alpha.sum()), 'total_attention_mass': total,
                    'targets_attention_mass': sum(float(weights[head,p]) for p in targets if p < weights.shape[-1]),
                    'targets_absolute_product_fraction_among_image': sum(abs(float(weights[head,p]*values[kv,p,dim])) for p in targets if p < weights.shape[-1])/max(float(product.abs().sum()),1e-30),
                    'top_attention_image': ranked(alpha,visible,3), 'top_product_image': ranked(product,visible,3)}
            if index == layer and head == channel//width:
                item.update(targets=[{'position': p, 'weight': float(weights[head,p]),
                                     'value': float(values[kv,p,dim]), 'product': float(weights[head,p]*values[kv,p,dim]),
                                     'attention_rank_among_image': by_attention.index(visible.index(p))+1,
                                     'product_rank_among_image': by_product.index(visible.index(p))+1,
                                     'key_norm': float(keys[kv,p].norm()), 'value_vector_norm': float(values[kv,p].norm())}
                                    for p in targets if p in visible],
                            top_key_norm=ranked(key_norm,visible), top_value_norm=ranked(value_norm,visible),
                            top_absolute_value=ranked(val,visible),
                            image_key_norm_median=float(key_norm.median()), image_value_norm_median=float(value_norm.median()),
                            image_abs_value_median=float(val.abs().median()),
                            top_all_source_products=ranked(weights[head]*values[kv,:,dim],list(range(weights.shape[-1]))),
                            channel_sum_before_gate=float((weights[head]*values[kv,:,dim]).sum()),
                            channel_mixed_after_gate=float(state.tensors['full_attention_mixing'].flatten()[channel]))
            records.append(item)
    return records


def prediction(execution, trace):
    logits, predictions = execution.vocabulary_projection(trace.states[-1].output_vector)
    probabilities = torch.tensor(logits).softmax(dim=-1)
    tracked = {}
    for text in [' cat',' dog',' ginger',' tab','A','An',' lying']:
        ids = execution.runtime.tokenizer.encode(text,add_special_tokens=False)
        if len(ids) == 1:
            tracked[text] = {'id':ids[0],'probability':float(probabilities[ids[0]]),'logit':logits[ids[0]]}
    return {'top': [{'text': p.text, 'probability': p.probability, 'logit': p.logit} for p in predictions[:5]],
            'tracked_tokens':tracked}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[2]/'llmdbg-usecase/Qwen35-2B-Vision.llmdbg.py')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=int, default=142)
    parser.add_argument('--layer', type=int, default=15)
    parser.add_argument('--channel', type=int, default=241)
    args = parser.parse_args()
    output = args.output.resolve()
    config_path = args.config.resolve()
    config = load_debug_config(config_path)
    os.chdir(config_path.parent)
    execution = prepare_model_execution(config['model'], revision=config.get('revision','main'),
        messages=config['messages'], device=config.get('device','auto'), dtype=config.get('dtype','auto'),
        progress=lambda msg: print(msg,end='',flush=True))
    image = image_description(execution,config['messages'])
    start, end, columns = image['startToken'], image['endToken'], image['gridColumns']
    targets = [args.source, args.source+columns-1, args.source+columns+1]
    if not all(start <= p <= end for p in targets):
        raise ValueError(f'Target tokens {targets} do not fit image span {start}..{end}')
    embeddings = execution.inputs_embeds[0,start:end+1].detach().float().cpu()
    native_positions = execution.position_ids[:,0,start:end+1].detach().cpu()
    rows = torch.arange(end-start+1)//columns
    cols = torch.arange(end-start+1)%columns
    mapping_valid = bool(torch.equal(native_positions[1]-native_positions[1,0], rows)
                         and torch.equal(native_positions[2]-native_positions[2,0], cols))
    if not mapping_valid:
        raise RuntimeError('Native M-RoPE row/column order disagrees with the UI grid')
    norms = embeddings.norm(dim=-1)
    top_norm = ranked(norms,list(range(start,end+1)),12)
    report = {'model':config['model'],'revision':config.get('revision'), 'prompt':execution.rendered_prompt,
              'initial_token_count':execution.token_count,'image':image,'mapping_matches_native_mrope':mapping_valid,
              'targets':[{'position':p,'row_1based':(p-start)//columns+1,'column_1based':(p-start)%columns+1,
                          'embedding_norm':float(norms[p-start]),
                          'embedding_norm_rank':int((norms>norms[p-start]).sum())+1,
                          'rope_position':execution.position_ids[...,p].detach().cpu().tolist()} for p in targets],
              'image_embedding_norm_median':float(norms.median()),'top_embedding_norms':top_norm,
              'records':[], 'predictions':[]}
    last = execution.token_count-1
    # Two image positions, final prompt position, then three generated positions.
    for query in sorted(set([min(end,200),min(end,250),last])):
        trace = execution.trace_token(query)
        report['records'].extend(summarize(execution,trace,query,image,targets,args.layer,args.channel))
        if query == last:
            report['predictions'].append({'query':query,**prediction(execution,trace)})
    for _ in range(3):
        _, choices = execution.vocabulary_projection(trace.states[-1].output_vector)
        execution.append_generated_token(choices[0].token_id)
        query = execution.token_count-1
        trace = execution.trace_token(query)
        report['records'].extend(summarize(execution,trace,query,image,targets,args.layer,args.channel))
        report['predictions'].append({'query':query,**prediction(execution,trace)})
    main_query = last+1
    # Actual row-major mapping can be cross-checked against the native M-RoPE coordinates.
    report['rope_image_grid'] = execution.position_ids[...,start:end+1].detach().cpu().tolist()
    frequencies = Counter()
    for record in report['records']:
        if record['query'] >= last:
            frequencies.update(p['position'] for p in record['top_attention_image'])
    report['top_attention_patch_frequency'] = frequencies.most_common(15)
    original = execution.inputs_embeds.detach().clone()
    mean = original[:,start:end+1].float().mean(dim=1).to(original.dtype)
    controls = [start+10,start+columns*5+10,end-10]
    report['ablations'] = []
    for name, selected in [('baseline',[]),('target_patches',targets),('nearby_patches',[args.source-1,args.source+1,args.source+columns]),('control_patches',controls)]:
        execution.inputs_embeds = original.clone()
        if selected:
            execution.inputs_embeds[:,selected] = mean[:,None]
        execution.past_key_values, execution.cached_through = None,-1
        trace = execution.trace_token(main_query)
        metrics = summarize(execution,trace,main_query,image,targets,args.layer,args.channel)
        report['ablations'].append({'name':name,'replaced_positions':selected,
            'replacement':'mean of original image embeddings; fixed baseline generated token history',
            **prediction(execution,trace),
            'selected_channel':next(r for r in metrics if r['layer']==args.layer and r['head']==args.channel//256)})
        sweep = []
        execution.past_key_values, execution.cached_through = None,-1
        for query in range(last,last+4):
            trace = execution.trace_token(query)
            sweep.append({'query':query,'query_text':execution.tokens[query],**prediction(execution,trace),
                'records':summarize(execution,trace,query,image,targets,args.layer,args.channel)})
        report['ablations'][-1]['query_sweep'] = sweep
        # Let each variant generate its own continuation: no baseline answer
        # tokens are supplied to this run. Preserve exactly the authored prompt.
        generated = replace(execution, input_ids=execution.input_ids[:,:last+1].clone(),
            tokens=execution.tokens[:last+1],inputs_embeds=execution.inputs_embeds[:,:last+1].clone(),
            position_ids=execution.position_ids[...,:last+1].clone(),past_key_values=None,cached_through=-1)
        generated_ids = []
        for _ in range(10):
            trace = generated.trace_token(generated.token_count-1)
            _, choices = generated.vocabulary_projection(trace.states[-1].output_vector)
            next_id = choices[0].token_id
            generated_ids.append(next_id)
            generated.append_generated_token(next_id)
        report['ablations'][-1]['free_greedy_continuation_10_tokens'] = execution.runtime.tokenizer.decode(generated_ids)
    # Same image, same teacher-forced query token, different question. Compare
    # relative patch locations since the text length shifts the image span.
    report['question_controls'] = []
    for question in ['What color is the cat? Answer briefly.', 'What furniture is in the background? Answer briefly.']:
        messages = deepcopy(config['messages'])
        for message in messages:
            if isinstance(message.get('content'), list):
                for part in message['content']:
                    if part.get('type') == 'text':
                        part['text'] = question
        control = prepare_model_execution(config['model'], revision=config.get('revision','main'), messages=messages,
            device=config.get('device','auto'),dtype=config.get('dtype','auto'),progress=lambda msg: print(msg,end='',flush=True))
        other_image = image_description(control,messages)
        other_targets = [p-start+other_image['startToken'] for p in targets]
        control.append_generated_token(int(execution.input_ids[0,main_query]))
        query = control.token_count-1
        trace = control.trace_token(query)
        records = summarize(control,trace,query,other_image,other_targets,args.layer,args.channel)
        report['question_controls'].append({'question':question,'query_token':control.tokens[query],
            'image_embedding_max_difference':float((control.inputs_embeds[0,other_image['startToken']:other_image['endToken']+1].detach().float().cpu()-embeddings).abs().max()),
            **prediction(control,trace), 'records':records})
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(f'Wrote {output}',flush=True)


if __name__ == '__main__':
    main()
