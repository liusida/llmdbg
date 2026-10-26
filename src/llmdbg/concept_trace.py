"""Snapshot-only Concept Trace replay and target-direction readouts.

This module never advances the live debugger execution. All replay caches,
hooks, and captured vectors belong to one bounded, explicitly requested snapshot.
"""
from __future__ import annotations

from dataclasses import replace
from contextlib import contextmanager
import importlib
import uuid
from typing import Any

from llmdbg.model_runtime import _runtime_lock
from llmdbg.residual_vocabulary_lenses import activate, available_lenses, lens


def metadata(adapter):
    execution = adapter.execution
    if execution is None:
        raise ValueError('Start and pause a model session before opening Concept Trace.')
    model = str(adapter.model_name)
    prompt_count, budget = generation_budget(adapter)
    return {'model': model, 'tokenCount': execution.token_count, 'currentToken': adapter.token_index,
            'promptTokenCount': prompt_count, 'maxNewTokens': budget, 'generationLimit': prompt_count + budget,
            'lenses': [{'id': 'logit', 'label': 'Logit Lens', 'method': 'logit'}] +
                      [{'id': 'lens:' + item['id'], **{k: v for k, v in item.items() if k != 'id'}}
                       for item in available_lenses(model)]}


def generation_budget(adapter):
    execution = adapter.execution
    generated = max(0, int(getattr(adapter, 'generated_tokens', 0)))
    prompt_count = execution.token_count - min(generated, execution.token_count)
    return prompt_count, max(0, int(getattr(adapter, 'max_new_tokens', 0)))


def target_tokens(execution, text):
    if not isinstance(text, str) or not text:
        raise ValueError('Enter a target token; leading spaces are significant.')
    ids = execution.runtime.tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError('The target does not tokenize to a vocabulary token.')
    return [{'id': int(token), 'text': execution.runtime.tokenizer.decode([int(token)])} for token in ids]


def cosine_scores(execution, vectors, token_id, mapping=None):
    """Match the Lens pipeline, computing just one target direction, not all logits."""
    import torch
    weight = execution.runtime.model.get_output_embeddings().weight
    target = weight[token_id].float()
    target = target / target.norm().clamp_min(1e-30)
    scores = []
    with torch.inference_mode():
        for chunk in vectors.reshape(-1, vectors.shape[-1]).split(64):
            directions = chunk.to(device=weight.device, dtype=weight.dtype)
            if mapping is not None:
                directions = (directions.float() @ mapping.to(device=weight.device, dtype=torch.float32).T).to(weight.dtype)
            decoded = execution.runtime.final_norm(directions[:, None, :])[:, 0].float()
            similarity = (decoded @ target) / decoded.norm(dim=-1).clamp_min(1e-30)
            if not torch.isfinite(similarity).all():
                raise ValueError('Non-finite vectors cannot be displayed as cosine scores.')
            scores.extend(similarity.clamp(-1, 1).cpu().tolist())
    return scores


def target_support(execution, parts, token_id, mapping=None):
    """Additive target-cosine attribution with final normalization statistics frozen."""
    import torch
    weight = execution.runtime.model.get_output_embeddings().weight
    parts = parts.to(device=weight.device, dtype=torch.float32)
    if mapping is not None: parts = parts @ mapping.to(device=weight.device, dtype=torch.float32).T
    total = parts.sum(0)
    norm = execution.runtime.final_norm
    with torch.inference_mode():
        actual = norm(total.to(weight.dtype)[None, None, :])[0, 0].float()
        if isinstance(norm, torch.nn.LayerNorm):
            denominator = (total.var(unbiased=False) + norm.eps).sqrt()
            normalized_parts = (parts - parts.mean(-1, keepdim=True)) / denominator
            if norm.weight is not None: normalized_parts *= norm.weight.float()
            bias = norm.bias.float() if norm.bias is not None else torch.zeros_like(total)
        else:
            eps = getattr(norm, 'eps', getattr(norm, 'variance_epsilon', None))
            if eps is None: raise ValueError('Target edge attribution does not support this final normalization.')
            # Infer the effective RMS weight, including Gemma/Qwen's (1 + weight).
            effective_weight = norm(torch.ones_like(total, dtype=weight.dtype)[None, None, :])[0, 0].float() * (1 + eps)**.5
            normalized_parts = parts * effective_weight / (total.square().mean() + eps).sqrt()
            bias = torch.zeros_like(total)
        target = weight[token_id].float()
        target = target / target.norm().clamp_min(1e-30)
        scale = actual.norm().clamp_min(1e-30)
        support = normalized_parts @ target / scale
        return support.cpu(), float(actual @ target / scale), float(bias @ target / scale)


class Snapshot:
    def __init__(self, adapter, args):
        import torch
        self.execution = adapter.execution
        if self.execution is None:
            raise ValueError('A paused model execution is required.')
        self.id = uuid.uuid4().hex
        self.target = target_tokens(self.execution, args.get('target'))
        supplied = args.get('tokenId')
        if supplied is None and len(self.target) != 1:
            raise ValueError('The target splits into multiple tokens. Choose one of the token IDs first.')
        self.token_id = int(self.target[0]['id'] if supplied is None else supplied)
        if self.token_id not in [item['id'] for item in self.target]:
            raise ValueError('Selected token ID does not belong to the entered target text.')
        self.mode = str(args.get('mode', 'logit'))
        self.selected = lens(self.mode[5:]) if self.mode.startswith('lens:') else None
        if self.mode != 'logit' and (self.selected is None or not self.selected.get('enabled', True)):
            raise ValueError('Choose an available vocabulary lens.')
        self.method = self.selected['method'] if self.selected else 'logit'
        self.prompt_count, self.max_new_tokens = generation_budget(adapter)
        limit = self.prompt_count + self.max_new_tokens
        self.start = int(args.get('start', 0))
        self.stop = int(args.get('stop', limit))
        if not 0 <= self.start < self.stop <= limit:
            raise ValueError(f'Token range must satisfy 0 ≤ start < stop ≤ {limit}: '
                             f'{self.prompt_count} prompt tokens + max_new_tokens={self.max_new_tokens} (stop is exclusive).')
        if self.stop - self.start > 1024:
            raise ValueError('Display at most 1024 token positions per snapshot; narrow the range.')
        self.eos_stopped = False
        self.newly_generated = 0
        self.model = str(adapter.model_name)
        self.rows, self.vectors, self.mixed, self.values, self.head_rows = [], {}, {}, {}, {}
        self.maps, self.edge_cache = {}, {}
        self.replay = replace(self.execution, input_ids=self.execution.input_ids.clone(),
            inputs_embeds=self.execution.inputs_embeds.clone() if self.execution.inputs_embeds is not None else None,
            past_key_values=None, cached_through=-1, capture_delta_core=False,
            interventions=dict(self.execution.interventions), ica_interventions=dict(self.execution.ica_interventions),
            vector_patches=dict(self.execution.vector_patches), ica_observations={})
        self.intervened = bool(self.replay.interventions or self.replay.ica_interventions or self.replay.vector_patches)
        self.current_token = adapter.token_index
        with _runtime_lock, torch.inference_mode(), activate(self.selected['id'] if self.selected else None):
            self._complete_tokens()
            self.tokens = tuple(self.replay.tokens[:self.stop])
            self._capture()
        self.replay.past_key_values = None  # Retain scores/vectors, not another model cache.

    def _next_token(self, hidden):
        """Match debugger greedy vocabulary projection without copying all logits to CPU."""
        import torch
        weight = self.execution.runtime.model.get_output_embeddings().weight
        logits = weight @ hidden.to(device=weight.device, dtype=weight.dtype)
        config = getattr(self.execution.runtime.model.config, 'text_config', self.execution.runtime.model.config)
        cap = getattr(config, 'final_logit_softcapping', None)
        if cap is not None: logits = torch.tanh(logits / cap) * cap
        return int(logits.float().argmax())

    def _complete_tokens(self):
        """Extend only the private replay, within the authored total generation budget."""
        import torch
        if self.stop <= self.replay.token_count:
            return
        runtime = self.execution.runtime
        eos = getattr(runtime.tokenizer, 'eos_token_id', None)
        if eos is None:
            eos = getattr(getattr(runtime.model, 'generation_config', None), 'eos_token_id', None)
        eos_ids = set(eos if isinstance(eos, (tuple, list, set)) else [eos])
        hidden = [None]
        def capture(_module, _inputs, output):
            hidden[0] = output[0, -1].detach()
        handle = runtime.final_norm.register_forward_hook(capture)
        try:
            if self.intervened:
                for position in range(self.replay.token_count):
                    trace = self.replay.trace_token(position)
                hidden[0] = torch.tensor(trace.states[-1].output_vector, device=self.execution.device)
            else:
                self.replay._prefill(self.replay.token_count)
            while self.replay.token_count < self.stop:
                token_id = self._next_token(hidden[0])
                if token_id in eos_ids:
                    self.eos_stopped = True
                    break  # Match the debugger: do not append EOS as a new position.
                self.replay.append_generated_token(token_id)
                self.newly_generated += 1
                if self.replay.token_count < self.stop:
                    if self.intervened:
                        trace = self.replay.trace_token(self.replay.token_count - 1)
                        hidden[0] = torch.tensor(trace.states[-1].output_vector, device=self.execution.device)
                    else:
                        self.replay._prefill(self.replay.token_count)
            self.stop = min(self.stop, self.replay.token_count)
            if self.start >= self.stop:
                raise ValueError('Generation reached EOS before the selected token range. Choose an earlier start position.')
        finally:
            handle.remove()
            self.replay.past_key_values = None
            self.replay.cached_through = -1

    def mapping(self, site, layer, width):
        key = (site, layer)
        if key not in self.maps:
            self.maps[key] = (None, {}) if self.method == 'logit' else self.execution._workspace_lens_source(
                self.method, self.model, site, layer, width)
        return self.maps[key]

    def row(self, key, label, layer, site, site_layer, vectors):
        source = self.mapping(site, site_layer, vectors.shape[-1])
        row = {'id': key, 'label': label, 'layer': layer, 'site': site, 'siteLayer': site_layer,
               'scores': cosine_scores(self.execution, vectors, self.token_id, source[0]) if source else [None] * len(vectors),
               'norms': vectors.float().norm(dim=-1).tolist(), 'available': source is not None,
               'map': source[1] if source else None,
               'reason': None if source else 'No fitted map for this lens and activation site.'}
        return row

    @contextmanager
    def capture_heads(self):
        """Only observe native values and projection inputs; always restore hooks/kernels."""
        handles, originals = [], []
        active = [None]
        runtime = self.execution.runtime
        def save(store, layer, tensor):
            if self.intervened:
                return  # Residual replay supports interventions; head observations need native consistency.
            store[layer] = tensor[0, self.start:self.stop].detach().cpu().clone()
        def mixed_hook(layer):
            def hook(_module, inputs): save(self.mixed, layer, inputs[0])
            return hook
        def value_hook(layer, fused=False):
            def hook(_module, _inputs, output):
                save(self.values, layer, output.chunk(3, -1)[2] if fused else output)
            return hook
        def enter(layer):
            def hook(_module, _inputs): active[0] = layer
            return hook
        def leave(_module, _inputs, _output): active[0] = None
        def wrap(original):
            def kernel(query, key, value, *args, **kwargs):
                if active[0] is not None: save(self.values, active[0], value)
                return original(query, key, value, *args, **kwargs)
            return kernel
        try:
            if not self.intervened:
                for index, block in enumerate(runtime.layers):
                    kind = runtime.layer_kind(index)
                    mixer = block.attn if runtime.architecture == 'gpt2' else block.linear_attn if kind == 'gated_linear_attention' else block.self_attn
                    projection = mixer.c_proj if runtime.architecture == 'gpt2' else getattr(mixer, 'o_proj', None) or mixer.out_proj
                    handles.append(projection.register_forward_pre_hook(mixed_hook(index)))
                    if kind == 'gated_linear_attention':
                        handles.extend([mixer.register_forward_pre_hook(enter(index)), mixer.register_forward_hook(leave)])
                        implementation = importlib.import_module(type(mixer).__module__)
                        for name in ('torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule'):
                            if any(module is implementation and field == name for module, field, _ in originals): continue
                            original = getattr(implementation, name)
                            originals.append((implementation, name, original))
                            setattr(implementation, name, wrap(original))
                    else:
                        module = mixer.c_attn if runtime.architecture == 'gpt2' else getattr(mixer, 'v_norm', None) if runtime.architecture == 'gemma4' else getattr(mixer, 'v_proj', None)
                        if module is not None:
                            handles.append(module.register_forward_hook(value_hook(index, runtime.architecture == 'gpt2')))
            yield
        finally:
            for module, name, original in reversed(originals): setattr(module, name, original)
            for handle in handles: handle.remove()

    def _capture(self):
        import torch
        runtime = self.execution.runtime
        handles = []
        if self.intervened:
            collected = {'embed': []} | {f'post:{i}': [] for i in range(len(runtime.layers))}
            for position in range(self.stop):
                trace = self.replay.trace_token(position)
                if position < self.start: continue
                for index in range(len(runtime.layers)):
                    kind = runtime.layer_kind(index)
                    attention = next(s for s in trace.states if s.layer.name == runtime.state_name(index, kind))
                    mlp = next(s for s in trace.states if s.layer.name == runtime.state_name(index, 'mlp'))
                    if index == 0: collected['embed'].append(torch.tensor(attention.input_vector))
                    collected[f'post:{index}'].append(torch.tensor(mlp.output_vector))
            self.vectors = {key: torch.stack(items) for key, items in collected.items()}
        else:
            def embedding(_module, inputs):
                self.vectors['embed'] = inputs[0][0, self.start:self.stop].detach().cpu().clone()
            def layer_output(index):
                def hook(_module, _inputs, output):
                    tensor = output[0] if isinstance(output, tuple) else output
                    self.vectors[f'post:{index}'] = tensor[0, self.start:self.stop].detach().cpu().clone()
                return hook
            try:
                handles.append(runtime.layers[0].register_forward_pre_hook(embedding))
                for index, layer in enumerate(runtime.layers):
                    handles.append(layer.register_forward_hook(layer_output(index)))
                with self.capture_heads(): self.replay._prefill(self.stop)
            finally:
                for handle in handles: handle.remove()
        for index in range(-1, len(runtime.layers)):
            key = 'embed' if index == -1 else f'post:{index}'
            if key not in self.vectors or len(self.vectors[key]) != self.stop - self.start:
                raise ValueError('Model did not expose all requested residual positions.')
            self.rows.append(self.row(key, 'Embedding' if index == -1 else f'Layer {index} · Post',
                                      index, 'embed' if index == -1 else 'post', None if index == -1 else index,
                                      self.vectors[key]))

    def result(self):
        return {'snapshotId': self.id, 'model': self.model, 'mode': self.mode,
                'lensLabel': self.selected['label'] if self.selected else 'Logit Lens',
                'targetId': self.token_id, 'target': self.execution.runtime.tokenizer.decode([self.token_id]),
                'start': self.start, 'stop': self.stop, 'tokens': list(self.tokens[self.start:self.stop]),
                'currentToken': self.current_token, 'rows': self.rows, 'interventions': self.intervened,
                'promptTokenCount': self.prompt_count, 'maxNewTokens': self.max_new_tokens,
                'generatedTokenCount': max(0, self.stop-self.prompt_count),
                'newlyGeneratedTokens': self.newly_generated, 'stoppedAtEos': self.eos_stopped,
                'generatedText': ''.join(self.tokens[self.prompt_count:]),
                'metric': 'cosine(FinalNorm(lens(vector)), target unembedding)',
                'warning': 'Lens readout, not probability or causal evidence. Embedding means the input to layer 0, including model-specific position/scaling operations.'}

    def weight(self, layer):
        runtime = self.execution.runtime
        block = runtime.layers[layer]
        if runtime.architecture == 'gpt2': return block.attn.c_proj.weight.detach()
        mixer = block.linear_attn if runtime.layer_kind(layer) == 'gated_linear_attention' else block.self_attn
        projection = getattr(mixer, 'o_proj', None) or mixer.out_proj
        return projection.weight.detach().T

    def heads(self, layer):
        import torch
        if layer in self.head_rows: return {'rows': self.head_rows[layer], 'snapshotId': self.id}
        if not 0 <= layer < len(self.execution.runtime.layers): raise ValueError('Invalid layer.')
        if layer not in self.mixed: raise ValueError('Head subrows are unavailable for this snapshot (including intervention replays).')
        runtime = self.execution.runtime
        config = getattr(runtime.model.config, 'text_config', runtime.model.config)
        delta = runtime.layer_kind(layer) == 'gated_linear_attention'
        count = int(config.linear_num_value_heads if delta else config.n_head if runtime.architecture == 'gpt2' else config.num_attention_heads)
        weight = self.weight(layer)
        width = weight.shape[0] // count
        result = []
        with _runtime_lock, torch.inference_mode(), activate(self.selected['id'] if self.selected else None):
            for head in range(count):
                for kind in ('value', 'output'):
                    key = f'{kind}:{layer}:{head}'
                    if kind == 'value' and layer not in self.values: continue
                    vector = self.head_vectors(kind, layer, head, count, width)
                    row = self.row(key, f'  Head {head} · {"Value" if kind == "value" else "Output"}', layer, 'mid', layer, vector)
                    row.update(head=head, kind=kind, parent=f'post:{layer}', delta=delta)
                    result.append(row)
            self.head_rows[layer] = result
        return {'rows': result, 'snapshotId': self.id}

    def head_vectors(self, kind, layer, head, count=None, width=None):
        import torch
        weight = self.weight(layer)
        if count is None:
            config = getattr(self.execution.runtime.model.config, 'text_config', self.execution.runtime.model.config)
            count = int(config.linear_num_value_heads if self.execution.runtime.layer_kind(layer) == 'gated_linear_attention'
                        else config.n_head if self.execution.runtime.architecture == 'gpt2' else config.num_attention_heads)
        width = weight.shape[0] // count if width is None else width
        tensor = self.values[layer] if kind == 'value' else self.mixed[layer]
        tensor = tensor.reshape(len(tensor), -1, width)
        source = head // (count // tensor.shape[1])
        with torch.inference_mode():
            return (tensor[:, source].to(device=weight.device, dtype=torch.float32) @
                    weight[head*width:(head+1)*width].float()).cpu()

    def inspect(self, row_id, position):
        import torch
        if not self.start <= position < self.stop: raise ValueError('Position is outside this snapshot.')
        if row_id.startswith(('output:', 'value:')):
            kind, layer, head = row_id.split(':'); layer, head = int(layer), int(head)
            rows = self.heads(layer)['rows']
            row = next((r for r in rows if r['id'] == row_id), None)
            if row is None: raise ValueError('Unknown head row.')
            vector = self.head_vectors(kind, layer, head)[position-self.start]
        else:
            row = next((r for r in self.rows if r['id'] == row_id), None)
            if row is None: raise ValueError('Unknown residual row.')
            vector = self.vectors[row_id][position-self.start]
        with _runtime_lock, torch.inference_mode(), activate(self.selected['id'] if self.selected else None):
            projection = self.execution.vector_lens(
                self.execution.runtime.state_name(max(0, row['layer']), self.execution.runtime.layer_kind(max(0, row['layer']))),
                tuple(vector.float().tolist()), source_site=row['site'], source_layer=row['siteLayer'], token_count=10,
                include_logit_lens=self.method == 'logit', include_j_lens=self.method == 'j',
                include_r_lens=self.method == 'r', include_jpp_lens=self.method == 'jpp')
        items = getattr(projection, {'logit':'vocabulary','j':'j_lens','r':'r_lens','jpp':'jpp_lens'}[self.method])
        return {'snapshotId': self.id, 'rowId': row_id, 'position': position, 'label': row['label'],
                'token': self.tokens[position], 'score': row['scores'][position-self.start],
                'norm': float(vector.float().norm()), 'tokens': [{'id':t.token_id,'text':t.text,'score':t.score} for t in items]}

    def edges(self, layer, head, position):
        import torch
        from llmdbg.model_runtime import delta_write_readout_contributions
        if not self.start <= position < self.stop: raise ValueError('Query position is outside this snapshot.')
        rows = self.heads(layer)['rows']
        selected = next((r for r in rows if r.get('head') == head and r.get('kind') == 'output'), None)
        if selected is None: raise ValueError('Choose an output-head point first.')
        if self.intervened: raise ValueError('Source-edge attribution is unavailable for intervention snapshots.')
        if self.execution.runtime.architecture == 'gpt2':
            raise ValueError('Source edges for GPT-2 are not yet supported; head/residual points remain available.')
        key = (layer, head, position)
        if key in self.edge_cache: return self.edge_cache[key]
        replay = replace(self.replay, past_key_values=None, cached_through=-1, capture_delta_core=True,
                         ica_observations={})
        runtime = replay.runtime
        kind = runtime.layer_kind(layer)
        width = self.weight(layer).shape[0] // len({r['head'] for r in rows})
        weight = self.weight(layer)[head*width:(head+1)*width].float()
        source = self.mapping('mid', layer, weight.shape[1])
        if source is None: raise ValueError('No fitted map for the selected head site.')
        histories = []
        original_kernels, handles = [], []
        active = [False]
        gate_output = []
        with _runtime_lock, torch.inference_mode(), activate(self.selected['id'] if self.selected else None):
            try:
                if kind == 'gated_linear_attention':
                    mixer = runtime.layers[layer].linear_attn
                    module = importlib.import_module(type(mixer).__module__)
                    def enter(_module, _args): active[0] = True
                    def leave(_module, _args, _output): active[0] = False
                    handles.extend([mixer.register_forward_pre_hook(enter), mixer.register_forward_hook(leave)])
                    def wrap(original):
                        def kernel(query, key, value, *args, **kwargs):
                            if active[0]:
                                normalized_key = module.l2norm(key.float()) if kwargs.get('use_qk_l2norm_in_kernel') else key.float()
                                histories.append((normalized_key[0, :, head].cpu(), value[0, :, head].float().cpu(),
                                                  kwargs['g'][0, :, head].float().exp().cpu(), kwargs['beta'][0, :, head].float().cpu()))
                            return original(query, key, value, *args, **kwargs)
                        return kernel
                    for name in ('torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule'):
                        original = getattr(module, name)
                        original_kernels.append((module, name, original)); setattr(module, name, wrap(original))
                elif runtime.architecture == 'qwen3_5':
                    def capture_gate(_module, _inputs, output): gate_output.append(output.detach())
                    handles.append(runtime.layers[layer].self_attn.q_proj.register_forward_hook(capture_gate))
                trace = replay.trace_token(position)
            finally:
                for module, name, original in reversed(original_kernels): setattr(module, name, original)
                for handle in handles: handle.remove()
            state = next(s for s in trace.states if s.layer.name == runtime.state_name(layer, kind))
            tensors = state.tensors
            if kind == 'gated_linear_attention':
                if sum(len(item[0]) for item in histories) != position + 1:
                    raise ValueError('Kernel did not expose a complete DeltaNet history.')
                keys, values, decays, betas = (torch.cat([item[i] for item in histories]) for i in range(4))
                contributions = delta_write_readout_contributions(keys, values, decays, betas,
                    tensors['delta_query'][0, 0, head].float().cpu()).to(weight.device)
                readout = tensors['delta_readout'][0, 0, head].float()
                if not torch.allclose(contributions.sum(0), readout, atol=1e-3, rtol=.02):
                    raise ValueError('DeltaNet readout reconstruction failed; no edges will be shown.')
                gate = tensors['delta_gate'][0, 0, head].float()
                factor = torch.rsqrt(readout.square().mean() + mixer.norm.variance_epsilon) * mixer.norm.weight.float() * torch.nn.functional.silu(gate)
                parts = (contributions * factor) @ weight
                actual = tensors['gated_delta_mixing'].flatten()[head*width:(head+1)*width].float() @ weight
                positions = list(range(position + 1))
            else:
                weights, values = tensors.get('attention_weights'), tensors.get('attention_cached_values')
                if weights is None or values is None: raise ValueError('This model does not expose source attention tensors for edges.')
                query_heads, value_heads = weights.shape[1], values.shape[1]
                kv = head // (query_heads // value_heads)
                contributions = weights[0, head, -1].float()[:, None] * values[0, kv].float()
                if runtime.architecture == 'qwen3_5':
                    gate = gate_output[-1][0, -1].reshape(query_heads, 2*width)[head, width:].float().sigmoid()
                    contributions *= gate
                parts = contributions @ weight
                actual = tensors['full_attention_mixing'].flatten()[head*width:(head+1)*width].float() @ weight
                source_positions = tensors.get('attention_source_positions')
                positions = source_positions.tolist() if source_positions is not None else list(range(len(parts)))
            error = float((parts.sum(0) - actual).norm())
            if error > .03*float(actual.norm()) + .001:
                raise ValueError('Projected head reconstruction failed; no edges will be shown.')
            displayed = self.head_vectors('output', layer, head)[position-self.start].to(actual.device)
            if float((displayed-actual).norm()) > .05*float(actual.norm()) + .002:
                raise ValueError('Replay differs from the lattice point; no edges will be shown. Narrow the range and replay.')
            support, score, bias = target_support(self.execution, parts, self.token_id, source[0])
            edge_rows = [{'position': int(p), 'token': self.tokens[int(p)], 'support': float(s),
                          'magnitude': float(part.norm())} for p, s, part in zip(positions, support, parts)]
        result = {'snapshotId': self.id, 'rowId': selected['id'], 'sourceRowId': f'value:{layer}:{head}', 'position': position,
                  'rows': edge_rows, 'targetScore': score, 'supportSum': float(support.sum()),
                  'normalizationBiasSupport': bias, 'reconstructionError': error,
                  'kind': kind, 'metric': 'Signed target-cosine support, with gates and normalization statistics frozen.',
                  'warning': 'Attribution, not a causal deletion effect. DeltaNet source positions are write positions; convolution may incorporate earlier tokens. Negative sources oppose the target direction.'}
        if len(self.edge_cache) >= 16: self.edge_cache.pop(next(iter(self.edge_cache)))
        self.edge_cache[key] = result
        return result


def dispatch(adapter, args):
    action = args.get('action', 'run')
    if action == 'candidates':
        if adapter.execution is None: raise ValueError('A paused model execution is required.')
        text = str(args.get('target', ''))
        if not text: return {'tokens': []}
        tokenizer = adapter.execution.runtime.tokenizer
        cache = getattr(adapter, '_concept_trace_vocabulary', None)
        if cache is None or cache[0] is not tokenizer:
            ids = sorted(set(tokenizer.get_vocab().values()))
            decoded = tokenizer.batch_decode([[i] for i in ids], skip_special_tokens=False)
            cache = (tokenizer, list(zip(ids, decoded)))
            adapter._concept_trace_vocabulary = cache
        query = text.casefold()
        matches = [(i, token) for i, token in cache[1] if query in token.casefold()]
        matches.sort(key=lambda item: (item[1] != text, item[1].strip().casefold() != query.strip(), not item[1].casefold().startswith(query), len(item[1]), item[0]))
        return {'tokens': [{'id': i, 'text': token} for i, token in matches[:30]]}
    if action == 'metadata': return metadata(adapter)
    if action == 'tokenize':
        if adapter.execution is None: raise ValueError('A paused model execution is required.')
        return {'tokens': target_tokens(adapter.execution, args.get('target'))}
    if action == 'run':
        snapshot = Snapshot(adapter, args)
        adapter._concept_trace_snapshot = snapshot
        result = snapshot.result()
        result['images'] = adapter._prompt_images() if hasattr(adapter, '_prompt_images') else []
        result['sourcePath'] = getattr(adapter, 'source_path', None)
        return result
    snapshot = getattr(adapter, '_concept_trace_snapshot', None)
    if snapshot is None or args.get('snapshotId') != snapshot.id:
        raise ValueError('This Concept Trace snapshot has expired. Replay to make a new snapshot.')
    if action == 'heads': return snapshot.heads(int(args.get('layer', -1)))
    if action == 'inspect': return snapshot.inspect(str(args.get('rowId', '')), int(args.get('position', -1)))
    if action == 'edges': return snapshot.edges(int(args.get('layer', -1)), int(args.get('head', -1)), int(args.get('position', -1)))
    raise ValueError('Unknown Concept Trace action.')
