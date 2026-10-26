"""Capture inspectable layer states from Hugging Face causal language models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import RLock
from typing import Any

from llmdbg.demo_model import Layer, LayerState


@dataclass(frozen=True)
class ModelTrace:
    """Captured model states plus the resolved prompt tokens."""

    states: tuple[LayerState, ...]
    tokens: tuple[str, ...]


@dataclass
class ModelRuntime:
    """A tokenizer and GPU-resident model reusable across debug sessions."""

    model: Any
    tokenizer: Any
    processor: Any = None

    @property
    def is_multimodal(self) -> bool:
        return hasattr(getattr(self.model, "model", None), "language_model")

    @property
    def text_model(self) -> Any:
        if self.is_multimodal:
            return self.model.model.language_model
        return self.model.transformer if self.architecture == "gpt2" else self.model.model

    @property
    def architecture(self) -> str:
        model_type = getattr(self.model.config, "model_type", "")
        if model_type in {"qwen3_vl", "qwen3_vl_text"}:
            return "qwen3_vl"
        if model_type in {"gemma4", "gemma4_text"}:
            return "gemma4"
        return "qwen3_5" if model_type in {"qwen3_5", "qwen3_5_text"} else model_type

    @property
    def visual_layer_count(self) -> int:
        vision = getattr(self.model.config, "vision_config", None)
        return len(getattr(vision, "deepstack_visual_indexes", ())) if self.architecture == "qwen3_vl" else 0

    @property
    def has_per_layer_inputs(self) -> bool:
        return self.architecture == "gemma4" and bool(self.text_model.config.hidden_size_per_layer_input)

    @property
    def layers(self) -> Any:
        return self.text_model.h if self.architecture == "gpt2" else self.text_model.layers

    @property
    def final_norm(self) -> Any:
        return self.text_model.ln_f if self.architecture == "gpt2" else self.text_model.norm

    def layer_kind(self, index: int) -> str:
        if self.architecture == "gpt2":
            return "attention"
        if self.architecture in {"gemma2", "gemma4", "qwen3_vl"}:
            return "full_attention"
        config = getattr(self.text_model, "config", self.model.config)
        kind = config.layer_types[index]
        return "gated_linear_attention" if kind == "linear_attention" else "full_attention"

    def state_name(self, index: int, kind: str) -> str:
        if self.architecture == "gpt2":
            return f"transformer.h.{index}.{kind}"
        return f"model.layers.{index}.{kind}"

    @property
    def final_norm_name(self) -> str:
        return "transformer.ln_f" if self.architecture == "gpt2" else "model.norm"


@dataclass(frozen=True)
class TokenScore:
    """One vocabulary token and its score for a neuron output direction."""

    token_id: int
    text: str
    score: float


@dataclass(frozen=True)
class TokenPrediction:
    """One next-token prediction from the model's final vocabulary logits."""

    token_id: int
    text: str
    logit: float
    probability: float


@dataclass(frozen=True)
class NeuronProjection:
    """An activated MLP neuron and its vocabulary readouts."""

    index: int
    activation: float
    direction_norm: float
    vocabulary: tuple[TokenScore, ...]
    contribution: tuple[TokenScore, ...]
    j_lens: tuple[TokenScore, ...] = ()
    j_lens_metadata: dict[str, Any] | None = None
    r_lens: tuple[TokenScore, ...] = ()
    r_lens_metadata: dict[str, Any] | None = None
    jpp_lens: tuple[TokenScore, ...] = ()
    jpp_lens_metadata: dict[str, Any] | None = None
    output_projection: float = 0.0
    residual_norm: float = 0.0
    residual_cosine: float | None = None
    potential_norm: float = 0.0
    vector: tuple[float, ...] = ()


@dataclass(frozen=True)
class VectorProjection:
    """Vocabulary readouts for one model-width vector."""

    norm: float
    vocabulary: tuple[TokenScore, ...]
    j_lens: tuple[TokenScore, ...] = ()
    j_lens_metadata: dict[str, Any] | None = None
    r_lens: tuple[TokenScore, ...] = ()
    r_lens_metadata: dict[str, Any] | None = None
    jpp_lens: tuple[TokenScore, ...] = ()
    jpp_lens_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ICAProjection:
    """One active ICA component and vocabulary readouts for its signed vector."""

    index: int
    rank: int
    score: float
    energy: float
    projection: VectorProjection
    vector: tuple[float, ...] = ()


def delta_write_readout_contributions(keys: Any, values: Any, decays: Any, betas: Any, query: Any) -> Any:
    """Exact zero-initial-memory decomposition, including subsequent overwrites.

    Backward adjoint scan avoids storing a 128x128 matrix per source position.
    Q/K/V, gates and decays are held fixed; this is not a causal deletion effect.
    """
    import torch
    if keys.ndim != 2 or values.ndim != 2 or len(keys) != len(values) or query.shape != keys.shape[1:]:
        raise ValueError('Invalid DeltaNet history dimensions')
    if decays.shape != (len(keys),) or betas.shape != (len(keys),):
        raise ValueError('Invalid DeltaNet decay/beta history dimensions')
    adjoint = query.float().clone()
    result = torch.empty_like(values, dtype=torch.float32)
    for source in range(len(keys)-1, -1, -1):
        key, value = keys[source].float(), values[source].float()
        match = adjoint @ key
        result[source] = betas[source] * match * value
        adjoint = decays[source] * (adjoint - betas[source] * match * key)
    return result


@dataclass
class ModelExecution:
    """A prompt execution that advances one token at a time with a KV cache."""

    runtime: ModelRuntime
    input_ids: Any
    tokens: tuple[str, ...]
    device: str
    progress: Callable[[str], None]
    dtype: str = "float32"
    rendered_prompt: str = ""
    inputs_embeds: Any = None
    position_ids: Any = None
    image_grids: tuple[tuple[int, int, int], ...] = ()
    image_merge_size: int = 1
    visual_pos_masks: Any = None
    deepstack_visual_embeds: Any = None
    per_layer_inputs: Any = None
    past_key_values: Any = None
    cached_through: int = -1
    interventions: dict[tuple[int, str, str, int], float] = field(default_factory=dict)
    ica_interventions: set[tuple[int, str, str, int]] = field(default_factory=set)
    vector_patches: dict[tuple[int, str, str], tuple[float, ...]] = field(
        default_factory=dict
    )
    ica_observations: dict[
        tuple[int, str, str, int], tuple[tuple[float, ...], tuple[float, ...]]
    ] = field(default_factory=dict)
    projection_norms: dict[tuple[int, tuple[int, ...]], Any] = field(
        default_factory=dict
    )
    semantic_token_masks: dict[tuple[int, str], Any] = field(default_factory=dict)
    capture_delta_core: bool = False

    def delta_memory_contributions(self, layer: int, head: int, position: int) -> dict[str, Any]:
        """On-demand replay with an independent cache; retain only one head's history."""
        import hashlib
        import torch

        if self.runtime.architecture != 'qwen3_5' or not 0 <= layer < len(self.runtime.layers):
            raise ValueError('Memory contributions require a Qwen3.5 DeltaNet layer')
        mixer = getattr(self.runtime.layers[layer], 'linear_attn', None)
        if mixer is None or not 0 <= head < mixer.num_v_heads or not 0 <= position < self.token_count:
            raise ValueError('Invalid DeltaNet layer, head, or query position')
        with _runtime_lock, torch.inference_mode():
            digest = hashlib.sha256()
            def fingerprint(value):
                if isinstance(value, torch.Tensor):
                    digest.update(str((tuple(value.shape), value.dtype)).encode())
                    digest.update(value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
                elif isinstance(value, dict):
                    for key in sorted(value, key=repr): fingerprint(key); fingerprint(value[key])
                elif isinstance(value, (list, tuple)):
                    digest.update(str((type(value).__name__, len(value))).encode())
                    for item in value: fingerprint(item)
                else: digest.update(repr((type(value).__name__, value)).encode())
            for value in (self.input_ids, self.inputs_embeds, self.position_ids, self.visual_pos_masks,
                          self.deepstack_visual_embeds, self.per_layer_inputs, self.interventions,
                          self.ica_interventions, self.vector_patches, self.tokens,
                          getattr(self.runtime.model.config, '_commit_hash', None)):
                fingerprint(value)
            key = (layer, head, position, digest.hexdigest())
            cached = getattr(self, '_delta_contribution_result', None)
            if cached is not None and cached[0] == key:
                return cached[1]
            replay = replace(self, past_key_values=None, cached_through=-1, capture_delta_core=True,
                             inputs_embeds=self.inputs_embeds.clone() if self.inputs_embeds is not None else None,
                             interventions=dict(self.interventions), ica_interventions=dict(self.ica_interventions),
                             vector_patches=dict(self.vector_patches), ica_observations={})
            name = f'model.layers.{layer}.gated_linear_attention'
            keys, values, decays, betas = [], [], [], []
            self.progress(f'Replaying tokens 0..{position} for DeltaNet layer {layer}, head {head}; debugger cache is unchanged.\n')
            if self.interventions or self.ica_interventions or self.vector_patches:
                # Apply each historical intervention at its recorded token.
                for source in range(position + 1):
                    trace = replay.trace_token(source)
                    state = next(s for s in trace.states if s.layer.name == name)
                    tensors = state.tensors
                    keys.append(tensors['delta_key'][0, 0, head].float().cpu())
                    values.append(tensors['delta_value'][0, 0, head].float().cpu())
                    decays.append(float(tensors['delta_decay'][0, 0, head]))
                    betas.append(float(tensors['delta_beta'][0, 0, head]))
            else:
                # Capture only this head at the actual kernel boundary, including
                # the batched prefill. No per-position memory snapshots are saved.
                import importlib
                implementation = importlib.import_module(type(mixer).__module__)
                selected = [False]
                def enter(_module, _args): selected[0] = True
                def leave(_module, _args, _output): selected[0] = False
                handles = [mixer.register_forward_pre_hook(enter), mixer.register_forward_hook(leave)]
                originals = []
                def capture(original):
                    def wrapped(query, key, value, *args, **kwargs):
                        if selected[0]:
                            normalized_key = implementation.l2norm(key.float()) if kwargs.get('use_qk_l2norm_in_kernel') else key.float()
                            keys.extend(normalized_key[0, :, head].cpu().unbind())
                            values.extend(value[0, :, head].detach().float().cpu().unbind())
                            decays.extend(kwargs['g'][0, :, head].float().exp().cpu().tolist())
                            betas.extend(kwargs['beta'][0, :, head].float().cpu().tolist())
                        return original(query, key, value, *args, **kwargs)
                    return wrapped
                try:
                    for kernel in ('torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule'):
                        original = getattr(implementation, kernel)
                        originals.append((kernel, original))
                        setattr(implementation, kernel, capture(original))
                    trace = replay.trace_token(position)
                finally:
                    for kernel, original in reversed(originals): setattr(implementation, kernel, original)
                    for handle in handles: handle.remove()
                state = next(s for s in trace.states if s.layer.name == name)
                tensors = state.tensors
            if len(keys) != position + 1:
                raise ValueError('DeltaNet kernel did not expose the complete source history for this backend')
            query = tensors['delta_query'][0, 0, head].float().cpu()
            contributions = delta_write_readout_contributions(
                torch.stack(keys), torch.stack(values), torch.tensor(decays), torch.tensor(betas), query)
            readout = tensors['delta_readout'][0, 0, head].float()
            reconstructed = contributions.sum(0).to(readout.device)
            if not torch.allclose(reconstructed, readout, atol=1e-3, rtol=.02):
                raise ValueError('Memory attribution cannot reconstruct this readout; captured-state interventions or unsupported kernels may alter the recurrence')
            # Freeze the actual RMS denominator, keeping an additive decomposition.
            gate = tensors['delta_gate'][0, 0, head].float()
            factor = torch.rsqrt(readout.square().mean() + mixer.norm.variance_epsilon)
            factor = factor * mixer.norm.weight.detach().float() * torch.nn.functional.silu(gate)
            width = mixer.head_v_dim
            weight = mixer.out_proj.weight.detach().T[head * width:(head + 1) * width].float()
            projected = (contributions.to(weight.device) * factor) @ weight
            actual = tensors['gated_delta_mixing'].flatten()[head * width:(head + 1) * width].float() @ weight
            summed = projected.sum(0)
            error = float((summed - actual).norm())
            if error > .03 * float(actual.norm()) + .001:
                raise ValueError('Memory attribution cannot reconstruct the gated projected head output; output interventions may be active')
            norm = float(actual.norm())
            support = projected @ (actual / max(norm, 1e-30))
            magnitudes = projected.norm(dim=-1)
            rows = [{'position':s, 'token':self.tokens[s], 'contribution':float(support[s]),
                     'magnitude':float(magnitudes[s]), 'beta':betas[s], 'decay':decays[s]}
                    for s in range(position + 1)]
            result = {'kind':'delta-memory', 'head':head, 'headDimension':width, 'rows':rows,
                      'sum':float(support.sum()), 'positiveSum':sum(max(0., r['contribution']) for r in rows),
                      'negativeSum':sum(min(0., r['contribution']) for r in rows),
                      'outputNorm':norm, 'reconstructionError':error,
                      'outputVector':tuple(actual.cpu().tolist()),
                      'readoutReconstructionError':float((reconstructed-readout).norm())}
            self._delta_contribution_result = (key, result)  # Bound cache to one snapshot.
            return result

    def _semantic_token_mask(self, device: Any, vocabulary_size: int) -> Any:
        """Return the J++ readout mask: decoded tokens containing a letter/digit."""
        import torch

        key = (vocabulary_size, str(device))
        mask = self.semantic_token_masks.get(key)
        if mask is None:
            self.progress(
                f"Building the J++ semantic filter for {vocabulary_size} vocabulary tokens...\n"
            )
            mask = torch.tensor(
                [
                    any(
                        character.isalnum()
                        for character in self.runtime.tokenizer.decode([token_id])
                    )
                    for token_id in range(vocabulary_size)
                ],
                dtype=torch.bool,
                device=device,
            )
            self.semantic_token_masks[key] = mask
            self.progress("Built the J++ semantic vocabulary filter.\n")
        return mask

    @property
    def token_count(self) -> int:
        return len(self.tokens)

    def trace_token(self, position: int) -> ModelTrace:
        """Prefill up to ``position`` and capture one token's layer trace."""
        if position < 0 or position >= self.token_count:
            raise ValueError(
                f"token position {position} is outside the prompt range 0..{self.token_count - 1}"
            )
        self.progress(
            f"Preparing token {position}; waiting for exclusive access to the resident model.\n"
        )
        with _runtime_lock:
            self.progress(f"Resident model lock acquired for token {position}.\n")
            if position <= self.cached_through:
                self.progress(f"Resetting KV cache before token {position}.\n")
                self.past_key_values = None
                self.cached_through = -1
            if position > self.cached_through + 1:
                self._prefill(position)
            trace, next_cache = _capture_token_trace(
                self.runtime,
                input_ids=self.input_ids[:, position : position + 1],
                past_key_values=self.past_key_values,
                total_length=position + 1,
                device=self.device,
                progress=self.progress,
                token_position=position,
                interventions=self.interventions,
                ica_interventions=self.ica_interventions,
                ica_observations=self.ica_observations,
                vector_patches=self.vector_patches,
                inputs_embeds=(
                    self.inputs_embeds[:, position : position + 1]
                    if self.inputs_embeds is not None else None
                ),
                position_ids=(
                    self.position_ids[..., position : position + 1]
                    if self.position_ids is not None else None
                ),
                **self._visual_slice(position, position + 1),
                **({"capture_delta_core": self.capture_delta_core} if self.runtime.architecture == "qwen3_5" else {}),
            )
            self.past_key_values = next_cache
            self.cached_through = position
            return ModelTrace(trace.states, self.tokens)

    def append_generated_token(self, token_id: int) -> str:
        """Append one greedy next token while preserving the sequential KV cache."""
        import torch

        token = torch.tensor(
            [[int(token_id)]], dtype=self.input_ids.dtype, device=self.input_ids.device
        )
        self.input_ids = torch.cat((self.input_ids, token), dim=1)
        if self.inputs_embeds is not None:
            embedding = self.runtime.text_model.get_input_embeddings()(
                token.to(self.device)
            )
            self.inputs_embeds = torch.cat((self.inputs_embeds, embedding), dim=1)
            next_position = self.position_ids[..., -1:] + 1
            self.position_ids = torch.cat((self.position_ids, next_position), dim=-1)
            if self.per_layer_inputs is not None:
                per_layer = self.runtime.text_model.get_per_layer_inputs(token.to(self.device), None)
                self.per_layer_inputs = torch.cat((self.per_layer_inputs, per_layer), dim=1)
            if self.visual_pos_masks is not None:
                self.visual_pos_masks = torch.cat((self.visual_pos_masks, torch.zeros(
                    (1, 1), dtype=torch.bool, device=self.visual_pos_masks.device
                )), dim=1)
        text = self.runtime.tokenizer.decode([int(token_id)])
        self.tokens = (*self.tokens, text)
        return text

    def vocabulary_projection(
        self, hidden: tuple[float, ...], *, count: int = 10
    ) -> tuple[tuple[float, ...], tuple[TokenPrediction, ...]]:
        """Project one final hidden state to logits and decode its top predictions."""
        import torch

        with _runtime_lock, torch.inference_mode():
            weight = self.runtime.model.get_output_embeddings().weight
            vector = torch.tensor(hidden, device=weight.device, dtype=weight.dtype)
            logits = torch.mv(weight, vector)
            config = getattr(self.runtime.model, "config", None)
            cap = getattr(getattr(config, "text_config", config), "final_logit_softcapping", None)
            if cap is not None:
                logits = torch.tanh(logits / cap) * cap
            logits = logits.float()
            values, ids = torch.topk(logits, min(count, logits.numel()))
            probabilities = torch.softmax(logits, dim=0).index_select(0, ids)
            predictions = tuple(
                TokenPrediction(
                    int(token_id),
                    self.runtime.tokenizer.decode([int(token_id)]),
                    float(logit),
                    float(probability),
                )
                for token_id, logit, probability in zip(
                    ids.tolist(), values.tolist(), probabilities.tolist(), strict=True
                )
            )
            return tuple(logits.cpu().tolist()), predictions

    def set_intervention(
        self,
        token_position: int,
        layer_name: str,
        vector_name: str,
        index: int,
        value: float,
    ) -> None:
        """Persist one vector override for subsequent replays of a token."""
        self.interventions[(token_position, layer_name, vector_name, index)] = value

    def remove_ica_component(
        self,
        token_position: int,
        layer_name: str,
        vector_name: str,
        component: int,
    ) -> None:
        """Persist an ICA component ablation for subsequent token replays."""
        self.ica_interventions.add(
            (token_position, layer_name, vector_name, component)
        )

    def patch_vector(
        self,
        token_position: int,
        layer_name: str,
        vector_name: str,
        vector: tuple[float, ...],
    ) -> None:
        """Replace one model-width vector at a computation boundary on replay."""
        self.vector_patches[(token_position, layer_name, vector_name)] = vector

    def _visual_slice(self, start: int, stop: int) -> dict[str, Any]:
        result = {}
        if self.per_layer_inputs is not None:
            result["per_layer_inputs"] = self.per_layer_inputs[:, start:stop]
        if self.visual_pos_masks is None or self.deepstack_visual_embeds is None:
            return result
        first = int(self.visual_pos_masks[:, :start].sum().item())
        last = int(self.visual_pos_masks[:, :stop].sum().item())
        return {**result, "visual_pos_masks": self.visual_pos_masks[:, start:stop],
                "deepstack_visual_embeds": [feature[first:last] for feature in self.deepstack_visual_embeds]}

    def _prefill(self, position: int) -> None:
        """Run uncaptured prompt tokens into the cache before a target token."""
        import torch

        start = self.cached_through + 1
        prefix = self.input_ids[:, start:position].to(self.device)
        if prefix.shape[1] == 0:
            return
        self.progress(f"Prefilling tokens {start}..{position - 1} into the KV cache.\n")
        attention_mask = torch.ones(
            (1, position), dtype=torch.long, device=self.device
        )
        with torch.inference_mode():
            model = (
                self.runtime.text_model if self.inputs_embeds is not None
                else self.runtime.model
            )
            output = model(
                input_ids=None if self.inputs_embeds is not None else prefix,
                inputs_embeds=(
                    self.inputs_embeds[:, start:position]
                    if self.inputs_embeds is not None else None
                ),
                position_ids=(
                    self.position_ids[..., start:position]
                    if self.position_ids is not None else None
                ),
                attention_mask=attention_mask,
                past_key_values=self.past_key_values,
                use_cache=True,
                **self._visual_slice(start, position),
            )
        self.past_key_values = output.past_key_values
        self.cached_through = position - 1

    def top_mlp_neurons(
        self,
        layer_name: str,
        activation: tuple[float, ...],
        *,
        output: tuple[float, ...] | None = None,
        residual: tuple[float, ...] | None = None,
        neuron_count: int = 5,
        token_count: int = 5,
        include_logit_lens: bool = True,
        include_j_lens: bool = False,
        include_r_lens: bool = False,
        include_jpp_lens: bool = False,
        include_indices: tuple[int, ...] = (),
    ) -> tuple[NeuronProjection, ...]:
        """Project active MLP neurons through direct, J-Lens, and R-Lens readouts."""
        import re

        import torch

        match = re.fullmatch(r"(?:transformer\.h|model\.layers)\.(\d+)\.mlp", layer_name)
        if match is None:
            return ()
        layer_index = int(match.group(1))
        block = self.runtime.layers[layer_index]
        down_projection = (
            block.mlp.c_proj.weight
            if self.runtime.architecture == "gpt2"
            else block.mlp.down_proj.weight.T
        )
        return self._top_projected_components(
            layer_index, activation, down_projection,
            source_site="post",
            output=output,
            residual=residual,
            component_count=neuron_count, token_count=token_count,
            include_logit_lens=include_logit_lens,
            include_j_lens=include_j_lens, include_r_lens=include_r_lens,
            include_jpp_lens=include_jpp_lens,
            include_indices=include_indices,
        )

    def top_attention_channels(
        self,
        layer_name: str,
        activation: tuple[float, ...],
        *,
        output: tuple[float, ...] | None = None,
        residual: tuple[float, ...] | None = None,
        channel_count: int = 5,
        head_index: int | None = None,
        token_count: int = 5,
        include_logit_lens: bool = True,
        include_j_lens: bool = False,
        include_r_lens: bool = False,
        include_jpp_lens: bool = False,
        include_indices: tuple[int, ...] = (),
    ) -> tuple[NeuronProjection, ...]:
        """Project active attention channels through W_O and vocabulary readouts."""
        import re

        match = re.fullmatch(
            r"(?:transformer\.h|model\.layers)\.(\d+)\."
            r"(?:attention|full_attention|gated_linear_attention)",
            layer_name,
        )
        if match is None:
            return ()
        layer_index = int(match.group(1))
        block = self.runtime.layers[layer_index]
        if self.runtime.architecture == "gpt2":
            projection = block.attn.c_proj.weight
        else:
            mixer = (
                block.self_attn
                if layer_name.endswith(".full_attention")
                else block.linear_attn
            )
            projection = (
                mixer.o_proj.weight.T
                if hasattr(mixer, "o_proj")
                else mixer.out_proj.weight.T
            )
        channel_range = None
        if head_index is not None:
            config = getattr(self.runtime.model.config, 'text_config', self.runtime.model.config)
            heads = int(config.linear_num_value_heads if layer_name.endswith('.gated_linear_attention')
                        else config.n_head if self.runtime.architecture == 'gpt2' else config.num_attention_heads)
            if not 0 <= head_index < heads or len(activation) % heads:
                raise ValueError('Invalid attention head index or channel layout')
            width = len(activation) // heads
            channel_range = (head_index * width, (head_index + 1) * width)
        return self._top_projected_components(
            layer_index, activation, projection,
            channel_range=channel_range,
            source_site="mid",
            output=output,
            residual=residual,
            component_count=channel_count, token_count=token_count,
            include_logit_lens=include_logit_lens,
            include_j_lens=include_j_lens, include_r_lens=include_r_lens,
            include_jpp_lens=include_jpp_lens,
            include_indices=include_indices,
        )

    def component_fan_geometry(
        self,
        layer_name: str,
        activation: tuple[float, ...],
        output: tuple[float, ...],
        *,
        component_count: int = 5,
    ) -> dict[str, Any] | None:
        """Project the strongest component updates into the top-1/output plane.

        Coordinates are measured in units of the strongest component's norm, so
        the top-1 vector is always (1, 0).  The other component vectors retain
        their measured relative lengths and directions in that plane.
        """
        import re

        import torch

        match = re.fullmatch(
            r"(?:transformer\.h|model\.layers)\.(\d+)\."
            r"(mlp|attention|full_attention|gated_linear_attention)",
            layer_name,
        )
        if match is None:
            return None
        layer_index = int(match.group(1))
        operation = match.group(2)
        block = self.runtime.layers[layer_index]
        if operation == "mlp":
            projection = (
                block.mlp.c_proj.weight
                if self.runtime.architecture == "gpt2"
                else block.mlp.down_proj.weight.T
            )
        elif self.runtime.architecture == "gpt2":
            projection = block.attn.c_proj.weight
        else:
            mixer = (
                block.self_attn
                if operation == "full_attention"
                else block.linear_attn
            )
            projection = (
                mixer.o_proj.weight.T
                if hasattr(mixer, "o_proj")
                else mixer.out_proj.weight.T
            )

        with _runtime_lock, torch.inference_mode():
            coefficients = torch.tensor(
                activation, dtype=projection.dtype, device=projection.device
            )
            result = torch.tensor(
                output, dtype=torch.float32, device=projection.device
            )
            result_norm = torch.linalg.vector_norm(result)
            if float(result_norm) == 0.0:
                return None
            output_unit = result / result_norm
            output_projections = coefficients.float() * (
                projection.float() @ output_unit
            )
            count = min(component_count, coefficients.numel())
            if count == 0:
                return None
            _, indices = torch.topk(output_projections, count)
            vectors = (
                projection.index_select(0, indices).float()
                * coefficients.index_select(0, indices).float().unsqueeze(1)
            )
            top_norm = torch.linalg.vector_norm(vectors[0])
            if float(top_norm) == 0.0:
                return None
            e1 = vectors[0] / top_norm
            perpendicular = result - torch.dot(result, e1) * e1
            perpendicular_norm = torch.linalg.vector_norm(perpendicular)
            if float(perpendicular_norm) <= 1e-8:
                for candidate in vectors[1:]:
                    perpendicular = candidate - torch.dot(candidate, e1) * e1
                    perpendicular_norm = torch.linalg.vector_norm(perpendicular)
                    if float(perpendicular_norm) > 1e-8:
                        break
            e2 = (
                perpendicular / perpendicular_norm
                if float(perpendicular_norm) > 1e-8
                else torch.zeros_like(e1)
            )

            def coordinates(vector: Any) -> dict[str, float]:
                return {
                    "x": float(torch.dot(vector, e1) / top_norm),
                    "y": float(torch.dot(vector, e2) / top_norm),
                }

            return {
                "topIndex": int(indices[0]),
                "topNorm": float(top_norm),
                "result": coordinates(result),
                "components": [
                    {"index": int(index), **coordinates(vector)}
                    for index, vector in zip(
                        indices[1:].tolist(), vectors[1:], strict=True
                    )
                ],
            }

    def _top_projected_components(
        self,
        layer_index: int,
        activation: tuple[float, ...],
        projection: Any,
        *,
        source_site: str = "post",
        component_count: int,
        token_count: int,
        include_logit_lens: bool,
        include_j_lens: bool,
        include_r_lens: bool,
        include_jpp_lens: bool = False,
        output: tuple[float, ...] | None = None,
        residual: tuple[float, ...] | None = None,
        include_indices: tuple[int, ...] = (),
        channel_range: tuple[int, int] | None = None,
    ) -> tuple[NeuronProjection, ...]:
        """Read individual activated input channels through a linear projection."""
        import torch

        unembedding = self.runtime.model.get_output_embeddings().weight
        model_id = str(getattr(self.runtime.model.config, "_name_or_path", ""))
        hidden_size = int(projection.shape[1])
        j_lens_source = (
            self._workspace_lens_source("j", model_id, source_site, layer_index, hidden_size)
            if include_j_lens else None
        )

        r_lens_source = (
            self._workspace_lens_source("r", model_id, source_site, layer_index, hidden_size)
            if include_r_lens
            else None
        )
        jpp_lens_source = (
            self._workspace_lens_source("jpp", model_id, source_site, layer_index, hidden_size)
            if include_jpp_lens
            else None
        )
        with _runtime_lock, torch.inference_mode():
            activation_tensor = torch.tensor(
                activation,
                dtype=projection.dtype,
                device=projection.device,
            )
            norm_key = (int(projection.data_ptr()), tuple(projection.shape))
            all_direction_norms = self.projection_norms.get(norm_key)
            if all_direction_norms is None:
                all_direction_norms = torch.linalg.vector_norm(
                    projection, ord=2, dim=1, dtype=torch.float32
                )
                self.projection_norms[norm_key] = all_direction_norms
            if output is None:
                ranking_scores = activation_tensor.abs().float() * all_direction_norms
                all_output_projections = torch.zeros_like(ranking_scores)
            else:
                output_tensor = torch.tensor(
                    output, dtype=torch.float32, device=projection.device
                )
                output_norm = torch.linalg.vector_norm(output_tensor)
                if float(output_norm) == 0.0:
                    ranking_scores = activation_tensor.abs().float() * all_direction_norms
                    all_output_projections = torch.zeros_like(ranking_scores)
                else:
                    output_unit = output_tensor / output_norm
                    all_output_projections = activation_tensor.float() * (
                        projection.float() @ output_unit
                    )
                    ranking_scores = all_output_projections
            start, stop = channel_range or (0, activation_tensor.numel())
            _, active_indices = torch.topk(ranking_scores[start:stop], min(component_count, stop-start))
            active_indices += start
            selected = active_indices.tolist()
            selected.extend(index for index in include_indices if index not in selected and (channel_range is None or start <= index < stop))
            if any(index < 0 or index >= activation_tensor.numel() for index in selected):
                raise ValueError(
                    f"component index is outside range 0..{activation_tensor.numel() - 1}"
                )
            active_indices = torch.tensor(
                selected, dtype=torch.long, device=activation_tensor.device
            )
            active_values = activation_tensor.index_select(0, active_indices)
            directions = projection.index_select(0, active_indices)
            vocabulary_values = vocabulary_ids = None
            contribution_values = contribution_ids = None
            if include_logit_lens:
                vocabulary_scores = (directions @ unembedding.T).float()
                contribution_scores = (
                    vocabulary_scores * active_values.float().unsqueeze(1)
                )
                vocabulary_values, vocabulary_ids = torch.topk(
                    vocabulary_scores,
                    min(token_count, vocabulary_scores.shape[1]),
                    dim=1,
                )
                contribution_values, contribution_ids = torch.topk(
                    contribution_scores,
                    min(token_count, contribution_scores.shape[1]),
                    dim=1,
                )
            direction_norms = all_direction_norms.index_select(0, active_indices)
            output_projections = all_output_projections.index_select(0, active_indices)
            component_vectors = directions.float() * active_values.float().unsqueeze(1)
            residual_norm = 0.0
            residual_cosines: list[float | None] = [None] * active_indices.shape[0]
            potential_norms = [0.0] * active_indices.shape[0]
            if residual is not None:
                residual_tensor = torch.tensor(
                    residual, dtype=torch.float32, device=projection.device
                )
                residual_norm_tensor = torch.linalg.vector_norm(residual_tensor)
                residual_norm = float(residual_norm_tensor)
                component_norms = torch.linalg.vector_norm(component_vectors, dim=1)
                dots = component_vectors @ residual_tensor
                potential = component_vectors + residual_tensor.unsqueeze(0)
                potential_norms = torch.linalg.vector_norm(potential, dim=1).tolist()
                if residual_norm > 0.0:
                    residual_cosines = [
                        (
                            float(dot / (residual_norm_tensor * component_norm))
                            if float(component_norm) > 0.0
                            else None
                        )
                        for dot, component_norm in zip(
                            dots, component_norms, strict=True
                        )
                    ]
            def transported_scores(
                source: Any, *, semantic_only: bool = False
            ) -> tuple[Any, Any]:
                if source is None:
                    return None, None
                mapping, _ = source
                mapping = mapping.to(device=directions.device, dtype=torch.float32)
                mapped = torch.einsum("oi,bi->bo", mapping, directions.float())
                normalized = self.runtime.final_norm(
                    mapped.to(dtype=projection.dtype)[:, None, :]
                )[:, 0]
                scores = (normalized @ unembedding.T).float()
                if semantic_only:
                    scores.masked_fill_(
                        ~self._semantic_token_mask(
                            scores.device, scores.shape[-1]
                        ).unsqueeze(0),
                        float("-inf"),
                    )
                values, ids = torch.topk(
                    scores, min(token_count, scores.shape[1]), dim=1
                )
                return values, ids

            j_lens_values, j_lens_ids = transported_scores(j_lens_source)
            r_lens_values, r_lens_ids = transported_scores(r_lens_source)
            jpp_lens_values, jpp_lens_ids = transported_scores(
                jpp_lens_source, semantic_only=True
            )

        def token_scores(ids: Any, values: Any) -> tuple[TokenScore, ...]:
            return tuple(
                TokenScore(
                    int(token_id),
                    self.runtime.tokenizer.decode([int(token_id)]),
                    float(score),
                )
                for token_id, score in zip(ids.tolist(), values.tolist(), strict=True)
            )

        return tuple(
            NeuronProjection(
                index=int(active_indices[rank]),
                activation=float(active_values[rank]),
                direction_norm=float(direction_norms[rank]),
                vocabulary=(
                    token_scores(vocabulary_ids[rank], vocabulary_values[rank])
                    if vocabulary_ids is not None and vocabulary_values is not None
                    else ()
                ),
                contribution=(
                    token_scores(contribution_ids[rank], contribution_values[rank])
                    if contribution_ids is not None and contribution_values is not None
                    else ()
                ),
                j_lens=(
                    token_scores(j_lens_ids[rank], j_lens_values[rank])
                    if j_lens_ids is not None and j_lens_values is not None
                    else ()
                ),
                j_lens_metadata=(j_lens_source[1] if j_lens_source else None),
                r_lens=(
                    token_scores(r_lens_ids[rank], r_lens_values[rank])
                    if r_lens_ids is not None and r_lens_values is not None
                    else ()
                ),
                r_lens_metadata=(r_lens_source[1] if r_lens_source else None),
                jpp_lens=(
                    token_scores(jpp_lens_ids[rank], jpp_lens_values[rank])
                    if jpp_lens_ids is not None and jpp_lens_values is not None
                    else ()
                ),
                jpp_lens_metadata=(jpp_lens_source[1] if jpp_lens_source else None),
                output_projection=float(output_projections[rank]),
                residual_norm=residual_norm,
                residual_cosine=residual_cosines[rank],
                potential_norm=float(potential_norms[rank]),
                vector=tuple(float(value) for value in component_vectors[rank].tolist()),
            )
            for rank in range(active_indices.shape[0])
        )

    def vector_lens(
        self,
        layer_name: str,
        vector: tuple[float, ...],
        *,
        source_site: str = "post",
        source_layer: int | None = None,
        token_count: int = 10,
        include_logit_lens: bool = True,
        include_j_lens: bool = False,
        include_r_lens: bool = False,
        include_jpp_lens: bool = False,
    ) -> VectorProjection:
        """Read one MLP/residual vector directly and through J/R transport maps."""
        return self._vectors_lens(
            layer_name,
            (vector,),
            source_site=source_site,
            source_layer=source_layer,
            token_count=token_count,
            include_logit_lens=include_logit_lens,
            include_j_lens=include_j_lens,
            include_r_lens=include_r_lens,
            include_jpp_lens=include_jpp_lens,
        )[0]

    def _vectors_lens(
        self,
        layer_name: str,
        vectors: tuple[tuple[float, ...], ...],
        *,
        source_site: str = "post",
        source_layer: int | None = None,
        token_count: int,
        include_logit_lens: bool,
        include_j_lens: bool,
        include_r_lens: bool,
        include_jpp_lens: bool = False,
    ) -> tuple[VectorProjection, ...]:
        """Batch model-width vectors through selected vocabulary readouts."""
        import re

        import torch

        match = re.match(r"(?:transformer\.h|model\.layers)\.(\d+)(?:\.|$)", layer_name)
        layer_index = int(match.group(1)) if match is not None else None
        if source_layer is None and source_site != "embed":
            source_layer = layer_index
        model_id = str(getattr(self.runtime.model.config, "_name_or_path", ""))
        if not vectors:
            return ()
        hidden_size = len(vectors[0])
        j_source = (
            self._workspace_lens_source("j", model_id, source_site, source_layer, hidden_size)
            if include_j_lens else None
        )
        r_source = (
            self._workspace_lens_source("r", model_id, source_site, source_layer, hidden_size)
            if include_r_lens else None
        )
        jpp_source = (
            self._workspace_lens_source("jpp", model_id, source_site, source_layer, hidden_size)
            if include_jpp_lens else None
        )
        with _runtime_lock, torch.inference_mode():
            unembedding = self.runtime.model.get_output_embeddings().weight
            directions = torch.tensor(
                vectors, device=unembedding.device, dtype=unembedding.dtype
            )

            def readout(
                values_to_read: Any, *, semantic_only: bool = False
            ) -> tuple[tuple[TokenScore, ...], ...]:
                normalized = self.runtime.final_norm(values_to_read[:, None, :])[:, 0]
                logits = (normalized @ unembedding.T).float()
                if semantic_only:
                    logits.masked_fill_(
                        ~self._semantic_token_mask(
                            logits.device, logits.shape[-1]
                        ).unsqueeze(0),
                        float("-inf"),
                    )
                values, ids = torch.topk(
                    logits, min(token_count, logits.shape[1]), dim=1
                )
                return tuple(
                    tuple(
                        TokenScore(
                            int(token_id),
                            self.runtime.tokenizer.decode([int(token_id)]),
                            float(score),
                        )
                        for token_id, score in zip(
                            row_ids.tolist(), row_values.tolist(), strict=True
                        )
                    )
                    for row_ids, row_values in zip(ids, values, strict=True)
                )

            def transported(
                source: Any, *, semantic_only: bool = False
            ) -> tuple[tuple[TokenScore, ...], ...]:
                if source is None:
                    return tuple(() for _ in vectors)
                mapping, _ = source
                mapped = directions.float() @ mapping.to(
                    device=directions.device, dtype=torch.float32
                ).T
                return readout(
                    mapped.to(dtype=directions.dtype), semantic_only=semantic_only
                )

            direct = (
                readout(directions)
                if include_logit_lens
                else tuple(() for _ in vectors)
            )
            j_readouts = transported(j_source)
            r_readouts = transported(r_source)
            jpp_readouts = transported(jpp_source, semantic_only=True)
            norms = torch.linalg.vector_norm(directions.float(), dim=1).tolist()
            return tuple(
                VectorProjection(
                    norm=float(norms[index]),
                    vocabulary=direct[index],
                    j_lens=j_readouts[index],
                    j_lens_metadata=(j_source[1] if j_source else None),
                    r_lens=r_readouts[index],
                    r_lens_metadata=(r_source[1] if r_source else None),
                    jpp_lens=jpp_readouts[index],
                    jpp_lens_metadata=(jpp_source[1] if jpp_source else None),
                )
                for index in range(len(vectors))
            )

    def _workspace_lens_source(
        self,
        method: str,
        model_id: str,
        site: str,
        layer: int | None,
        hidden_size: int,
    ) -> tuple[Any, dict[str, Any]] | None:
        """Resolve v2 site-aware maps, then an explicitly Post-only legacy map."""
        from llmdbg.workspace_lens import source_map as v2_source_map

        model_revision = getattr(self.runtime.model.config, "_commit_hash", None)
        tokenizer_revision = getattr(self.runtime.tokenizer, "_commit_hash", None)
        if tokenizer_revision is None:
            tokenizer_revision = getattr(self.runtime.tokenizer, "init_kwargs", {}).get(
                "_commit_hash"
            )
        source = v2_source_map(
            method, model_id, site, layer, hidden_size,
            layer_count=len(self.runtime.layers),
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            progress=self.progress,
        )
        if source is not None:
            return source.matrix, source.metadata
        # A repository-shaped local source is explicitly the site-aware v2
        # protocol (index.json -> manifest.json -> safetensors). If the exact
        # method/site is absent, report it as unavailable instead of treating
        # the repository root as a legacy directory containing lens.pt.
        from llmdbg.residual_vocabulary_lenses import active_lens

        configured = active_lens(method)
        if configured is not None:
            configured_source = str(configured.get("source", "")).strip()
            if configured_source.startswith("file://"):
                configured_source = configured_source.removeprefix("file://")
            if configured_source and not configured_source.startswith("hf://"):
                configured_root = Path(configured_source).expanduser()
                if (configured_root / "index.json").is_file():
                    return None
        fallback_from_site = site if method in {"j", "jpp"} and site == "mid" else None
        legacy_site = "post" if fallback_from_site is not None else site
        if legacy_site != "post" or layer is None:
            return None
        if fallback_from_site is not None and layer == len(self.runtime.layers) - 1:
            import torch

            fallback_label = "J-Lens" if method == "j" else "J++ Lens"
            self.progress(
                f"Using the final Post identity as a {fallback_label} fallback "
                f"for Mid layer {layer}.\n"
            )
            return torch.eye(hidden_size), {
                "format_version": 2,
                "model_id": model_id,
                "method": method,
                "site": "post",
                "layer": layer,
                "tensor_key": f"post.{layer}",
                "artifact_kind": "identity",
                "fallback": True,
                "fallback_from_site": fallback_from_site,
                "fallback_from_tensor_key": f"{fallback_from_site}.{layer}",
            }
        if method == "j":
            from llmdbg.j_lens import source_map as legacy_source_map
        elif method == "r":
            from llmdbg.r_lens import source_map as legacy_source_map
        else:
            from llmdbg.jpp_lens import source_map as legacy_source_map
        self.progress(
            f"Resolving legacy Post-only {method.upper()} Lens for layer {layer}; "
            "uncached files may be downloaded and loaded.\n"
        )
        legacy = legacy_source_map(model_id, layer, hidden_size)
        if legacy is None:
            self.progress(
                f"Legacy Post-only {method.upper()} Lens is unavailable for layer {layer}.\n"
            )
            return None
        matrix, metadata = legacy
        self.progress(f"Loaded legacy Post-only {method.upper()} Lens for layer {layer}.\n")
        resolved_metadata = {
            **metadata, "format_version": 1, "site": "post", "legacy": True,
        }
        if fallback_from_site is not None:
            resolved_metadata.update({
                "fallback": True,
                "fallback_from_site": fallback_from_site,
                "fallback_from_tensor_key": f"{fallback_from_site}.{layer}",
            })
            fallback_label = "J-Lens" if method == "j" else "J++ Lens"
            self.progress(
                f"Using Post layer {layer} {fallback_label} as a fallback "
                f"for Mid layer {layer}.\n"
            )
        return matrix, resolved_metadata

    def ica_components(
        self,
        layer_name: str,
        vector: tuple[float, ...],
        *,
        is_delta: bool,
        source_site: str = "post",
        source_layer: int | None = None,
        component_count: int = 3,
        token_count: int = 10,
        include_logit_lens: bool = True,
        include_j_lens: bool = False,
        include_r_lens: bool = False,
        include_jpp_lens: bool = False,
        include_indices: tuple[int, ...] = (),
        vector_overrides: dict[int, tuple[float, ...]] | None = None,
    ) -> tuple[ICAProjection, ...]:
        """Project a vector through the published resid-post ICA basis."""
        import re

        from llmdbg.ica_lens import top_components

        match = re.match(r"(?:transformer\.h|model\.layers)\.(\d+)(?:\.|$)", layer_name)
        if match is None:
            return ()
        model_id = str(getattr(self.runtime.model.config, "_name_or_path", ""))
        components = top_components(
            model_id,
            int(match.group(1)),
            vector,
            is_delta=is_delta,
            count=component_count,
            include_indices=include_indices,
            progress=self.progress,
        )
        vector_overrides = vector_overrides or {}
        lens_vectors = tuple(
            vector_overrides.get(component["index"], component["vector"])
            for component in components
        )
        projections = self._vectors_lens(
            layer_name,
            lens_vectors,
            source_site=source_site,
            source_layer=source_layer,
            token_count=token_count,
            include_logit_lens=include_logit_lens,
            include_j_lens=include_j_lens,
            include_r_lens=include_r_lens,
            include_jpp_lens=include_jpp_lens,
        )
        return tuple(
            ICAProjection(
                index=component["index"],
                rank=component["rank"],
                score=0.0 if component["index"] in vector_overrides else component["score"],
                energy=0.0 if component["index"] in vector_overrides else component["energy"],
                projection=projection,
                vector=tuple(float(value) for value in lens_vector),
            )
            for component, projection, lens_vector in zip(
                components, projections, lens_vectors, strict=True
            )
        )

_runtime_lock = RLock()
_runtime_key: tuple[str, str, str, str] | None = None
_runtime: ModelRuntime | None = None


def _resolve_device_and_dtype(device: str, dtype: str) -> tuple[str, str]:
    """Resolve portable automatic settings against the installed torch backend."""
    import torch

    if device not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(f"unsupported device: {device}")
    if dtype not in {"auto", "bfloat16", "float16", "float32"}:
        raise ValueError(f"unsupported dtype: {dtype}")
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this PyTorch runtime cannot use CUDA. "
            f"Installed torch is {torch.__version__} (built CUDA: {torch.version.cuda}). "
            "Run 'llmdbg: Show Runtime Diagnostics' for details."
        )
    if device == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError(
            "MPS was requested, but this PyTorch runtime cannot use Apple Metal. "
            "Run 'llmdbg: Show Runtime Diagnostics' for details."
        )
    native_cuda_bfloat16 = False
    if device == "cuda":
        capability = torch.cuda.get_device_capability()
        native_cuda_bfloat16 = (
            capability[0] >= 8 and torch.cuda.is_bf16_supported()
        )
    if dtype == "auto":
        if device == "cuda":
            dtype = "bfloat16" if native_cuda_bfloat16 else "float16"
        elif device == "mps":
            dtype = "float16"
        else:
            dtype = "float32"
    elif device == "cuda" and dtype == "bfloat16" and not native_cuda_bfloat16:
        capability = torch.cuda.get_device_capability()
        raise RuntimeError(
            "bfloat16 was requested, but this GPU does not provide native BF16 "
            f"execution (CUDA compute capability {capability[0]}.{capability[1]}). "
            "Use dtype='float16' or dtype='auto'."
        )
    return device, dtype


def _get_model_runtime(
    model_name_or_path: str,
    *,
    revision: str,
    device: str,
    dtype: str,
    progress: Callable[[str], None],
) -> ModelRuntime:
    """Return the matching resident model, loading it only on a cache miss."""
    global _runtime, _runtime_key
    key = (model_name_or_path, revision, device, dtype)
    if _runtime is not None and _runtime_key == key:
        progress(f"Reusing resident {model_name_or_path} model tensors.\n")
        return _runtime

    import torch
    from gb10_load_llm import from_pretrained_to_cuda
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForMultimodalLM,
        AutoProcessor,
        AutoTokenizer,
    )

    if _runtime is not None:
        progress("Model configuration changed; releasing the previous resident model.\n")
        _runtime = None
        _runtime_key = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]
    progress(
        f"Resolving tokenizer for {model_name_or_path}; "
        "uncached files may be downloaded.\n"
    )
    config = AutoConfig.from_pretrained(model_name_or_path, revision=revision)
    is_multimodal = getattr(config, "vision_config", None) is not None
    processor = None
    if is_multimodal:
        processor = AutoProcessor.from_pretrained(model_name_or_path, revision=revision)
        tokenizer = processor.tokenizer
        model_class = AutoModelForMultimodalLM
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, revision=revision)
        model_class = AutoModelForCausalLM
    progress(
        f"Loading {model_name_or_path} tensors on {device} through gb10-load-llm; "
        "uncached checkpoint shards may be downloaded.\n"
    )
    model = from_pretrained_to_cuda(
        model_class,
        model_name_or_path,
        revision=revision,
        dtype=torch_dtype,
        device=device,
        touch="auto",
    )
    model.eval()
    progress(
        f"Loaded {model_name_or_path} on {device} with {dtype} tensors; "
        "the model is ready.\n"
    )
    runtime = ModelRuntime(model=model, tokenizer=tokenizer, processor=processor)
    _runtime = runtime
    _runtime_key = key
    return runtime


def _tensor_from_output(output: Any):
    """Return the hidden-state tensor from a module output."""
    if isinstance(output, tuple):
        return output[0]
    return output


def _last_token_vector(tensor) -> tuple[float, ...]:
    """Copy the last-token hidden state to a plain CPU tuple."""
    return tuple(float(value) for value in tensor[0, -1].detach().float().cpu().tolist())


def _inspection_snapshots(hidden: Any, previous: Any, details: dict[str, Any]) -> dict[str, Any]:
    """Keep independent tensors for inspection without retaining autograd graphs."""
    return {
        name: tensor.detach().clone()
        for name, tensor in {
            "input": previous if previous is not None else hidden,
            "output": hidden,
            **details,
        }.items()
    }


def _logit_lens(model, tokenizer, hidden, *, apply_final_norm: bool) -> str:
    """Describe an intermediate vector using its strongest next-token predictions."""
    import torch

    with torch.inference_mode():
        runtime = ModelRuntime(model, tokenizer)
        projected = runtime.final_norm(hidden) if apply_final_norm else hidden
        logits = model.lm_head(projected)[0, -1].float()
        values, indices = torch.topk(logits, k=5)
    predictions = []
    for value, index in zip(values.tolist(), indices.tolist(), strict=True):
        token = tokenizer.decode([index]).replace("\n", "\\n")
        predictions.append(f"{token!r} ({value:+.2f})")
    vector = hidden[0, -1].detach().float()
    return (
        f"Logit-lens next-token evidence: {', '.join(predictions)}. "
        f"Vector mean={vector.mean().item():+.4f}, "
        f"std={vector.std().item():.4f}, norm={vector.norm().item():.4f}."
    )


def load_model_trace(
    model_name_or_path: str,
    *,
    revision: str = "main",
    prompt: str,
    device: str = "auto",
    dtype: str = "auto",
    progress: Callable[[str], None] | None = None,
) -> ModelTrace:
    """Load a supported model and return a layer-by-layer trace for one prompt.

    Model tensors are loaded through :mod:`gb10_load_llm`, as required for
    efficient cold-page transfers on GB10/GX10 systems.
    """
    execution = prepare_model_execution(
        model_name_or_path,
        revision=revision,
        prompt=prompt,
        device=device,
        dtype=dtype,
        progress=progress,
    )
    return execution.trace_token(execution.token_count - 1)


def prepare_model_execution(
    model_name_or_path: str,
    *,
    revision: str = "main",
    prompt: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    add_generation_prompt: bool = True,
    device: str = "auto",
    dtype: str = "auto",
    progress: Callable[[str], None] | None = None,
) -> ModelExecution:
    """Load/reuse a model and tokenize a prompt without running its forward pass."""
    try:
        import torch
        import gb10_load_llm  # noqa: F401 - validates the private runtime
        import transformers  # noqa: F401 - validates the private runtime
    except ImportError as error:
        raise RuntimeError(
            "The private llmdbg environment is missing model dependencies. "
            "Reinstall the VS Code extension or install gb10-load-llm into its runtime."
        ) from error

    if not prompt and not messages:
        raise ValueError("prompt or chat messages must not be empty")
    report = progress or (lambda _message: None)
    requested_device, requested_dtype = device, dtype
    device, dtype = _resolve_device_and_dtype(device, dtype)
    report(
        f"Runtime selection: device={device}, dtype={dtype} "
        f"(requested device={requested_device}, dtype={requested_dtype}).\n"
    )
    if device == "cuda":
        gpu_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(gpu_index)
        capability = torch.cuda.get_device_capability(gpu_index)
        report(
            f"CUDA device {gpu_index}: {properties.name}; compute capability "
            f"{capability[0]}.{capability[1]}; "
            f"{properties.total_memory / (1024 ** 3):.1f} GiB VRAM.\n"
        )
    with _runtime_lock:
        runtime = _get_model_runtime(
            model_name_or_path,
            revision=revision,
            device=device,
            dtype=dtype,
            progress=report,
        )
        if runtime.architecture not in {"gpt2", "qwen3_5", "qwen3_vl", "gemma2", "gemma4"}:
            raise ValueError(
                "llmdbg supports GPT-2, Qwen3.5, Qwen3-VL, Gemma 2, and Gemma 4 E2B models, "
                f"got {runtime.model.config.model_type!r}"
            )
        inputs_embeds = None
        position_ids = None
        image_grids = ()
        image_merge_size = 1
        visual_inputs = {}
        if messages and runtime.is_multimodal:
            multimodal_messages = _normalize_multimodal_messages(messages)
            rendered_prompt = runtime.processor.apply_chat_template(
                multimodal_messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
            )
            processed = runtime.processor.apply_chat_template(
                multimodal_messages,
                add_generation_prompt=add_generation_prompt, tokenize=True,
                return_dict=True, return_tensors="pt",
                # Keep interactive debugging practical: Qwen otherwise preserves
                # enough pixels from a large photograph to create >10k tokens.
                processor_kwargs={
                    "images_kwargs": {"max_soft_tokens": 280} if runtime.architecture == "gemma4" else {
                        "size": {
                            "shortest_edge": 65_536,
                            "longest_edge": 262_144,
                        }
                    }
                },
            )
            input_ids = processed["input_ids"]
            grid = processed.get("image_grid_thw")
            if grid is not None:
                image_grids = tuple(tuple(int(value) for value in row) for row in grid.tolist())
                image_merge_size = int(runtime.processor.image_processor.merge_size)
            if runtime.architecture == "gemma4":
                inputs_embeds, position_ids, visual_inputs = _prepare_gemma4_embeddings(runtime, processed, device)
                coordinates = processed.get("image_position_ids")
                if coordinates is not None:
                    image_grids = tuple((1, int(points[:, 1].max()) + 1, int(points[:, 0].max()) + 1)
                                        for points in coordinates)
                    image_merge_size = int(runtime.model.config.vision_config.pooling_kernel_size)
            else:
                inputs_embeds, position_ids, visual_inputs = _prepare_multimodal_embeddings(
                    runtime, processed, device
                )
        elif messages:
            rendered_prompt = runtime.tokenizer.apply_chat_template(
                messages, add_generation_prompt=add_generation_prompt, tokenize=False
            )
            input_ids = runtime.tokenizer.apply_chat_template(
                messages, add_generation_prompt=add_generation_prompt, tokenize=True,
                return_tensors="pt",
            )
        else:
            rendered_prompt = str(prompt)
            encoded = runtime.tokenizer(rendered_prompt, return_tensors="pt")
            input_ids = encoded["input_ids"]
        input_ids = _normalize_input_ids(input_ids, torch)
        tokens = tuple(
            runtime.tokenizer.decode([token_id]) for token_id in input_ids[0].tolist()
        )
    report(f"Prepared {len(tokens)} prompt tokens; forward execution is deferred.\n")
    return ModelExecution(
        runtime, input_ids, tokens, device, report,
        dtype=dtype, rendered_prompt=rendered_prompt,
        inputs_embeds=inputs_embeds, position_ids=position_ids,
        image_grids=image_grids, image_merge_size=image_merge_size,
        **visual_inputs,
    )


def _normalize_multimodal_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize local image sources for the Transformers multimodal processor."""
    import base64
    import io

    from PIL import Image

    def processor_source(source: str) -> str:
        path = Path(source).expanduser()
        if path.suffix.lower() not in {".avif", ".heic", ".heif"} or not path.is_file():
            return source
        with Image.open(path) as image:
            output = io.BytesIO()
            image.convert("RGB").save(output, format="PNG")
        return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")

    normalized: list[dict[str, Any]] = []
    for message in messages:
        copied = dict(message)
        content = copied.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                item = dict(part)
                if item.get("type") == "image":
                    source = item.get("url", item.get("path"))
                    if isinstance(source, str):
                        item["url"] = processor_source(source)
                        item.pop("path", None)
                parts.append(item)
            copied["content"] = parts
        normalized.append(copied)
    return normalized


def _prepare_multimodal_embeddings(
    runtime: ModelRuntime, processed: Any, device: str
) -> tuple[Any, Any, dict[str, Any]]:
    """Encode Qwen images once, retaining any DeepStack decoder features."""
    import torch

    model = runtime.model.model
    input_ids = processed["input_ids"].to(device)
    attention_mask = processed.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    with torch.inference_mode():
        visual_inputs = {}
        inputs_embeds = model.get_input_embeddings()(input_ids)
        pixel_values = processed.get("pixel_values")
        image_grid_thw = processed.get("image_grid_thw")
        if pixel_values is not None:
            image_grid_thw = image_grid_thw.to(device)
            image_outputs = model.get_image_features(
                pixel_values.to(device), image_grid_thw, return_dict=True
            )
            image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            deepstack = getattr(image_outputs, "deepstack_features", None)
            if deepstack is not None:
                visual_inputs = {
                    "visual_pos_masks": image_mask[..., 0],
                    "deepstack_visual_embeds": [torch.cat(feature, dim=0).to(
                        device=inputs_embeds.device, dtype=inputs_embeds.dtype
                    ) for feature in deepstack],
                }
        position_ids = model.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            mm_token_type_ids=(
                processed.get("mm_token_type_ids").to(device)
                if processed.get("mm_token_type_ids") is not None else None
            ),
        )
    return inputs_embeds, position_ids, visual_inputs


def _prepare_gemma4_embeddings(runtime: ModelRuntime, processed: Any, device: str) -> tuple[Any, Any, dict[str, Any]]:
    """Match Gemma 4's native image merge and token-identity PLE lookup."""
    import torch
    model = runtime.model.model
    config = model.config.get_text_config()
    if config.enable_moe_block or config.use_bidirectional_attention is not None:
        raise ValueError("Gemma 4 debugging currently supports dense, causal E2B checkpoints")
    ids = processed["input_ids"].to(device)
    with torch.inference_mode():
        masks = model.get_placeholder_mask(ids, None)
        multimodal = masks[0] | masks[1] | masks[2]
        safe_ids = torch.where(multimodal, config.pad_token_id, ids)
        embeddings = model.get_input_embeddings()(safe_ids)
        per_layer = model.language_model.get_per_layer_inputs(safe_ids, embeddings)
        if processed.get("pixel_values") is not None:
            features = model.get_image_features(
                processed["pixel_values"].to(device), processed["image_position_ids"].to(device), return_dict=True,
            )
            values = torch.cat(features.pooler_output, dim=0).to(embeddings.device, embeddings.dtype)
            embeddings = embeddings.masked_scatter(masks[0].unsqueeze(-1).expand_as(embeddings), values)
        positions = torch.arange(ids.shape[1], device=device).unsqueeze(0)
    return embeddings, positions, {"per_layer_inputs": per_layer}


def _normalize_input_ids(value: Any, torch: Any) -> Any:
    """Normalize tokenizer/chat-template outputs to a batched integer tensor."""
    if isinstance(value, dict):
        value = value["input_ids"]
    elif hasattr(value, "input_ids"):
        value = value.input_ids
    elif hasattr(value, "ids"):
        value = value.ids
    if not torch.is_tensor(value):
        value = torch.tensor(value, dtype=torch.long)
    else:
        value = value.to(dtype=torch.long)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2:
        raise ValueError(
            f"Tokenizer returned input IDs with shape {tuple(value.shape)}; expected one sequence"
        )
    return value


def _capture_gpt2_token_trace(
    runtime: ModelRuntime,
    *,
    input_ids: Any,
    past_key_values: Any,
    total_length: int,
    device: str,
    progress: Callable[[str], None],
    token_position: int,
    interventions: dict[tuple[int, str, str, int], float],
    ica_interventions: set[tuple[int, str, str, int]],
    ica_observations: dict[
        tuple[int, str, str, int], tuple[tuple[float, ...], tuple[float, ...]]
    ],
    vector_patches: dict[tuple[int, str, str], tuple[float, ...]],
    inputs_embeds: Any = None,
    position_ids: Any = None,
) -> tuple[ModelTrace, Any]:
    """Capture all layer states for one token using an existing KV cache."""
    import torch

    model = runtime.model
    tokenizer = runtime.tokenizer
    captured: list[tuple[str, str, str, Any, bool]] = []
    detailed: dict[str, dict[str, Any]] = {}
    handles = []

    def apply_vector_patch(tensor: Any, state_name: str, vector_name: str) -> Any:
        replacement = vector_patches.get((token_position, state_name, vector_name))
        if replacement is None:
            return tensor
        if len(replacement) != tensor.shape[-1]:
            raise ValueError(
                f"patch width {len(replacement)} does not match {state_name}.{vector_name} "
                f"width {tensor.shape[-1]}"
            )
        tensor[..., -1, :].copy_(
            torch.tensor(replacement, dtype=tensor.dtype, device=tensor.device)
        )
        return tensor

    def apply_interventions(tensor: Any, state_name: str, vector_name: str) -> Any:
        for (position, layer, vector, index), value in interventions.items():
            if position == token_position and layer == state_name and vector == vector_name:
                if index < 0 or index >= tensor.shape[-1]:
                    raise ValueError(
                        f"dimension {index} is outside {state_name}.{vector_name} "
                        f"range 0..{tensor.shape[-1] - 1}"
                    )
                tensor[..., index] = value
        return tensor

    def apply_ica_interventions(tensor: Any, state_name: str, vector_name: str) -> Any:
        from llmdbg.ica_lens import remove_component

        for position, layer, vector, component in sorted(ica_interventions):
            if position == token_position and layer == state_name and vector == vector_name:
                layer_index = int(state_name.split(".")[2])
                before = tuple(float(value) for value in tensor[..., -1, :].float().cpu().tolist()[0])
                remove_component(
                    str(getattr(model.config, "_name_or_path", "")),
                    layer_index,
                    tensor,
                    component,
                    progress=progress,
                )
                after = tuple(float(value) for value in tensor[..., -1, :].float().cpu().tolist()[0])
                ica_observations[(position, layer, vector, component)] = (before, after)
        return tensor

    def capture(name: str, operation: str, meaning: str, apply_final_norm: bool):
        def hook(_module, _inputs, output) -> None:
            tensor = _tensor_from_output(output)
            apply_vector_patch(tensor, name, "output")
            apply_interventions(tensor, name, "output")
            apply_ica_interventions(tensor, name, "output")
            captured.append((name, operation, meaning, tensor.detach(), apply_final_norm))

        return hook

    def capture_input(name: str, operation: str, meaning: str):
        def hook(_module, inputs) -> None:
            hidden = inputs[0]
            apply_vector_patch(hidden, name, "output")
            apply_interventions(hidden, name, "output")
            apply_ica_interventions(hidden, name, "output")
            captured.append((name, operation, meaning, hidden.detach(), True))

        return hook

    def capture_detail(state_name: str, detail_name: str):
        def hook(_module, _inputs, output) -> None:
            tensor = _tensor_from_output(output)
            apply_vector_patch(tensor, state_name, detail_name)
            apply_interventions(tensor, state_name, detail_name)
            detailed.setdefault(state_name, {})[detail_name] = tensor.detach()

        return hook

    def capture_detail_input(state_name: str, detail_name: str):
        def hook(_module, inputs) -> None:
            tensor = inputs[0]
            apply_vector_patch(tensor, state_name, detail_name)
            apply_interventions(tensor, state_name, detail_name)
            detailed.setdefault(state_name, {})[detail_name] = tensor.detach()

        return hook

    handles.append(
        model.transformer.drop.register_forward_hook(
            capture(
                "token_embedding",
                "token + positional embedding",
                "Encodes the prompt tokens and their positions into the residual stream.",
                True,
            )
        )
    )
    for index, block in enumerate(model.transformer.h):
        attention_name = f"transformer.h.{index}.attention"
        mlp_name = f"transformer.h.{index}.mlp"
        def intervene_block_input(_module, inputs, name=attention_name):
            apply_vector_patch(inputs[0], name, "input")
            apply_ica_interventions(inputs[0], name, "input")
        handles.append(block.register_forward_pre_hook(intervene_block_input))
        handles.append(
            block.ln_1.register_forward_hook(
                capture_detail(attention_name, "normalized_input")
            )
        )
        handles.append(
            block.attn.c_attn.register_forward_hook(
                capture_detail(attention_name, "attention_qkv")
            )
        )
        handles.append(
            block.attn.c_proj.register_forward_pre_hook(
                capture_detail_input(attention_name, "attention_mixing")
            )
        )
        handles.append(
            block.attn.c_proj.register_forward_hook(
                capture_detail(attention_name, "attention_projection")
            )
        )
        handles.append(
            block.ln_2.register_forward_pre_hook(
                capture_input(
                    attention_name,
                    "causal self-attention + residual",
                    f"Block {index} has mixed information across prompt positions.",
                )
            )
        )
        handles.append(
            block.ln_2.register_forward_hook(
                capture_detail(mlp_name, "normalized_input")
            )
        )
        handles.append(
            block.mlp.c_fc.register_forward_hook(
                capture_detail(mlp_name, "mlp_expansion_pre_activation")
            )
        )
        if hasattr(block.mlp.act, "register_forward_hook"):
            handles.append(
                block.mlp.act.register_forward_hook(
                    capture_detail(mlp_name, "mlp_expansion")
                )
            )
        handles.append(
            block.mlp.c_proj.register_forward_hook(
                capture_detail(mlp_name, "mlp_projection")
            )
        )
        handles.append(
            block.register_forward_hook(
                capture(
                    mlp_name,
                    "feed-forward MLP + residual",
                    f"Block {index} has applied its learned feature transformation.",
                    True,
                )
            )
        )
    handles.append(
        model.transformer.ln_f.register_forward_hook(
            capture(
                "transformer.ln_f",
                "final layer normalization",
                "Normalizes the final residual stream before vocabulary projection.",
                False,
            )
        )
    )
    handles.append(
        model.transformer.ln_f.register_forward_hook(
            capture_detail("transformer.ln_f", "normalized_output")
        )
    )

    try:
        attention_mask = torch.ones(
            (1, total_length), dtype=torch.long, device=device
        )
        progress(f"Running token {token_position} as a one-token cached forward pass.\n")
        with torch.inference_mode():
            model_output = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    states: list[LayerState] = []
    previous: tuple[float, ...] | None = None
    previous_tensor = None
    for name, operation, meaning, hidden, apply_final_norm in captured:
        output = _last_token_vector(hidden)
        interpretation = (
            f"{meaning} "
            f"{_logit_lens(model, tokenizer, hidden, apply_final_norm=apply_final_norm)}"
        )
        states.append(
            LayerState(
                Layer(name, operation, meaning, 1.0, 0.0),
                previous if previous is not None else output,
                output,
                interpretation,
                {
                    detail_name: _last_token_vector(detail_tensor)
                    for detail_name, detail_tensor in detailed.get(name, {}).items()
                },
                _inspection_snapshots(hidden, previous_tensor, detailed.get(name, {})),
            )
        )
        previous = output
        previous_tensor = hidden

    progress(f"Captured {len(states)} layer states for token {token_position}.\n")
    return ModelTrace(tuple(states), ()), model_output.past_key_values


def _capture_qwen35_token_trace(
    runtime: ModelRuntime,
    *,
    input_ids: Any,
    past_key_values: Any,
    total_length: int,
    device: str,
    progress: Callable[[str], None],
    token_position: int,
    interventions: dict[tuple[int, str, str, int], float],
    ica_interventions: set[tuple[int, str, str, int]],
    ica_observations: dict[
        tuple[int, str, str, int], tuple[tuple[float, ...], tuple[float, ...]]
    ],
    vector_patches: dict[tuple[int, str, str], tuple[float, ...]],
    inputs_embeds: Any = None,
    position_ids: Any = None,
    visual_pos_masks: Any = None,
    deepstack_visual_embeds: Any = None,
    per_layer_inputs: Any = None,
    capture_delta_core: bool = False,
) -> tuple[ModelTrace, Any]:
    """Capture Qwen and Gemma residual, attention, and gated-MLP states."""
    import torch

    model = runtime.model
    is_gemma4 = runtime.architecture == "gemma4"
    is_gemma2 = runtime.architecture in {"gemma2", "gemma4"}
    captured: list[tuple[str, str, str, Any, bool]] = []
    detailed: dict[str, dict[str, Any]] = {}
    inspection_details: dict[str, dict[str, Any]] = {}
    handles = []
    attention_configs: dict[int, tuple[Any, str]] = {}
    gemma_cache_before: dict[str, tuple[Any, Any]] = {}
    per_layer_residuals: dict[str, Any] = {}
    active_delta: list[str | None] = [None]
    delta_patches: list[tuple[Any, str, Any]] = []

    def store_delta(name, detail, tensor):
        snapshot = tensor.detach().clone()
        inspection_details.setdefault(name, {})[detail] = snapshot
        detailed.setdefault(name, {})[detail] = snapshot.reshape(snapshot.shape[0], 1, -1)

    def delta_enter(name):
        def hook(_module, _args):
            active_delta[0] = name
        return hook

    def delta_norm_input(name, mixer):
        def hook(_module, inputs):
            store_delta(name, "delta_gate", inputs[1].reshape(1, 1, mixer.num_v_heads, mixer.head_v_dim))
        return hook

    def wrap_delta_kernel(original, module):
        def wrapped(query, key, value, *args, **kwargs):
            name = active_delta[0]
            if name is None or query.shape[1] != 1:
                return original(query, key, value, *args, **kwargs)
            q, k = module.l2norm(query.float()), module.l2norm(key.float())
            q = q / q.shape[-1] ** 0.5
            memory = kwargs.get("initial_state")
            if memory is None:
                memory = torch.zeros(query.shape[0], value.shape[2], key.shape[-1], value.shape[-1], device=query.device)
            store_delta(name, "delta_previous_memory", memory.float())
            for detail, tensor in (("delta_query", q), ("delta_key", k), ("delta_value", value),
                                   ("delta_decay", kwargs["g"].float().exp()), ("delta_beta", kwargs["beta"])):
                store_delta(name, detail, tensor)
            store_delta(name, "delta_qkv", torch.cat((q.flatten(2), k.flatten(2), value.flatten(2)), dim=-1))
            result = original(query, key, value, *args, **kwargs)
            store_delta(name, "delta_updated_memory", result[1])
            store_delta(name, "delta_readout", result[0])
            return result
        return wrapped

    def token_heads(tensor: Any) -> Any:
        """Represent [batch, token, heads, width] as a debugger vector."""
        return tensor.reshape(tensor.shape[0], tensor.shape[1], -1)

    def attention_matrix(tensor: Any) -> Any:
        """Represent [batch, heads, query, key] at each query position."""
        return tensor.transpose(1, 2).reshape(tensor.shape[0], tensor.shape[2], -1)

    def cached_sequence(tensor: Any) -> Any:
        """Expose every cached position as one inspectable debugger vector."""
        return tensor.transpose(1, 2).reshape(tensor.shape[0], 1, -1)

    def apply_vector_patch(tensor: Any, state_name: str, vector_name: str) -> Any:
        replacement = vector_patches.get((token_position, state_name, vector_name))
        if replacement is None:
            return tensor
        if len(replacement) != tensor.shape[-1]:
            raise ValueError(
                f"patch width {len(replacement)} does not match {state_name}.{vector_name} "
                f"width {tensor.shape[-1]}"
            )
        tensor[..., -1, :].copy_(
            torch.tensor(replacement, dtype=tensor.dtype, device=tensor.device)
        )
        return tensor

    def apply_interventions(tensor: Any, state_name: str, vector_name: str) -> Any:
        for (position, layer, vector, index), value in interventions.items():
            if position == token_position and layer == state_name and vector == vector_name:
                if index < 0 or index >= tensor.shape[-1]:
                    raise ValueError(
                        f"dimension {index} is outside {state_name}.{vector_name} "
                        f"range 0..{tensor.shape[-1] - 1}"
                    )
                tensor[..., index] = value
        return tensor

    def apply_ica_interventions(tensor: Any, state_name: str, vector_name: str) -> Any:
        from llmdbg.ica_lens import remove_component

        for position, layer, vector, component in sorted(ica_interventions):
            if position == token_position and layer == state_name and vector == vector_name:
                layer_index = int(state_name.split(".")[2])
                before = tuple(float(value) for value in tensor[..., -1, :].float().cpu().tolist()[0])
                remove_component(
                    str(getattr(model.config, "_name_or_path", "")),
                    layer_index,
                    tensor,
                    component,
                    progress=progress,
                )
                after = tuple(float(value) for value in tensor[..., -1, :].float().cpu().tolist()[0])
                ica_observations[(position, layer, vector, component)] = (before, after)
        return tensor

    def capture(name: str, operation: str, meaning: str, apply_final_norm: bool):
        def hook(_module, _inputs, output) -> None:
            tensor = _tensor_from_output(output)
            apply_vector_patch(tensor, name, "output")
            apply_interventions(tensor, name, "output")
            apply_ica_interventions(tensor, name, "output")
            captured.append((name, operation, meaning, tensor.detach(), apply_final_norm))
        return hook

    def capture_input(name: str, operation: str, meaning: str):
        def hook(_module, inputs) -> None:
            tensor = inputs[0]
            apply_vector_patch(tensor, name, "output")
            apply_interventions(tensor, name, "output")
            apply_ica_interventions(tensor, name, "output")
            captured.append((name, operation, meaning, tensor.detach(), True))
        return hook

    def capture_detail(state_name: str, detail_name: str):
        def hook(_module, _inputs, output) -> None:
            tensor = _tensor_from_output(output)
            apply_vector_patch(tensor, state_name, detail_name)
            apply_interventions(tensor, state_name, detail_name)
            detailed.setdefault(state_name, {})[detail_name] = tensor.detach()
        return hook

    def capture_detail_input(state_name: str, detail_name: str):
        def hook(_module, inputs) -> None:
            tensor = inputs[0]
            apply_vector_patch(tensor, state_name, detail_name)
            apply_interventions(tensor, state_name, detail_name)
            detailed.setdefault(state_name, {})[detail_name] = tensor.detach()
        return hook

    def capture_attention_heads(state_name: str, detail_name: str):
        def hook(_module, _inputs, output) -> None:
            tensor = token_heads(_tensor_from_output(output))
            detailed.setdefault(state_name, {})[detail_name] = tensor.detach()
            inspection_details.setdefault(state_name, {})[detail_name] = output.detach()
        return hook

    def capture_attention_result(state_name: str):
        def hook(_module, _inputs, output) -> None:
            if not isinstance(output, tuple) or output[1] is None:
                return
            weights = attention_matrix(output[1])
            detailed.setdefault(state_name, {})["attention_weights"] = weights.detach()
            inspection_details.setdefault(state_name, {})["attention_weights"] = output[1].detach()
        return hook

    def capture_gemma_cache_before(state_name: str):
        def hook(module, _args, kwargs):
            cache = kwargs.get("past_key_values")
            layers = getattr(cache, "layers", ())
            if not module.is_kv_shared_layer and module.layer_idx < len(layers):
                layer = layers[module.layer_idx]
                if getattr(layer, "keys", None) is not None:
                    gemma_cache_before[state_name] = (layer.keys.detach().clone(), layer.values.detach().clone())
        return hook

    def capture_gemma_attention(state_name: str):
        def hook(module, _args, kwargs, output):
            weights = output[1]
            if weights is None:
                raise RuntimeError("Gemma 4 tracing requires eager attention weights")
            count = int(weights.shape[-1])
            if module.is_kv_shared_layer:
                keys, values = kwargs["shared_kv_states"][module.layer_type]
            else:
                layer = kwargs["past_key_values"].layers[module.layer_idx]
                keys, values = layer.keys, layer.values
                if keys.shape[-2] < count:
                    old_keys, old_values = gemma_cache_before[state_name]
                    keys = torch.cat((old_keys, keys[..., -1:, :]), dim=-2)
                    values = torch.cat((old_values, values[..., -1:, :]), dim=-2)
                keys, values = keys[..., -count:, :], values[..., -count:, :]
            for detail, tensor in (("attention_cached_keys", keys), ("attention_cached_values", values)):
                inspection_details.setdefault(state_name, {})[detail] = tensor.detach()
                detailed.setdefault(state_name, {})[detail] = cached_sequence(tensor).detach()
            inspection_details[state_name]["attention_source_positions"] = torch.arange(total_length - count, total_length)
            inspection_details[state_name]["attention_weights"] = weights.detach()
            detailed[state_name]["attention_weights"] = attention_matrix(weights).detach()
        return hook

    def remember_mlp_residual(name: str):
        def hook(_module, inputs):
            per_layer_residuals[name] = inputs[0].detach()
        return hook

    def capture_per_layer_residual(name: str, mlp_name: str):
        def hook(_module, _inputs, output):
            update = _tensor_from_output(output)
            tensor = per_layer_residuals[mlp_name] + update
            detailed.setdefault(name, {})["per_layer_update"] = update.detach()
            captured.append((name, "per-layer embedding + residual", "Adds the gated per-layer token-identity embedding.", tensor.detach(), True))
        return hook

    def capture_gated_activation(state_name: str, activation: Any):
        def hook(_module, _inputs, output) -> None:
            gate = _tensor_from_output(output)
            apply_interventions(gate, state_name, "mlp_gate_pre_activation")
            detailed.setdefault(state_name, {})["mlp_gate_pre_activation"] = gate.detach()
            gate_activation = activation(gate)
            apply_interventions(gate_activation, state_name, "mlp_gate_activation")
            detailed.setdefault(state_name, {})["mlp_gate_activation"] = gate_activation.detach()
        return hook

    text_model = runtime.text_model
    if inputs_embeds is not None:
        captured.append((
            "token_embedding", "multimodal embedding",
            "Encodes this image or text token into the initial residual stream.",
            inputs_embeds.detach(), True,
        ))
    else:
        handles.append(
            text_model.embed_tokens.register_forward_hook(
            capture(
                "token_embedding",
                "token embedding",
                "Encodes the prompt token into the initial residual stream; positions enter through RoPE.",
                True,
            )
            )
        )
    for index, block in enumerate(runtime.layers):
        attention_kind = runtime.layer_kind(index)
        attention_name = runtime.state_name(index, attention_kind)
        mlp_name = runtime.state_name(index, "mlp")
        if index > 0 and index <= runtime.visual_layer_count:
            handles.append(block.register_forward_pre_hook(capture_input(
                runtime.state_name(index - 1, "visual_residual"),
                "DeepStack visual residual addition",
                f"Injects vision-encoder features after layer {index - 1} at image-token positions.",
            )))
        def intervene_block_input(_module, inputs, name=attention_name):
            apply_vector_patch(inputs[0], name, "input")
            apply_ica_interventions(inputs[0], name, "input")
        handles.append(block.register_forward_pre_hook(intervene_block_input))
        handles.append(
            block.input_layernorm.register_forward_hook(
                capture_detail(attention_name, "normalized_input")
            )
        )
        mixer = block.self_attn if attention_kind == "full_attention" else block.linear_attn
        if attention_kind == "gated_linear_attention" and capture_delta_core:
            handles.append(mixer.register_forward_pre_hook(delta_enter(attention_name)))
            handles.append(mixer.norm.register_forward_pre_hook(delta_norm_input(attention_name, mixer)))
        output_projection = mixer.o_proj if attention_kind == "full_attention" else mixer.out_proj
        if attention_kind == "full_attention" and is_gemma2 and not is_gemma4:
            handles.append(mixer.v_proj.register_forward_hook(capture_detail(attention_name, "attention_value")))
        if attention_kind == "full_attention" and (not is_gemma2 or is_gemma4):
            config = mixer.config
            config_key = id(config)
            if config_key not in attention_configs:
                attention_configs[config_key] = (
                    config,
                    str(getattr(config, "_attn_implementation", "eager")),
                )
                config._attn_implementation = "eager"
            handles.append(
                mixer.q_norm.register_forward_hook(
                    capture_attention_heads(attention_name, "attention_query")
                )
            )
            if hasattr(mixer, "k_norm"):
                handles.append(mixer.k_norm.register_forward_hook(
                    capture_attention_heads(attention_name, "attention_key")
                ))
            value_projection = getattr(mixer, "v_norm", None) if is_gemma4 else getattr(mixer, "v_proj", None)
            if value_projection is not None:
                handles.append(value_projection.register_forward_hook(
                    capture_attention_heads(attention_name, "attention_value") if is_gemma4 else
                    capture_detail(attention_name, "attention_value")
                ))
            if is_gemma4:
                handles.append(mixer.register_forward_pre_hook(capture_gemma_cache_before(attention_name), with_kwargs=True))
                handles.append(mixer.register_forward_hook(capture_gemma_attention(attention_name), with_kwargs=True))
            else:
                handles.append(mixer.register_forward_hook(capture_attention_result(attention_name)))
        mixing_detail = (
            "full_attention_mixing"
            if attention_kind == "full_attention"
            else "gated_delta_mixing"
        )
        handles.append(
            output_projection.register_forward_pre_hook(
                capture_detail_input(attention_name, mixing_detail)
            )
        )
        if not is_gemma2:
            handles.append(
                output_projection.register_forward_hook(
                    capture_detail(attention_name, "attention_projection")
                )
            )
        else:
            if is_gemma4:
                handles.append(output_projection.register_forward_hook(capture_detail(attention_name, "attention_projection_raw")))
            handles.append(
                block.post_attention_layernorm.register_forward_hook(
                    capture_detail(attention_name, "attention_projection")
                )
            )
        attention_boundary = (
            block.pre_feedforward_layernorm
            if is_gemma2 else block.post_attention_layernorm
        )
        handles.append(
            attention_boundary.register_forward_pre_hook(
                capture_input(
                    attention_name,
                    attention_kind.replace("_", " ") + " + residual",
                    f"Layer {index} has applied {attention_kind.replace('_', ' ')}.",
                )
            )
        )
        handles.append(
            attention_boundary.register_forward_hook(
                capture_detail(mlp_name, "normalized_input")
            )
        )
        handles.append(
            block.mlp.gate_proj.register_forward_hook(
                capture_gated_activation(mlp_name, block.mlp.act_fn)
            )
        )
        handles.append(
            block.mlp.up_proj.register_forward_hook(
                capture_detail(mlp_name, "mlp_up_projection")
            )
        )
        handles.append(
            block.mlp.down_proj.register_forward_pre_hook(
                capture_detail_input(mlp_name, "mlp_expansion")
            )
        )
        mlp_projection = (
            block.post_feedforward_layernorm if is_gemma2 else block.mlp.down_proj
        )
        handles.append(
            mlp_projection.register_forward_hook(
                capture_detail(mlp_name, "mlp_projection")
            )
        )
        if is_gemma4 and block.hidden_size_per_layer_input:
            handles.append(block.per_layer_input_gate.register_forward_pre_hook(capture_input(
                mlp_name, "GeGLU MLP + residual", f"Layer {index} has applied its MLP residual addition.",
            )))
            handles.append(block.per_layer_input_gate.register_forward_pre_hook(remember_mlp_residual(mlp_name)))
            handles.append(block.post_per_layer_input_norm.register_forward_hook(capture_per_layer_residual(
                runtime.state_name(index, "per_layer_residual"), mlp_name,
            )))
            handles.append(block.register_forward_hook(capture(
                runtime.state_name(index, "layer_scale"), "layer scaling", "Scales the complete block output by its learned layer scalar.", True,
            )))
        else:
            handles.append(block.register_forward_hook(
                capture(
                    mlp_name,
                    "SwiGLU MLP + residual",
                    f"Layer {index} has applied its gated feature transformation.",
                    True,
                )
            ))
    if runtime.visual_layer_count == len(runtime.layers) and runtime.visual_layer_count:
        handles.append(runtime.final_norm.register_forward_pre_hook(capture_input(
            runtime.state_name(len(runtime.layers) - 1, "visual_residual"),
            "DeepStack visual residual addition", "Injects vision features after the final block.",
        )))
    handles.append(
        runtime.final_norm.register_forward_hook(
            capture(
                runtime.final_norm_name,
                "final RMS normalization",
                "Normalizes the final residual stream before vocabulary projection.",
                False,
            )
        )
    )
    handles.append(
        runtime.final_norm.register_forward_hook(
            capture_detail(runtime.final_norm_name, "normalized_output")
        )
    )

    try:
        if runtime.architecture == "qwen3_5" and capture_delta_core:
            import importlib
            for block in runtime.layers:
                if not hasattr(block, "linear_attn"):
                    continue
                module = importlib.import_module(type(block.linear_attn).__module__)
                for name in ("torch_recurrent_gated_delta_rule", "torch_chunk_gated_delta_rule"):
                    original = getattr(module, name)
                    delta_patches.append((module, name, original))
                    setattr(module, name, wrap_delta_kernel(original, module))
                break
        attention_mask = torch.ones((1, total_length), dtype=torch.long, device=device)
        progress(f"Running token {token_position} as a one-token cached forward pass.\n")
        with torch.inference_mode():
            forward_model = text_model if inputs_embeds is not None else model
            model_output = forward_model(
                input_ids=None if inputs_embeds is not None else input_ids.to(device),
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                **({"visual_pos_masks": visual_pos_masks,
                    "deepstack_visual_embeds": deepstack_visual_embeds}
                   if runtime.architecture == "qwen3_vl" else {}),
                **({"per_layer_inputs": per_layer_inputs} if is_gemma4 else {}),
            )
        cache = model_output.past_key_values
        cache_layers = getattr(cache, "layers", ())
        for index, block in enumerate(runtime.layers):
            if is_gemma4:
                break  # Captured from the actual attention inputs, including shared/sliding KV.
            if runtime.layer_kind(index) != "full_attention" or index >= len(cache_layers):
                continue
            cache_layer = cache_layers[index]
            keys = getattr(cache_layer, "keys", None)
            values = getattr(cache_layer, "values", None)
            state_name = runtime.state_name(index, "full_attention")
            if keys is not None:
                inspection_details.setdefault(state_name, {})["attention_cached_keys"] = keys.detach()
                detailed.setdefault(state_name, {})["attention_cached_keys"] = (
                    cached_sequence(keys).detach()
                )
            if values is not None:
                inspection_details.setdefault(state_name, {})["attention_cached_values"] = values.detach()
                detailed.setdefault(state_name, {})["attention_cached_values"] = (
                    cached_sequence(values).detach()
                )
    finally:
        for module, name, original in delta_patches:
            setattr(module, name, original)
        for handle in handles:
            handle.remove()
        for config, implementation in attention_configs.values():
            config._attn_implementation = implementation

    states: list[LayerState] = []
    previous: tuple[float, ...] | None = None
    previous_tensor = None
    for name, operation, meaning, hidden, apply_final_norm in captured:
        output = _last_token_vector(hidden)
        interpretation = (
            f"{meaning} "
            f"{_logit_lens(model, runtime.tokenizer, hidden, apply_final_norm=apply_final_norm)}"
        )
        states.append(
            LayerState(
                Layer(name, operation, meaning, 1.0, 0.0),
                previous if previous is not None else output,
                output,
                interpretation,
                {
                    detail_name: _last_token_vector(detail_tensor)
                    for detail_name, detail_tensor in detailed.get(name, {}).items()
                },
                _inspection_snapshots(hidden, previous_tensor, {
                    **detailed.get(name, {}), **inspection_details.get(name, {}),
                }),
            )
        )
        previous = output
        previous_tensor = hidden
    progress(f"Captured {len(states)} layer states for token {token_position}.\n")
    return ModelTrace(tuple(states), ()), model_output.past_key_values


def _capture_token_trace(runtime: ModelRuntime, **kwargs: Any) -> tuple[ModelTrace, Any]:
    if runtime.architecture in {"qwen3_5", "qwen3_vl", "gemma2", "gemma4"}:
        return _capture_qwen35_token_trace(runtime, **kwargs)
    return _capture_gpt2_token_trace(runtime, **kwargs)
