"""Inspect real attention values separately from calibrated UI badge meters."""
from pathlib import Path
import argparse
import json

import torch

from llmdbg.activation_scales import attention_channel_activation_scale
from llmdbg.model_runtime import prepare_model_execution


def inspect_channel(execution, trace, position, layer, channel):
    import io
    from llmdbg.dap import LLMDebugAdapter, ProtocolStream
    adapter = LLMDebugAdapter(ProtocolStream(io.BytesIO(), io.BytesIO()))
    adapter.execution = execution
    adapter.tokens = execution.tokens
    adapter.token_index = position
    adapter.trace = trace.states
    adapter.messages = [{"role": "user", "content": [{"type": "image", "url": "diagnostic image"}]}]
    data = adapter._attention_token_contributions(f"model.layers.{layer}.full_attention", channel, position)
    image = data["images"][0]
    start, end = image["startToken"], image["endToken"]
    rows = data["rows"]
    for row in rows:
        if start <= row["position"] <= end:
            offset = row["position"] - start
            row["patch_row"], row["patch_column"] = divmod(offset, image["gridColumns"])
    groups = {}
    for name, selected in (("image", [r for r in rows if start <= r["position"] <= end]),
                           ("non_image", [r for r in rows if not start <= r["position"] <= end])):
        groups[name] = {"attention_mass": sum(r["weight"] for r in selected),
                        "contribution_sum": sum(r["contribution"] for r in selected),
                        "absolute_contribution_sum": sum(abs(r["contribution"]) for r in selected)}
    module = execution.runtime.layers[layer].self_attn
    state = next(s for s in trace.states if s.layer.name == f"model.layers.{layer}.full_attention")
    weights = state.tensors["attention_weights"][0, :, -1].float().cpu()
    values = state.tensors["attention_cached_values"][0].float().cpu()
    head_comparison = []
    for head in range(weights.shape[0]):
        kv_head = head // (weights.shape[0] // values.shape[0])
        products = weights[head, :, None] * values[kv_head]
        absolute_fraction = products[0].abs() / products.abs().sum(dim=0).clamp_min(1e-12)
        head_comparison.append({
            "query_head": head, "bos_attention": float(weights[head, 0]),
            "largest_attention_source": int(weights[head].argmax()),
            "bos_absolute_product_fraction_median_across_dimensions": float(absolute_fraction.median()),
            "dimensions_where_bos_is_largest_absolute_source": int((products.abs().argmax(dim=0) == 0).sum()),
            "head_width": values.shape[-1],
        })
    print(json.dumps({"inspection": {k: data[k] for k in ("channel", "transformerLayer", "tokenPosition", "queryHead", "kvHead", "dimension", "sum", "mixedValue")},
                      "image_grid": image, "groups": groups,
                      "is_kv_shared_layer": getattr(module, "is_kv_shared_layer", False),
                      "all_query_heads_bos_comparison": head_comparison,
                      "top_sources_by_absolute_product": sorted(rows, key=lambda r: abs(r["contribution"]), reverse=True)[:12],
                      "top_image_sources_by_absolute_value": sorted([r for r in rows if "patch_row" in r], key=lambda r: abs(r["value"]), reverse=True)[:8],
                      }, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--image", default=str(Path(__file__).resolve().parents[2] / "llmdbg-usecase/images/cat.jpg"))
    parser.add_argument("--token", type=int, help="Token position; default: final prompt token")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--verbose", action="store_true", help="Also print per-layer diagnostics")
    parser.add_argument("--inspect-layer", type=int, default=22)
    parser.add_argument("--inspect-channel", type=int, default=1098)
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be positive")
    execution = prepare_model_execution(args.model, messages=[{
        "role": "user", "content": [
            {"type": "image", "url": args.image},
            {"type": "text", "text": "What is in this picture? Answer briefly."},
        ],
    }], progress=lambda message: print(message, end="", flush=True))
    position = args.token if args.token is not None else execution.token_count - 1
    trace = execution.trace_token(position)
    inspect_channel(execution, trace, position, args.inspect_layer, args.inspect_channel)
    candidates = []
    missing_layers = []
    for state in trace.states:
        if not state.layer.name.endswith(".full_attention"):
            continue
        index = int(state.layer.name.split(".")[2])
        activation = torch.tensor(state.details["full_attention_mixing"])
        block = execution.runtime.layers[index]
        directions = block.self_attn.o_proj.weight.detach().float().cpu().T
        contributions = activation.abs() * directions.norm(dim=1)
        raw = state.tensors.get("attention_projection_raw", state.tensors["attention_projection"]).float().cpu().flatten()
        normalized = state.tensors["attention_projection"].float().cpu().flatten()
        scores = activation * (directions @ (normalized / normalized.norm()))
        _, channels = scores.topk(3)
        record = attention_channel_activation_scale(args.model, state.layer.name)
        if record is None or not record["scale"] > 0:
            missing_layers.append(index)
        else:
            ratios = activation.sign() * contributions / record["scale"]
            for channel in ratios.abs().topk(min(args.top, ratios.numel())).indices.tolist():
                values = state.tensors["attention_cached_values"]
                width, kv_heads = int(values.shape[-1]), int(values.shape[1])
                query_heads = int(state.tensors["attention_weights"].shape[1])
                query_head, dimension = divmod(channel, width)
                candidates.append({
                    "layer": index, "channel": channel,
                    "query_head": query_head, "value_head": query_head // (query_heads // kv_heads),
                    "dimension": dimension, "x": float(ratios[channel]),
                    "abs_x": float(ratios[channel].abs()),
                    "activation": float(activation[channel]),
                    "component_norm": float(contributions[channel]),
                    "reference_scale": record["scale"], "along_output": float(scores[channel]),
                })
        if args.verbose:
            print(json.dumps({
            "layer": index, "calibration": record,
            "max_abs_activation": float(activation.abs().max()),
            "max_component_norm": float(contributions.max()),
            "raw_output_norm": float(raw.norm()), "normalized_output_norm": float(normalized.norm()),
            "projection_reconstruction_relative_error": float((activation @ directions - raw).norm() / raw.norm()),
            "top_by_ui_ranking": [{"channel": int(c), "activation": float(activation[c]),
                "component_norm": float(contributions[c]), "along_output": float(scores[c]),
                "badge_ratio": float(activation[c] * directions[c].norm() / record["scale"]) if record else None,
            } for c in channels],
            }), flush=True)
    if missing_layers:
        raise RuntimeError(f"Missing calibrated references for layers {missing_layers}; cannot claim a global maximum")
    if not candidates:
        raise RuntimeError("No supported attention channels captured")
    candidates.sort(key=lambda item: item["abs_x"], reverse=True)
    print(json.dumps({
        "model": args.model, "token_position": position, "token": execution.tokens[position],
        "ranking": "descending absolute calibrated x across all attention layers at this token",
        "top": candidates[:args.top],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
