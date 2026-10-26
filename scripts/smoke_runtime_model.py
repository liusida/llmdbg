"""Run with a profile's Python and PYTHONPATH=src to test its real model path."""
import argparse
import os
from pathlib import Path

from llmdbg.config import load_debug_config
from llmdbg.model_runtime import prepare_model_execution

parser = argparse.ArgumentParser()
parser.add_argument("config")
args = parser.parse_args()
config = load_debug_config(args.config)
os.chdir(Path(args.config).resolve().parent)
execution = prepare_model_execution(
    config["model"], revision=config.get("revision", "main"),
    messages=config.get("messages"), prompt=config.get("prompt"),
    device=config.get("device", "auto"), dtype=config.get("dtype", "auto"),
    progress=lambda message: print(message, end="", flush=True),
)
execution.capture_delta_core = any(step['operation']=='convolved_qkv'
    for plan in config.get('sourcePlans',{}).values() for step in plan['steps'])
trace = execution.trace_token(execution.token_count - 1)
if execution.capture_delta_core:
    import io
    from llmdbg.dap import LLMDebugAdapter, ProtocolStream
    adapter = LLMDebugAdapter(ProtocolStream(io.BytesIO(), io.BytesIO()))
    adapter.execution = execution
    adapter._configure_model_source(config)
    adapter.trace = trace.states
    adapter.inside_attention = True
    adapter.layer_index = next(i for i, s in enumerate(trace.states) if s.layer.name.endswith(".gated_linear_attention"))
    planned = config['sourcePlans']['gated_linear_attention']['steps']
    adapter.mlp_step_index = next(i for i,step in enumerate(planned) if step['operation']=='gated_output_norm')
    for name in ("query", "key", "value", "decay", "beta", "gate", "memory", "readout"):
        tensor = adapter._inspect_expression(name)
        assert tensor is not None, name
        print("CORE_TENSOR", name, tuple(tensor.shape), flush=True)
    return_index = next((i for i,step in enumerate(planned) if step['kind']=='return'
                         and step['frames'][-1]['bindings'].get('mixed')=='gated_delta_mixing'),None)
    if return_index is not None:
        import torch
        adapter.mlp_step_index = return_index
        mixed = adapter._inspect_expression('mixed')
        torch.testing.assert_close(mixed,adapter.outer_state.tensors['gated_delta_mixing'])
        print('SOURCE_RETURN_TENSOR mixed',tuple(mixed.shape),flush=True)
_, predictions = execution.vocabulary_projection(trace.states[-1].output_vector)
print("NEXT_TOKEN", predictions[0].text, flush=True)
execution.append_generated_token(predictions[0].token_id)
trace = execution.trace_token(execution.token_count - 1)
print("SMOKE_OK", config["model"], "states", len(trace.states), "image_grids", execution.image_grids, flush=True)
