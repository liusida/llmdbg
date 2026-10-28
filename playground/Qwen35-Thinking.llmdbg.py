"""Observe thinking tokens and the final response on a text-only question."""
from llmdbg import debug, nop, qwen35

model = qwen35(
    "Qwen/Qwen3.5-2B",
    revision="main",
    device="auto",
    dtype="auto",
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Why does the sky look blue during the day but red or orange at sunset?"},
]
# Thinking and the final answer share this generation budget.
tokens = model.chat(messages, max_new_tokens=4096, enable_thinking=True,
                    add_generation_prompt=True)


def attention(normalized, block):
    query = block.query_projection(normalized)
    key = block.key_projection(normalized)
    value = block.value_projection(normalized)
    keys = block.cached_keys(key)
    values = block.cached_values(value)
    # values: [batch, KV_heads, cached_token_positions, head_dimension]
    weights = block.attention_weights(query, keys)
    # weights: [batch, query_heads, query_positions, key_positions]
    mixed = block.mix_values(weights, values)
    # mixed: [batch, query_positions, query_heads × head_dimension]
    # Head counts and dimensions follow the selected model's configuration.
    return mixed


def full_attention(x, block):
    normalized = block.input_layernorm(x)
    mixed = attention(normalized, block)
    update = block.output_projection(mixed)
    return x + update


def gated_delta(normalized, block):
    query, key, value = block.convolved_qkv(normalized)
    decay, beta, gate = block.gates(normalized)
    memory = block.cached_memory()
    memory = block.update_memory(memory, key, value, decay, beta)
    # memory: [batch, value_heads, key_dimension, value_dimension]
    readout = block.read_memory(query, memory)
    # readout: [batch, query_positions, value_heads, value_dimension]
    mixed = block.gated_output_norm(readout, gate)
    # mixed: [batch, query_positions, value_heads × value_dimension]
    return mixed


def gated_linear_attention(x, block):
    normalized = block.input_layernorm(x)
    mixed = gated_delta(normalized, block)
    update = block.output_projection(mixed)
    return x + update


def mlp(x, block):
    normalized = block.post_attention_layernorm(x)
    gate = block.gate_projection(normalized)
    gate = block.silu(gate)
    up = block.up_projection(normalized)
    hidden = gate * up
    update = block.down_projection(hidden)
    return x + update


with debug(model, tokens) as run:
    for token in run.tokens:
        x = model.token_embedding(tokens)
        for layer, block in enumerate(model.layers):
            nop()  # resid-pre
            if block.kind == "full_attention":
                x = full_attention(x, block)
            else:
                x = gated_linear_attention(x, block)
            nop()  # resid-mid
            x = mlp(x, block)
            nop()  # resid-post

        x = model.final_layernorm(x)
        logits = model.vocabulary_projection(x)
        predictions = model.top_tokens(logits, k=10)
        nop()  # predicting next token
