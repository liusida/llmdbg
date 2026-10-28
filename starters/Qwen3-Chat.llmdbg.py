"""Text chat with a dense, full-attention Qwen3 model."""
from llmdbg import debug, nop, qwen3

# Change this ID to select another dense Qwen3 size.
model = qwen3(
    "Qwen/Qwen3-8B",
    revision="main",
    device="auto",
    dtype="auto",
)

messages = [
    {"role": "system", "content": "You are a helpful assistant. Be concise."},
    {"role": "user", "content": "Why does ice float on water? Answer in two sentences."},
]
tokens = model.chat(messages, max_new_tokens=128, enable_thinking=False,
                    add_generation_prompt=True)


def attention(normalized, block):
    query = block.query_projection(normalized)
    key = block.key_projection(normalized)
    value = block.value_projection(normalized)
    keys = block.cached_keys(key)
    values = block.cached_values(value)
    weights = block.attention_weights(query, keys)
    mixed = block.mix_values(weights, values)
    return mixed


def full_attention(x, block):
    normalized = block.input_layernorm(x)
    mixed = attention(normalized, block)
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
            x = full_attention(x, block)
            nop()  # resid-mid
            x = mlp(x, block)
            nop()  # resid-post
        x = model.final_layernorm(x)
        logits = model.vocabulary_projection(x)
        predictions = model.top_tokens(logits, k=10)
        nop()  # predicting next token
