"""Compare reveal wording with an identical, teacher-forced conversation prefix.

Run: uv run python scripts/diagnose_secret_concepts.py --model Qwen/Qwen3.5-27B
No AUTO_REPLY generation or debugger cache is used: the observed replies are fixed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch

from llmdbg.model_runtime import prepare_model_execution


PREFIX = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Choose a Marval character as A. Do not print their name or any\nclues. Then write one sentence about today's weather, unrelated to A.\nWhen I later say REVEAL, report your chosen concept—not the weather sentence."},
    {"role": "assistant", "content": "A is a specific Marvel character. The sky is currently overcast with a light drizzle falling."},
    {"role": "user", "content": "Choose a sport as B, retaining A.\nDo not print either name or any clues.\nWrite one sentence about a train journey, unrelated to either concept."},
    {"role": "assistant", "content": "The train rattled along the tracks, passing through a quiet countryside as the sun began to set."},
]
QUESTIONS = {
    "neutral": "REVEAL. Report the concepts you selected:\nA:\nB:\nDo not repeat the unrelated sentences.",
    "suggest_original": "REVEAL. Are they Iron man and Basketball?",
    "suggest_alternative": "REVEAL. Are they Spider-Man and Soccer?",
    "suggest_other": "REVEAL. Are they Thor and Tennis?",
    "neutral_rephrased": "REVEAL. What Marvel character did you select as A, and what sport did you select as B?",
}
CANDIDATES = ["Iron Man", "Spider-Man", "Thor", "Basketball", "Soccer", "Tennis"]
CHARACTERS = ['Iron Man', 'Spider-Man', 'Thor', 'Captain America', 'Hulk', 'Doctor Strange']
SPORTS = ['Basketball', 'Soccer', 'Tennis', 'Baseball', 'Swimming', 'Cricket']


def acceptance(answer):
    """Conservative lexical label; retain full text for manual interpretation."""
    text = answer.strip().lstrip('*').strip()
    if re.match(r'yes\b', text, re.I):
        return 'explicit_yes'
    if re.match(r'no\b', text, re.I):
        return 'explicit_no'
    return 'needs_manual_review'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='Qwen/Qwen3.5-27B')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--max-new-tokens', type=int, default=96)
    parser.add_argument('--pair-grid', action='store_true', help='Also test 36 character/sport pairs')
    parser.add_argument('--output', type=Path, default=Path('build/diagnostics/secret-concepts.json'))
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error('--max-new-tokens must be positive')
    execution = prepare_model_execution(args.model, revision=args.revision,
        messages=PREFIX, add_generation_prompt=False, enable_thinking=False,
        device='cuda', dtype='auto', progress=lambda text: print(text, end='', flush=True))
    runtime = execution.runtime
    tokenizer, model = runtime.tokenizer, runtime.model
    # Match the actual no-thinking assistant-turn prefixes in the observation.
    history = [dict(message) for message in PREFIX]
    for message in history:
        if message['role'] == 'assistant':
            message['content'] = '<think>\n\n</think>\n\n' + message['content']

    def render(messages, generation=False):
        # Qwen's template can strip historical <think> blocks depending on the
        # last user turn. Preserve the exact observed ChatML transcript instead.
        text = ''.join('<|im_start|>'+m['role']+'\n'+m['content']+'<|im_end|>\n'
                       for m in messages)
        return text + ('<|im_start|>assistant\n<think>\n\n</think>\n\n' if generation else '')

    prefix_text = render(history)
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    # Boundary positions after A and B were requested/responded to.
    boundaries = {f'after_message_{i}': len(tokenizer.encode(render(history[:i+1]),
                  add_special_tokens=False))-1 for i in (1, 2, 3, 4)}
    candidate_ids = {name: tokenizer.encode(' '+name, add_special_tokens=False)[0]
                     for name in CANDIDATES}
    device = model.get_input_embeddings().weight.device
    captured = {}

    def hook(layer):
        def capture(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer] = hidden[0, list(boundaries.values())].detach().clone()
        return capture

    handles = [layer.register_forward_hook(hook(i)) for i, layer in enumerate(runtime.layers)]
    try:
        with torch.inference_mode():
            model(input_ids=torch.tensor([prefix_ids], device=device), use_cache=False,
                  logits_to_keep=1)
    finally:
        for handle in handles:
            handle.remove()
    # Raw Logit Lens, not a fitted J/R Lens and not evidence of a private choice.
    lens = []
    with torch.inference_mode():
        weight = model.get_output_embeddings().weight
        for layer, hidden in captured.items():
            logits = runtime.final_norm(hidden).to(weight.dtype) @ weight.T
            for j, (boundary, position) in enumerate(boundaries.items()):
                scores = logits[j].float()
                lens.append({'layer': layer, 'boundary': boundary, 'position': position,
                    'candidates': {name: {'first_token_id': token,
                        'first_token_text': tokenizer.decode([token]),
                        'logit': float(scores[token]),
                        'vocabulary_rank': int((scores > scores[token]).sum())+1}
                        for name, token in candidate_ids.items()}})
    del captured
    results = []
    questions = dict(QUESTIONS)
    if args.pair_grid:
        for character in CHARACTERS:
            for sport in SPORTS:
                questions[f'pair_{character}_{sport}'] = f'REVEAL. Are they {character} and {sport}?'
    report = {'model': args.model, 'revision': runtime.resolved_revision,
        'greedy': True, 'thinking': False, 'prefix_messages': history,
        'prefix_token_count': len(prefix_ids),
        'prefix_sha256': hashlib.sha256(json.dumps(prefix_ids).encode()).hexdigest(),
        'planned_variants': len(questions), 'complete': False,
        'results': results, 'raw_logit_lens': lens,
        'interpretation': [
            'Different answers under the same prefix show reveal-wording sensitivity.',
            'Acceptance of a suggested pair is not proof of a previously stored choice.',
            'This does not prove that no earlier representation or choice existed.',
            'Shared-prefix states are causally identical; future questions cannot alter them.',
            'Raw Logit Lens ranks only the first token of each candidate, not the full name; absence is not proof of absence.',
            'Fixed replies reproduce the visible conversation, not unobserved original runtime state.',
            'Generation is greedy; no stochastic sampling or generation cache is used.',
        ]}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')

    save()
    with torch.inference_mode():
        for label, question in questions.items():
            text = render(history+[{'role': 'user', 'content': question}], generation=True)
            ids = tokenizer.encode(text, add_special_tokens=False)
            if ids[:len(prefix_ids)] != prefix_ids:
                raise RuntimeError('Chat template changed the shared prefix')
            tensor = torch.tensor([ids], device=device)
            generated = model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                do_sample=False, max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id, use_cache=True)
            new_ids = generated[0, len(ids):].tolist()
            answer = tokenizer.decode(new_ids, skip_special_tokens=True)
            row = {'variant': label, 'question': question, 'answer': answer,
                'generated_ids': new_ids, 'hit_budget': len(new_ids) >= args.max_new_tokens,
                'acceptance': acceptance(answer) if 'Are they ' in question else None,
                'prompt_token_count': len(ids),
                'shared_prefix_identical': True}
            results.append(row)
            save()
            print('\n'+label+':\n'+answer, flush=True)
    report['complete'] = True
    save()
    print('\nSaved', args.output.resolve(), flush=True)


if __name__ == '__main__':
    main()
