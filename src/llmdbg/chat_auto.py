"""Explicit, JSON-safe assistant generation placeholders for chat programs."""
from __future__ import annotations

from enum import Enum
from types import SimpleNamespace
import uuid


class _AutoReply(Enum):
    VALUE = 'AUTO_REPLY'


AUTO_REPLY = _AutoReply.VALUE
AUTO_REPLY_MARKER = {'__llmdbg_auto_reply__': True}


class AutoReplyLimitReached(ValueError):
    """A valid but unfinished intermediate response, not a model execution failure."""


def is_auto_reply(content):
    return content is AUTO_REPLY or (isinstance(content, dict) and content == AUTO_REPLY_MARKER)


def normalize_messages(messages):
    if not all(isinstance(message,dict) for message in messages):
        raise ValueError('Each chat message must be a dictionary.')
    result = []
    for index, message in enumerate(messages):
        copied = dict(message)
        if is_auto_reply(copied.get('content')):
            if copied.get('role') != 'assistant':
                raise ValueError('AUTO_REPLY is only valid as assistant content.')
            if not index or messages[index-1].get('role') != 'user':
                raise ValueError('AUTO_REPLY must follow a user message.')
            if index+1 == len(messages) or messages[index+1].get('role') != 'user':
                raise ValueError('AUTO_REPLY must be followed by a user follow-up message.')
            copied['content'] = dict(AUTO_REPLY_MARKER)
        result.append(copied)
    return result


def validate_continuation(messages, add_generation_prompt, continue_final_message):
    if type(continue_final_message) is not bool:
        raise ValueError('continue_final_message must be a boolean.')
    if not continue_final_message:
        return
    if add_generation_prompt:
        raise ValueError('continue_final_message=True requires add_generation_prompt=False.')
    if not messages or messages[-1].get('role') != 'assistant':
        raise ValueError('continue_final_message=True requires a final assistant prefill message.')
    content = messages[-1].get('content')
    if isinstance(content,list) and content and isinstance(content[-1],dict) and content[-1].get('type')=='text':
        content = content[-1].get('text')
    if not isinstance(content,str) or not content:
        raise ValueError('The final assistant prefill must contain non-empty text.')


def _turn_cache(execution):
    cache = getattr(execution.runtime,'auto_reply_cache',None)
    if cache is None:
        cache = {}
        execution.runtime.auto_reply_cache = cache
    return cache


def _remember_turn(execution, key, ids, eos):
    cache = _turn_cache(execution)
    old = cache.get(key)
    if old and len(old['ids'])>len(ids) and old['ids'][:len(ids)]==ids:
        return
    cache[key] = {'ids':list(ids),'eos':bool(eos)}
    while len(cache)>32:
        cache.pop(next(iter(cache)))


def generate_reply(execution, budget, model, revision, *, start=None):
    """Complete one turn using native cached forwards and the persistent ID cache."""
    import torch
    from llmdbg import generation_cache

    start = execution.token_count if start is None else start
    adapter = SimpleNamespace(execution=execution, model_name=model, model_revision=revision)
    key, persistent, metadata = generation_cache.identity(adapter, start)
    key = generation_cache.fingerprint(key, 'auto-reply-v1')
    modified = any(getattr(execution,name,{}) for name in ('interventions','ica_interventions','vector_patches'))
    cacheable = not modified
    entry = _turn_cache(execution).get(key) if not modified else None
    source = 'resident memory' if entry else 'disk'
    if entry is None and persistent:
        entry = generation_cache.load(key, metadata['vocabulary_size'])
    prefix = execution.input_ids[0,start:].tolist()
    if entry and entry['ids'][:len(prefix)] != prefix:
        entry = None
        persistent = False
        cacheable = False
        reason = 'cached continuation conflicts with the current token prefix'
    else:
        reason = 'no matching completed entry' if persistent else 'disk cache unavailable; no matching resident entry'
    if entry:
        for token in entry['ids'][len(prefix):budget]:
            execution.append_generated_token(token)
        if entry['eos'] and len(entry['ids']) <= budget:
            if cacheable:
                _remember_turn(execution,key,entry['ids'],True)
            execution.progress(f'AUTO_REPLY: reused {len(entry["ids"])} cached tokens ({source}); no generation.\n')
            return execution.input_ids[0,start:].tolist()
        if execution.token_count-start >= budget:
            execution.progress(f'AUTO_REPLY: reused {min(len(entry["ids"]),budget)} cached tokens ({source}); partial reply at token limit, no generation.\n')
            raise AutoReplyLimitReached(f'AUTO_REPLY reached max_new_tokens={budget} before EOS. '
                                       'Increase max_new_tokens so the intermediate reply can finish.')

    execution.progress(f'AUTO_REPLY: cache miss ({reason}); generating after {execution.token_count-start} existing/cached reply tokens.\n')

    eos = getattr(execution.runtime.tokenizer, 'eos_token_id', None)
    if eos is None:
        eos = getattr(getattr(execution.runtime.model, 'generation_config', None), 'eos_token_id', None)
    eos_ids = set(eos if isinstance(eos, (tuple,list,set)) else [eos]) - {None}
    if not eos_ids:
        raise ValueError('AUTO_REPLY requires an EOS token to delimit assistant turns.')
    hidden = [None]
    def capture(_module, _inputs, output):
        hidden[0] = output[0,-1].detach()
    handle = execution.runtime.final_norm.register_forward_hook(capture)
    finished = False
    try:
        while execution.token_count-start < budget:
            modified = any(getattr(execution,name,{}) for name in ('interventions','ica_interventions','vector_patches'))
            if modified:
                for position in range(execution.cached_through+1,execution.token_count):
                    execution.trace_token(position)
                if hidden[0] is None:
                    execution.trace_token(execution.token_count-1)
            else:
                execution._prefill(execution.token_count)
            if hidden[0] is None:
                raise RuntimeError('AUTO_REPLY could not capture the final normalized state.')
            weight = execution.runtime.model.get_output_embeddings().weight
            logits = weight @ hidden[0].to(device=weight.device,dtype=weight.dtype)
            config = getattr(execution.runtime.model.config,'text_config',execution.runtime.model.config)
            cap = getattr(config,'final_logit_softcapping',None)
            if cap is not None:
                logits = torch.tanh(logits/cap)*cap
            token = int(logits.float().argmax())
            if token in eos_ids:
                finished = True
                break
            execution.append_generated_token(token)
        ids = execution.input_ids[0,start:].tolist()
        if cacheable:
            _remember_turn(execution,key,ids,finished)
        if persistent:
            generation_cache.save(key,ids,finished,metadata)
        if not finished:
            raise AutoReplyLimitReached(f'AUTO_REPLY reached max_new_tokens={budget} before EOS. '
                                        'Increase max_new_tokens so the intermediate reply can finish.')
        return ids
    finally:
        handle.remove()


def prepare_auto_chat(model, messages, *, max_new_tokens, **kwargs):
    """Prepare only the initial prompt; defer replies and follow-ups until reached."""
    from llmdbg.model_runtime import prepare_model_execution

    messages = normalize_messages(messages)
    if max_new_tokens <= 0:
        raise ValueError('AUTO_REPLY requires max_new_tokens > 0.')
    first = next(index for index,message in enumerate(messages) if is_auto_reply(message.get('content')))
    if any(isinstance(m.get('content'),list) and any(p.get('type')=='image' for p in m['content'])
           for m in messages[first+1:]):
        raise ValueError('Place image inputs before the first AUTO_REPLY; follow-ups currently support text only.')
    initial_kwargs = {**kwargs, 'add_generation_prompt': True, 'continue_final_message': False}
    execution = prepare_model_execution(model,messages=messages[:first],**initial_kwargs)
    turn_kwargs = {key:kwargs[key] for key in ('revision','add_generation_prompt','enable_thinking','continue_final_message') if key in kwargs}
    execution.auto_chat = {'messages':messages,'history':messages[:first],'current':first,
                           'start':execution.token_count,'budget':int(max_new_tokens),
                           'model':model,'kwargs':turn_kwargs}
    execution.progress('AUTO_REPLY: intermediate reply and follow-up deferred until reached.\n')
    return execution


def finish_auto_turn(execution, *, save_cache=True):
    """At EOS, append only the authored follow-up, without another forward pass."""
    from llmdbg import generation_cache
    from llmdbg.model_runtime import _normalize_multimodal_messages
    state = execution.auto_chat
    if not state:
        return False
    messages,history,current = state['messages'],list(state['history']),state['current']
    ids = execution.input_ids[0,state['start']:].tolist()
    if save_cache:
        adapter = SimpleNamespace(execution=execution,model_name=state['model'],model_revision=state['kwargs'].get('revision','main'))
        key,persistent,metadata = generation_cache.identity(adapter,state['start'])
        key = generation_cache.fingerprint(key,'auto-reply-v1')
        if not any(getattr(execution,name,{}) for name in ('interventions','ica_interventions','vector_patches')):
            _remember_turn(execution,key,ids,True)
        if persistent:
            generation_cache.save(key,ids,True,metadata)
    text = execution.runtime.tokenizer.decode(ids,skip_special_tokens=False)
    history.append({'role':'assistant','content':text})
    following = current+1
    next_auto = next((i for i in range(following,len(messages))
                     if is_auto_reply(messages[i].get('content'))),len(messages))
    additions = messages[following:next_auto]
    # Derive the closing EOS and next prompt from the model's template, without
    # retokenizing or stripping any already-generated reasoning or token IDs.
    marker = 'llmdbg_auto_reply_'+uuid.uuid4().hex
    probe = history[:-1]+[{'role':'assistant','content':marker}]+additions
    runtime = execution.runtime
    templater = runtime.processor if runtime.is_multimodal else runtime.tokenizer
    rendered = templater.apply_chat_template(
        _normalize_multimodal_messages(probe) if runtime.is_multimodal else probe,
        tokenize=False,add_generation_prompt=True if next_auto<len(messages) else state['kwargs'].get('add_generation_prompt',True),
        continue_final_message=next_auto==len(messages) and state['kwargs'].get('continue_final_message',False),
        enable_thinking=state['kwargs'].get('enable_thinking',False))
    if rendered.count(marker)!=1:
        raise ValueError('Chat template did not preserve AUTO_REPLY assistant content.')
    suffix = rendered.split(marker,1)[1]
    for token in runtime.tokenizer.encode(suffix,add_special_tokens=False):
        execution.append_generated_token(token)
    execution.rendered_prompt += text+suffix
    history.extend(additions)
    execution.auto_chat = ({**state,'history':history,'current':next_auto,'start':execution.token_count}
                           if next_auto<len(messages) else None)
    execution.progress('AUTO_REPLY: appended the follow-up; ready to continue debugging.\n')
    return True


def resolve_auto_chat(execution, *, allow_partial=False):
    """Resolve pending turns only for an explicitly requested full-sequence replay."""
    import torch
    from llmdbg.model_runtime import _runtime_lock
    with _runtime_lock, torch.inference_mode():
        while execution.auto_chat:
            state = execution.auto_chat
            execution.progress('AUTO_REPLY: resolving intermediate assistant turn…\n')
            try:
                generate_reply(execution,state['budget'],state['model'],state['kwargs'].get('revision','main'),start=state['start'])
            except AutoReplyLimitReached:
                if not allow_partial:
                    raise
                execution.progress('AUTO_REPLY: token limit reached; keeping partial reply for inspection. Later turns have not run.\n')
                break
            finish_auto_turn(execution,save_cache=False)
    return execution
