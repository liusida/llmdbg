"""Private, atomic JSON cache of generated IDs; never deserialize model tensors."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import time


def fingerprint(*values):
    import torch
    digest = hashlib.sha256()
    def visit(value):
        if isinstance(value, torch.Tensor):
            digest.update(str((tuple(value.shape), str(value.dtype))).encode())
            digest.update(value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
        elif isinstance(value, dict):
            digest.update(b'dict')
            for key in sorted(value, key=str): visit(key); visit(value[key])
        elif isinstance(value, (tuple, list)):
            digest.update(str(('sequence',len(value))).encode())
            for item in value: visit(item)
        else: digest.update(repr((type(value).__name__,value)).encode())
    for value in values: visit(value)
    return digest.hexdigest()


def identity(adapter, prompt_count):
    execution = adapter.execution
    runtime, tokenizer = execution.runtime, execution.runtime.tokenizer
    config = runtime.model.config
    revision = getattr(runtime,'resolved_revision',None) or getattr(config, '_commit_hash', None)
    requested = getattr(adapter, 'model_revision', 'main')
    revision = revision or (requested if requested not in ('main','master','') else None)
    # A local mutable checkpoint or unresolved branch cannot safely share a disk entry.
    persistent = bool(revision) and not (execution.interventions or execution.ica_interventions or execution.vector_patches)
    vocab = getattr(tokenizer, '_llmdbg_generation_vocab_hash', None)
    if vocab is None:
        vocab = fingerprint(tokenizer.get_vocab() if hasattr(tokenizer,'get_vocab') else {},
                            getattr(tokenizer,'chat_template',None),getattr(tokenizer,'init_kwargs',{}))
        try: tokenizer._llmdbg_generation_vocab_hash = vocab
        except (AttributeError,TypeError): pass
    versions = {}
    for name in ('torch','transformers'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    eos = getattr(tokenizer,'eos_token_id',None)
    if eos is None: eos = getattr(getattr(runtime.model,'generation_config',None),'eos_token_id',None)
    metadata = {'schema':1, 'model':str(adapter.model_name), 'revision':revision, 'tokenizer':vocab,
                'dtype':str(runtime.model.get_output_embeddings().weight.dtype), 'device':str(execution.device),
                'versions':versions, 'policy':'greedy-argmax-v1',
                'eos':eos,
                'config':config.to_dict() if hasattr(config,'to_dict') else vars(config)}
    key = fingerprint(metadata, execution.rendered_prompt, execution.input_ids[:,:prompt_count],
        execution.inputs_embeds[:,:prompt_count] if execution.inputs_embeds is not None else None,
        execution.position_ids[...,:prompt_count] if execution.position_ids is not None else None,
        execution.visual_pos_masks[:,:prompt_count] if execution.visual_pos_masks is not None else None,
        execution.deepstack_visual_embeds,
        execution.per_layer_inputs[:,:prompt_count] if execution.per_layer_inputs is not None else None,
        execution.interventions,execution.ica_interventions,execution.vector_patches)
    return key, persistent, {'schema':1,'model':metadata['model'],'revision':revision,
                             'vocabulary_size':int(runtime.model.get_output_embeddings().weight.shape[0])}


def cache_directory():
    override = os.environ.get('LLMDBG_GENERATION_CACHE_DIR')
    return Path(override).expanduser() if override else Path(os.environ.get('XDG_CACHE_HOME') or Path.home()/'.cache')/'llmdbg'/'generations'


def token_checksum(ids,eos):
    return hashlib.sha256(json.dumps([ids,eos],separators=(',',':')).encode()).hexdigest()


def load(key, vocabulary_size):
    try:
        path = cache_directory() / f'{key}.json'
        if path.stat().st_size > 16*1024*1024: return None
        data = json.loads(path.read_text(encoding='utf-8'))
        ids = data.get('ids')
        if data.get('schema') != 1 or data.get('key') != key or type(data.get('eos')) is not bool:
            return None
        if not isinstance(ids,list) or any(type(i) is not int or not 0<=i<vocabulary_size for i in ids):
            return None
        if data.get('token_sha256') != token_checksum(ids,data['eos']): return None
        return data
    except (OSError,ValueError,TypeError,AttributeError): return None


def save(key, ids, eos, metadata, *, force=False):
    """Cache failures must not prevent a successful debugger replay."""
    temporary = None
    try:
        root = cache_directory()
        root.mkdir(parents=True,exist_ok=True,mode=0o700)
        if not force:
            old = load(key, metadata['vocabulary_size'])
            if old and len(old['ids'])>len(ids) and old['ids'][:len(ids)]==ids: return
        with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=root,prefix='.generation-',delete=False) as stream:
            temporary = Path(stream.name)
            json.dump({**metadata,'key':key,'ids':ids,'eos':bool(eos),'created':time.time(),
                       'token_sha256':token_checksum(ids,bool(eos))},stream,separators=(',',':'))
            stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,root/f'{key}.json')
    except (OSError,TypeError,ValueError): pass
    finally:
        if temporary is not None:
            try: temporary.unlink(missing_ok=True)
            except OSError: pass
