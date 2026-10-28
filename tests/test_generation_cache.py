import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import Mock
from types import SimpleNamespace

from llmdbg import generation_cache


class GenerationCacheTest(unittest.TestCase):
    def test_loader_pins_config_tokenizer_and_weights_to_same_resolved_commit(self):
        from llmdbg.model_runtime import _get_model_runtime
        sha='1234567890abcdef1234567890abcdef12345678'
        config=SimpleNamespace(vision_config=None)
        model=SimpleNamespace(eval=Mock())
        with patch('llmdbg.model_runtime._runtime',None),patch('llmdbg.model_runtime._runtime_key',None), \
             patch('transformers.utils.hub.cached_file',return_value='/cache/snapshots/'+sha+'/config.json'), \
             patch('transformers.AutoConfig.from_pretrained',return_value=config) as config_load, \
             patch('transformers.AutoTokenizer.from_pretrained',return_value=object()) as tokenizer_load, \
             patch('gb10_load_llm.from_pretrained_to_cuda',return_value=model) as weight_load:
            runtime=_get_model_runtime('test/model',revision='main',device='cpu',dtype='float32',progress=lambda _:None)
        self.assertEqual(runtime.resolved_revision,sha)
        for loader in (config_load,tokenizer_load,weight_load):
            self.assertEqual(loader.call_args.kwargs['revision'],sha)
        self.assertIs(weight_load.call_args.kwargs['config'],config)

    def test_safe_json_validation_atomic_storage_and_longer_prefix_preservation(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,{'LLMDBG_GENERATION_CACHE_DIR':directory}):
            key='a'*64
            metadata={'schema':1,'model':'test','revision':'pinned','vocabulary_size':64}
            generation_cache.save(key,[5,6,63],True,metadata)
            path=Path(directory)/f'{key}.json'
            self.assertEqual(path.stat().st_mode & 0o777,0o600)
            self.assertEqual(generation_cache.load(key,64)['ids'],[5,6,63])
            generation_cache.save(key,[5],False,metadata)
            self.assertEqual(generation_cache.load(key,64)['ids'],[5,6,63])
            generation_cache.save(key,[7],False,metadata,force=True)
            self.assertEqual(generation_cache.load(key,64)['ids'],[7])
            payload=json.loads(path.read_text())
            for field,value in (('ids',[True]),('ids',[-1]),('ids',[64]),('ids',[8]),('eos','yes'),('schema',9),('key','other')):
                path.write_text(json.dumps({**payload,field:value}))
                self.assertIsNone(generation_cache.load(key,64))
            path.write_text('incomplete JSON')
            self.assertIsNone(generation_cache.load(key,64))
            generation_cache.save(key,[8],True,metadata)
            self.assertEqual(generation_cache.load(key,64)['ids'],[8])
            self.assertFalse(list(Path(directory).glob('.generation-*')))

    def test_io_errors_are_nonfatal(self):
        with patch('llmdbg.generation_cache.cache_directory',side_effect=OSError('not writable')):
            generation_cache.save('a'*64,[1],False,{'vocabulary_size':64})
            self.assertIsNone(generation_cache.load('a'*64,64))


if __name__=='__main__':unittest.main()
