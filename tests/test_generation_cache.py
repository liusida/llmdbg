import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llmdbg import generation_cache


class GenerationCacheTest(unittest.TestCase):
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
