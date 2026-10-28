"""Fast CPU-only real-DAP coverage; no Hub downloads or GPU allocations."""
from pathlib import Path
import tempfile
import unittest

from dap_client import DebugHost


class StartersDAPTest(unittest.TestCase):
    def test_tiny_checkpoint_launch_steps_predictions_generation_and_reconnect(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
        from llmdbg.config import load_debug_config

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root/'tiny-gpt2'
            tokenizer = Tokenizer(WordLevel({'[UNK]': 0, **{f'w{i}': i for i in range(1, 32)}}, unk_token='[UNK]'))
            tokenizer.pre_tokenizer = Whitespace()
            PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]').save_pretrained(checkpoint)
            config = GPT2Config(vocab_size=32, n_embd=16, n_layer=2, n_head=2,
                                n_positions=64, bos_token_id=None, eos_token_id=None)
            GPT2LMHeadModel(config).eval().save_pretrained(checkpoint)
            source = Path(__file__).resolve().parents[1]/'starters/GPT2-Text.llmdbg.py'
            program = load_debug_config(source)
            host = DebugHost(root/'logs', env_overrides={'CUDA_VISIBLE_DEVICES': ''})
            try:
                for session in range(2):
                    client = host.client(f'session-{session}')
                    try:
                        client.request('initialize', adapterID='llmdbg')
                        client.request('launch', config=str(source), model=str(checkpoint),
                                       device='cpu', dtype='float32', stopOnEntry=True, maxNewTokens=2)
                        client.request('configurationDone')
                        self.assertEqual(client.stopped()['reason'], 'entry')
                        prompt = client.request('llmdbg/promptTokens')
                        self.assertTrue(prompt['tokens'])
                        client.request('next', threadId=1)
                        client.stopped()
                        client.request('stepIn', threadId=1)
                        client.stopped()
                        frames = client.request('stackTrace', threadId=1)['stackFrames']
                        self.assertEqual(Path(frames[0]['source']['path']), source)
                        scopes = client.request('scopes', frameId=frames[0]['id'])['scopes']
                        self.assertTrue(any(client.request('variables', variablesReference=s['variablesReference'])['variables']
                                            for s in scopes))
                        vector = client.request('llmdbg/vector')
                        self.assertEqual(vector['dimensions'], 16)
                        client.request('stepOut', threadId=1)
                        client.stopped()
                        bp = client.request('setBreakpoints', source={'path': str(source)},
                            breakpoints=[{'line': program['callLines']['mlp'], 'condition': 'token=0, layer=0'}])
                        self.assertTrue(bp['breakpoints'][0]['verified'])
                        client.request('continue', threadId=1)
                        client.stopped()
                        client.request('stepIn', threadId=1)
                        client.stopped()
                        self.assertIn('mlp', client.request('stackTrace', threadId=1)['stackFrames'][0]['name'])
                        last = len(prompt['tokens'])-1
                        bp = client.request('setBreakpoints', source={'path': str(source)},
                            breakpoints=[{'line': max(program['callLines']['nop']), 'condition': f'token={last}'}])
                        self.assertTrue(bp['breakpoints'][0]['verified'])
                        client.request('continue', threadId=1)
                        client.stopped()
                        self.assertIn('tokens', client.request('evaluate', expression='predictions')['result'])
                        self.assertTrue(client.request('evaluate', expression='predictions[0].token')['result'])
                        client.request('setBreakpoints', source={'path': str(source)}, breakpoints=[])
                        for _ in range(4):
                            client.request('next', threadId=1)
                            client.stopped()
                            after = client.request('llmdbg/promptTokens')
                            if after['currentToken'] > last:
                                break
                        self.assertGreater(after['currentToken'], last)
                        self.assertGreater(len(after['tokens']), len(prompt['tokens']))
                        client.request('disconnect')
                    finally:
                        client.close()
            finally:
                host.close()
