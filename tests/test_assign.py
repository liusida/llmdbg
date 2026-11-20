"""Expression assignment (llmdbg/assign) and named vectors on a tiny CPU GPT-2."""
from pathlib import Path
import math
import tempfile
import unittest

from dap_client import DebugHost


class AssignTest(unittest.TestCase):
    def test_expression_assignment_rewrites_the_residual_stream(self):
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
            GPT2LMHeadModel(GPT2Config(vocab_size=32, n_embd=16, n_layer=2, n_head=2, n_positions=64,
                                       bos_token_id=None, eos_token_id=None)).eval().save_pretrained(checkpoint)
            source = Path(__file__).resolve().parents[1]/'starters/GPT2-Text.llmdbg.py'
            program = load_debug_config(source)
            host = DebugHost(root/'logs', env_overrides={'CUDA_VISIBLE_DEVICES': ''})
            client = host.client('assign')
            try:
                client.request('initialize', adapterID='llmdbg')
                client.request('launch', config=str(source), model=str(checkpoint),
                               device='cpu', dtype='float32', stopOnEntry=True, shareSession=False)
                client.request('configurationDone')
                client.stopped()

                def refused(message, command, **arguments):
                    with self.assertRaisesRegex(AssertionError, message):
                        client.request(command, **arguments)
                    client.request('threads')  # collect the trailing computationFailed event
                    client.events.clear()

                refused('transformer layer', 'llmdbg/assign', target='x', expression='x * 2')
                client.request('setBreakpoints', source={'path': str(source)},
                               breakpoints=[{'line': program['callLines']['attention'], 'condition': 'token=1, layer=1'}])
                client.request('continue', threadId=1)
                client.stopped()

                def residual():
                    return client.request('llmdbg/vector')['vectors']['residual_stream']

                before = residual()
                steer = [float(i) / 10 for i in range(16)]
                defined = client.request('llmdbg/defineVector', name='steer', values=steer)
                self.assertEqual(defined['shape'], [16])
                result = client.request('llmdbg/assign', target='x', expression='x * 2 + steer')
                self.assertTrue(result['patched'])
                self.assertEqual(result['changedValues'], 16)
                after = residual()
                for a, b, s in zip(after, before, steer):
                    self.assertAlmostEqual(a, 2 * b + s, places=4)

                # Slices, and a saved copy of the current residual stream.
                client.request('llmdbg/defineVector', name='saved', expression='x')
                client.request('llmdbg/assign', target='x[0, 0, 3:5]', expression='0')
                sliced = residual()
                self.assertEqual(sliced[3:5], [0.0, 0.0])
                self.assertEqual(sliced[:3] + sliced[5:], after[:3] + after[5:])
                client.request('llmdbg/assign', target='x', expression='saved')
                self.assertEqual([round(v, 4) for v in residual()], [round(v, 4) for v in after])

                # Projection removal: the result is orthogonal to the unit direction d.
                d = [1.0 / 4] * 16
                client.request('llmdbg/defineVector', name='d', values=d)
                client.request('llmdbg/assign', target='x', expression='x - (x @ d) * d')
                self.assertAlmostEqual(sum(a * b for a, b in zip(residual(), d)), 0.0, places=4)
                norm = client.request('evaluate', expression='x.norm()')['result']
                self.assertAlmostEqual(float(norm), math.sqrt(sum(v * v for v in residual())), places=3)

                names = [v['name'] for v in client.request('llmdbg/listVectors')['vectors']]
                self.assertEqual(names, ['steer', 'saved', 'd'])
                for target, expression, message in [
                    ('hidden', 'x', 'Assignable targets'),
                    ('x', 'nonsense_name', 'Unknown tensor expression'),
                    ('x[0, 0, i]', '0', 'integer indices'),
                ]:
                    refused(message, 'llmdbg/assign', target=target, expression=expression)
                refused('built-in', 'llmdbg/defineVector', name='x', values=[1.0])
                client.request('disconnect')
            finally:
                client.close()
                host.close()


class TerminalAssignCommandsTest(unittest.TestCase):
    def test_let_vec_and_vecs_map_to_requests(self):
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock
        from llmdbg.terminal import Terminal
        client = Mock(); client.request.return_value = {}
        terminal = Terminal(client, SimpleNamespace(demo=False)); terminal.launched = True
        terminal.execute('let x[0, 0, :4] = x[0, 0, :4] * 2')
        client.request.assert_called_with('llmdbg/assign', target='x[0, 0, :4]', expression='x[0, 0, :4] * 2')
        terminal.execute('vec base = x')
        client.request.assert_called_with('llmdbg/defineVector', name='base', expression='x')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'steer.json'
            path.write_text(json.dumps([0.5, -1.0]))
            terminal.execute(f'vec steer from {path}')
            client.request.assert_called_with('llmdbg/defineVector', name='steer', values=[0.5, -1.0])
        terminal.execute('vecs')
        client.request.assert_called_with('llmdbg/listVectors')
        for bad in ('let x', 'vec onlyname'):
            with self.assertRaises(ValueError):
                terminal.execute(bad)


if __name__ == '__main__':
    unittest.main()
