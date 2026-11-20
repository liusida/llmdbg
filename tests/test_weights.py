"""Session-scoped weight edits on a tiny CPU GPT-2: exact effects, undo, and restore at session end."""
from pathlib import Path
import tempfile
import unittest

from dap_client import DebugHost

BIAS = "transformer.h.0.mlp.c_proj.bias"


def tiny_checkpoint(root: Path) -> Path:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    checkpoint = root/'tiny-gpt2'
    tokenizer = Tokenizer(WordLevel({'[UNK]': 0, **{f'w{i}': i for i in range(1, 32)}}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]').save_pretrained(checkpoint)
    GPT2LMHeadModel(GPT2Config(vocab_size=32, n_embd=16, n_layer=2, n_head=2, n_positions=64,
                               bos_token_id=None, eos_token_id=None)).eval().save_pretrained(checkpoint)
    return checkpoint


class WeightEditTest(unittest.TestCase):
    def setUp(self):
        from llmdbg.config import load_debug_config
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.checkpoint = tiny_checkpoint(root)
        self.source = Path(__file__).resolve().parents[1]/'starters/GPT2-Text.llmdbg.py'
        self.attention_line = load_debug_config(self.source)['callLines']['attention']
        self.host = DebugHost(root/'logs', env_overrides={'CUDA_VISIBLE_DEVICES': ''})
        self.addCleanup(self.host.close)

    def session(self, name):
        """A client paused at layer 1's attention call for token 1 (x = output of layer 0)."""
        client = self.host.client(name)
        client.request('initialize', adapterID='llmdbg')
        client.request('launch', config=str(self.source), model=str(self.checkpoint),
                       device='cpu', dtype='float32', stopOnEntry=True, shareSession=False)
        client.request('configurationDone')
        client.stopped()
        client.request('setBreakpoints', source={'path': str(self.source)},
                       breakpoints=[{'line': self.attention_line, 'condition': 'token=1, layer=1'}])
        client.request('continue', threadId=1)
        client.stopped()
        return client

    @staticmethod
    def residual(client):
        return client.request('llmdbg/vector')['vectors']['residual_stream']

    def test_edit_undo_and_restore_at_session_end(self):
        client = self.session('first')
        try:
            original = self.residual(client)
            listed = client.request('llmdbg/listParameters', filter='h.0.mlp.c_proj')
            self.assertEqual([p['name'] for p in listed['parameters']],
                             ['transformer.h.0.mlp.c_proj.weight', BIAS])

            # Layer 0's MLP output bias is added to the residual stream at every token,
            # so +1 there must shift x at layer 1 by exactly +1 in every dimension.
            edit = client.request('llmdbg/assign', target=f'weight["{BIAS}"]',
                                  expression=f'weight["{BIAS}"] + 1')
            self.assertEqual((edit['parameter'], edit['changedValues'], edit['roundedValues']), (BIAS, 16, 0))
            for after, before in zip(self.residual(client), original):
                self.assertAlmostEqual(after, before + 1, places=4)
            self.assertEqual(client.request('llmdbg/listWeightEdits')['edits'],
                             [{'parameter': BIAS, 'formulas': [f'weight["{BIAS}"] = weight["{BIAS}"] + 1']}])

            # Slices and the current layer's W_o (layer 1 here); reads see the edited weights.
            client.request('llmdbg/assign', target=f'weight["{BIAS}"][:4]', expression='0')
            self.assertEqual(float(client.request('evaluate', expression=f'weight["{BIAS}"][3]')['result']), 0.0)
            w_o = client.request('llmdbg/assign', target='W_o[:8, :]', expression='W_o[:8, :] * 0')
            self.assertEqual(w_o['parameter'], 'transformer.h.1.attn.c_proj.weight')
            self.assertEqual(float(client.request('evaluate', expression='W_o[0, 0]')['result']), 0.0)

            restored = client.request('llmdbg/restoreWeights', parameter=BIAS)['restored']
            self.assertEqual(restored, [BIAS])
            for after, before in zip(self.residual(client), original):
                self.assertAlmostEqual(after, before, places=5)
            # W_o is still edited; ending the session must restore it too.
            self.assertEqual([e['parameter'] for e in client.request('llmdbg/listWeightEdits')['edits']],
                             ['transformer.h.1.attn.c_proj.weight'])
            client.request('llmdbg/assign', target=f'weight["{BIAS}"]', expression='5')
            client.request('disconnect')
        finally:
            client.close()

        second = self.session('second')
        try:
            self.assertEqual(second.request('llmdbg/listWeightEdits')['edits'], [])
            for after, before in zip(self.residual(second), original):
                self.assertAlmostEqual(after, before, places=5)
            self.assertNotEqual(float(second.request('evaluate', expression='W_o[0, 0]')['result']), 0.0)
            second.request('disconnect')
        finally:
            second.close()

    def test_bad_weight_targets_are_refused(self):
        client = self.session('errors')
        try:
            for target, expression, message in [
                ('weight["no.such.parameter"]', '0', 'Unknown parameter'),
                ('weight[0]', '0', 'Name a parameter'),
                (f'weight["{BIAS}"]', 'x', 'Cannot assign shape'),
                (f'weight["{BIAS}"][:, 3]', '0', 'too many indices|IndexError|index'),
            ]:
                with self.assertRaisesRegex(AssertionError, message):
                    client.request('llmdbg/assign', target=target, expression=expression)
                client.request('threads')
                client.events.clear()
            self.assertEqual(client.request('llmdbg/listWeightEdits')['edits'], [])
            client.request('disconnect')
        finally:
            client.close()


class TerminalWeightCommandsTest(unittest.TestCase):
    def test_augmented_let_and_weight_commands(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from llmdbg.terminal import Terminal
        client = Mock(); client.request.return_value = {}
        terminal = Terminal(client, SimpleNamespace(demo=False)); terminal.launched = True
        terminal.execute('let W_o[:64, :] *= 0.5')
        client.request.assert_called_with('llmdbg/assign', target='W_o[:64, :]', expression='W_o[:64, :] * (0.5)')
        terminal.execute('let x -= 2 * steer')
        client.request.assert_called_with('llmdbg/assign', target='x', expression='x - (2 * steer)')
        terminal.execute('params down_proj')
        client.request.assert_called_with('llmdbg/listParameters', filter='down_proj')
        terminal.execute('weights')
        client.request.assert_called_with('llmdbg/listWeightEdits')
        terminal.execute('undo weights')
        client.request.assert_called_with('llmdbg/restoreWeights')
        terminal.execute('undo weights lm_head.weight')
        client.request.assert_called_with('llmdbg/restoreWeights', parameter='lm_head.weight')


if __name__ == '__main__':
    unittest.main()
