"""The generation budget belongs to the run: debug(model, inputs, max_new_tokens=N)."""
from pathlib import Path
import tempfile
import textwrap
import unittest

from llmdbg.config import load_debug_config

ROOT = Path(__file__).resolve().parents[1]
GPT2_STARTER = ROOT / 'starters/GPT2-Text.llmdbg.py'


def program(directory: Path, *, inputs: str, debug_args: str = '') -> Path:
    """The GPT-2 starter with its input line and debug(...) call replaced."""
    source = GPT2_STARTER.read_text()
    source = source.replace('inputs = model.tokenize(prompt)', inputs)
    source = source.replace('with debug(model, inputs, max_new_tokens=0) as run:',
                            f'with debug(model, inputs{debug_args}) as run:')
    path = directory / 'program.llmdbg.py'
    path.write_text(source)
    return path


class ParsedBudgetTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def budget(self, **kwargs):
        return load_debug_config(program(self.root, **kwargs))['maxNewTokens']

    def test_tokenize_with_debug_budget(self):
        self.assertEqual(self.budget(inputs='inputs = model.tokenize(prompt)', debug_args=', max_new_tokens=8'), 8)

    def test_tokenize_without_budget_generates_nothing(self):
        self.assertEqual(self.budget(inputs='inputs = model.tokenize(prompt)'), 0)

    def test_debug_budget_wins_over_legacy_chat_budget(self):
        chat = 'inputs = model.chat([{"role": "user", "content": "hi"}], max_new_tokens=128)'
        self.assertEqual(self.budget(inputs=chat, debug_args=', max_new_tokens=5'), 5)
        self.assertEqual(self.budget(inputs=chat), 128)  # older files keep working

    def test_negative_budget_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'non-negative'):
            self.budget(inputs='inputs = model.tokenize(prompt)', debug_args=', max_new_tokens=-1')

    def test_starters_keep_their_budgets_on_debug(self):
        expected = {'GPT2-Text': 0, 'Qwen3-Chat': 128, 'Qwen3-MultiTurn': 256,
                    'Qwen35-Thinking': 4096, 'Qwen35-Vision': 128}
        for name, budget in expected.items():
            path = ROOT / 'starters' / f'{name}.llmdbg.py'
            source = path.read_text()
            self.assertIn(f'with debug(model, inputs, max_new_tokens={budget}) as run:', source, name)
            self.assertNotIn('max_new_tokens=', source.split('with debug(')[0].split('def ')[0], name)
            self.assertEqual(load_debug_config(path)['maxNewTokens'], budget, name)


class InProcessBudgetTest(unittest.TestCase):
    def test_debug_session_budget(self):
        from llmdbg.program import ChatPrompt, DebugSession, Prompt
        self.assertEqual(DebugSession(None, Prompt('x'), None).max_new_tokens, 0)
        self.assertEqual(DebugSession(None, Prompt('x'), None, 4).max_new_tokens, 4)
        chat = ChatPrompt(({'role': 'user', 'content': 'hi'},), max_new_tokens=128)
        self.assertEqual(DebugSession(None, chat, None).max_new_tokens, 128)
        self.assertEqual(DebugSession(None, chat, None, 3).max_new_tokens, 3)
        with self.assertRaises(ValueError):
            DebugSession(None, Prompt('x'), None, -1)


class DebuggerGeneratesFromProgramBudgetTest(unittest.TestCase):
    def test_debug_budget_alone_drives_generation(self):
        from dap_client import DebugHost
        from test_weights import tiny_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = tiny_checkpoint(root)
            source = program(root, inputs='inputs = model.tokenize("w1 w2 w3")', debug_args=', max_new_tokens=2')
            config = load_debug_config(source)
            host = DebugHost(root / 'logs', env_overrides={'CUDA_VISIBLE_DEVICES': ''})
            client = host.client('budget')
            try:
                client.request('initialize', adapterID='llmdbg')
                # No maxNewTokens here: the budget comes from debug(...) in the file.
                client.request('launch', config=str(source), model=str(checkpoint),
                               device='cpu', dtype='float32', stopOnEntry=True, shareSession=False)
                client.request('configurationDone')
                client.stopped()
                prompt = client.request('llmdbg/promptTokens')
                last = len(prompt['tokens']) - 1
                client.request('setBreakpoints', source={'path': str(source)},
                               breakpoints=[{'line': config['callLines']['attention'], 'condition': f'token={last + 2}, layer=0'}])
                client.request('continue', threadId=1)
                client.stopped()
                after = client.request('llmdbg/promptTokens')
                self.assertEqual(after['currentToken'], last + 2)
                self.assertEqual(len(after['tokens']), len(prompt['tokens']) + 2)
                client.request('disconnect')
            finally:
                client.close()
                host.close()


    def test_run_to_end_reports_tokens_generated_after_the_last_stop(self):
        """A continue with no breakpoint generates the budget, then ends without a stop event.
        Editors refresh on stops, so the final sequence must arrive as an event before termination."""
        import time
        from dap_client import DebugHost
        from test_weights import tiny_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = tiny_checkpoint(root)
            source = program(root, inputs='inputs = model.tokenize("w1 w2 w3")', debug_args=', max_new_tokens=2')
            host = DebugHost(root / 'logs', env_overrides={'CUDA_VISIBLE_DEVICES': ''})
            client = host.client('to-end')
            try:
                client.request('initialize', adapterID='llmdbg')
                client.request('launch', config=str(source), model=str(checkpoint),
                               device='cpu', dtype='float32', stopOnEntry=True, shareSession=False)
                client.request('configurationDone')
                client.stopped()
                input_count = len(client.request('llmdbg/promptTokens')['tokens'])
                client.request('continue', threadId=1)
                deadline = time.monotonic() + 60
                while not any(e.get('event') == 'terminated' for e in client.events):
                    client.events.append(client.read(deadline))
                names = [e.get('event') for e in client.events]
                final = next(e for e in client.events if e.get('event') == 'llmdbg/promptTokens')
                self.assertLess(names.index('llmdbg/promptTokens'), names.index('terminated'))
                self.assertTrue(final['body']['ended'])
                self.assertEqual(len(final['body']['tokens']), input_count + 2)
            finally:
                client.close()
                host.close()


if __name__ == '__main__':
    unittest.main()
