"""Real-checkpoint DAP smoke suite. Explicitly opt in before a major release."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
import unittest

from dap_client import DebugHost


@unittest.skipUnless(os.environ.get('LLMDBG_RELEASE_TESTS') == '1',
                     'opt-in GPU release suite (set LLMDBG_RELEASE_TESTS=1)')
class StartersReleaseTest(unittest.TestCase):
    def test_all_curated_demos_through_dap(self):
        # Imports and subprocess creation stay behind the opt-in gate.
        import torch
        from llmdbg.config import load_debug_config
        self.assertTrue(torch.cuda.is_available(), 'Release suite requires CUDA')
        root = Path(__file__).resolve().parents[2]
        stamp = time.strftime('%Y%m%d-%H%M%S')+f'-{os.getpid()}'
        logs = root/'build'/'test-results'/('starters-'+stamp)
        timeout = float(os.environ.get('LLMDBG_RELEASE_TIMEOUT', '1800'))
        host = DebugHost(logs, timeout=timeout)
        self.addCleanup(host.close)
        summary = []
        # Keep identical model IDs adjacent so the host reuses their weights.
        files = ['GPT2-Text.llmdbg.py', 'Qwen3-Chat.llmdbg.py', 'Qwen3-MultiTurn.llmdbg.py',
                 'Qwen35-Vision.llmdbg.py', 'Qwen35-Thinking.llmdbg.py']
        print(f'\nRelease debugger logs: {logs}', flush=True)
        for filename in files:
            with self.subTest(demo=filename):
                started = time.monotonic()
                record = {'demo': filename, 'passed': False}
                client = host.client(filename)
                try:
                    source = root/'starters'/filename
                    config = load_debug_config(source)
                    client.request('initialize', adapterID='llmdbg')
                    launch = {'config': str(source), 'stopOnEntry': True, 'device': 'cuda'}
                    if config['modelKind'] == 'gpt2':
                        # Raw tokenize() has no generation setting; exercise two extra tokens.
                        launch['maxNewTokens'] = 2
                    client.request('launch', **launch)
                    record['model'] = config['model']
                    record['max_new_tokens'] = launch.get('maxNewTokens', config['maxNewTokens'])
                    record['model_loaded'] = next((event.get('body') for event in client.events
                                                  if event.get('event') == 'llmdbg/modelLoaded'), None)
                    self.assertIsNotNone(record['model_loaded'])
                    client.request('configurationDone')
                    client.stopped()
                    initial = client.request('llmdbg/promptTokens')
                    self.assertTrue(initial['tokens'])
                    if 'Vision' in filename:
                        self.assertTrue(initial['images'], 'Vision demo must expose image regions')
                    if 'Thinking' in filename:
                        self.assertTrue(config['enableThinking'])
                        self.assertIn('<think>', initial['prompt'])
                    # Enter attention, inspect locals, then step into the MLP.
                    client.request('next', threadId=1)
                    client.stopped()
                    client.request('stepIn', threadId=1)
                    client.stopped()
                    frames = client.request('stackTrace', threadId=1)['stackFrames']
                    self.assertEqual(Path(frames[0]['source']['path']), source)
                    scopes = client.request('scopes', frameId=frames[0]['id'])['scopes']
                    variables = [v for scope in scopes for v in client.request('variables',
                        variablesReference=scope['variablesReference'])['variables']]
                    self.assertTrue(variables, 'Attention locals must be inspectable')
                    client.request('stepOut', threadId=1)
                    client.stopped()
                    vector = client.request('llmdbg/vector')
                    self.assertGreater(vector['dimensions'], 0)
                    self.assertTrue(all(math.isfinite(v) for v in vector['vectors']['residual_stream']))
                    # Stop on the MLP call regardless of internal helper nesting.
                    mlp_line = config['callLines']['mlp']
                    bp = client.request('setBreakpoints', source={'path': str(source)},
                        breakpoints=[{'line': mlp_line, 'condition': 'token=0, layer=0'}])
                    self.assertTrue(bp['breakpoints'][0]['verified'])
                    client.request('continue', threadId=1)
                    client.stopped()
                    client.request('stepIn', threadId=1)
                    client.stopped()
                    mlp_frames = client.request('stackTrace', threadId=1)['stackFrames']
                    self.assertIn('mlp', mlp_frames[0]['name'])
                    # Predict at the last known prompt position, skipping unneeded positions.
                    prediction_line = max(config['callLines']['nop'])
                    deadline = time.monotonic()+timeout
                    while True:
                        self.assertLess(time.monotonic(), deadline, 'AUTO_REPLY turns exceeded timeout')
                        prompt = client.request('llmdbg/promptTokens')
                        last = len(prompt['tokens'])-1
                        bp = client.request('setBreakpoints', source={'path': str(source)},
                            breakpoints=[{'line': prediction_line, 'condition': f'token>={last}'}])
                        self.assertTrue(bp['breakpoints'][0]['verified'])
                        client.request('continue', threadId=1)
                        client.stopped()
                        current = client.request('llmdbg/promptTokens')
                        if 'MultiTurn' not in filename or current['prompt'].endswith('One useful improvement is'):
                            break
                    self.assertIn('tokens', client.request('evaluate', expression='predictions')['result'])
                    self.assertTrue(client.request('evaluate', expression='predictions[0].token')['result'])
                    if 'MultiTurn' in filename:
                        self.assertIn('Using that name', current['prompt'])
                        self.assertIn('Suggest one useful improvement', current['prompt'])
                        self.assertGreater(len(current['tokens']), len(initial['tokens']))
                    # Next advances across the token boundary into generated-token execution.
                    before = current['currentToken']
                    client.request('setBreakpoints', source={'path': str(source)}, breakpoints=[])
                    for _ in range(4):
                        client.request('next', threadId=1)
                        client.stopped()
                        after = client.request('llmdbg/promptTokens')
                        if after['currentToken'] > before:
                            break
                    self.assertGreater(after['currentToken'], before)
                    self.assertGreater(len(after['tokens']), len(current['tokens']))
                    generated_vector = client.request('llmdbg/vector')
                    self.assertEqual(generated_vector['tokenPosition'], after['currentToken'])
                    self.assertEqual(generated_vector['dimensions'], vector['dimensions'])
                    self.assertTrue(all(math.isfinite(v) for v in generated_vector['vectors']['residual_stream']))
                    record.update(passed=True, prompt_tokens=len(current['tokens']),
                                  generated_token=after['currentToken'])
                except BaseException as error:
                    record['error'] = repr(error)
                    raise
                finally:
                    try:
                        if record['passed']:
                            client.request('disconnect')
                    finally:
                        client.close()
                        record['seconds'] = time.monotonic()-started
                        summary.append(record)
                        (logs/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
                        print(f'{filename}: {"PASS" if record["passed"] else "FAIL"}', flush=True)
                        if not record['passed']:
                            # A timed-out request may still be running. Kill only our
                            # private host before the next case; do not queue more work.
                            host.close()
                            if filename != files[-1]:
                                host = DebugHost(logs/('after-'+filename), timeout=timeout)
                                self.addCleanup(host.close)
