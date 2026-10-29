import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from llmdbg.terminal import Terminal, DAPError


class TerminalTest(unittest.TestCase):
    def cli(self,*args,input=None):
        root=Path(__file__).resolve().parents[1]
        env=dict(os.environ,PYTHONPATH=str(root/'src'))
        return subprocess.run([sys.executable,'-m','llmdbg','debug','--demo','--json',*args],
            input=input,text=True,capture_output=True,cwd=root,env=env,timeout=30)

    def test_real_dap_json_commands_and_clean_stdout(self):
        result=self.cli('-c','where','-c','next','-c','print residual_stream','-c','vector','-c','vars')
        self.assertEqual(result.returncode,0,result.stderr)
        records=[json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual([r['command'] for r in records],['launch','where','next','print residual_stream','vector','vars'])
        self.assertTrue(all(r['ok'] for r in records))
        self.assertEqual(records[2]['body']['reason'],'step')
        self.assertEqual(records[4]['body']['dimensions'],8)
        self.assertTrue(records[5]['body']['scopes'])
        self.assertIn('llmdbg:',result.stderr)

    def test_batch_stdin_raw_request_and_termination(self):
        result=self.cli('--batch',input='request threads {}\ncontinue\nquit\n')
        self.assertEqual(result.returncode,0,result.stderr)
        records=[json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(records[1]['body']['threads'][0]['id'],1)
        self.assertEqual(records[2]['body']['event'],'terminated')

    def test_batch_errors_fail_fast(self):
        result=self.cli('--batch',input='nonsense\nnext\n')
        self.assertEqual(result.returncode,1)
        records=[json.loads(line) for line in result.stdout.splitlines()]
        self.assertFalse(records[-1]['ok'])
        self.assertEqual(len(records),2)

    def test_conditional_named_breakpoint_and_clear(self):
        client=Mock();client.request.return_value={'breakpoints':[{'verified':True}]}
        terminal=Terminal(client,SimpleNamespace(demo=False))
        terminal.launched=True;terminal.source='/tmp/example.llmdbg.py';terminal.calls={'mlp':[25,35]}
        terminal.execute('break mlp token=45,layer=9')
        self.assertEqual(client.request.call_args.kwargs['breakpoints'],[
            {'line':25,'condition':'token=45,layer=9'},{'line':35,'condition':'token=45,layer=9'}])
        terminal.execute('clear')
        self.assertEqual(client.request.call_args.kwargs['breakpoints'],[])

    def test_unverified_breakpoint_is_an_error(self):
        client=Mock();client.request.return_value={'breakpoints':[{'verified':False}]}
        terminal=Terminal(client,SimpleNamespace(demo=False));terminal.launched=True;terminal.source='/tmp/x'
        with self.assertRaises(DAPError):terminal.execute('break 40')

    def test_request_rejects_nonobject_arguments(self):
        terminal=Terminal(Mock(),SimpleNamespace(demo=False))
        with self.assertRaises(ValueError):terminal.execute('request threads []')

    def test_lens_variable_and_expression_use_abc_decoder(self):
        client=Mock();client.request.return_value={'rows':[]}
        terminal=Terminal(client,SimpleNamespace(demo=False));terminal.launched=True
        for expression in ('residual_stream','normalized - x','x + 0.5 * output','value[0]'):
            terminal.execute('lens '+expression)
            client.request.assert_called_with('llmdbg/decodeExpression',expression=expression)

    def test_site_based_lens_syntax_is_preserved(self):
        client=Mock();terminal=Terminal(client,SimpleNamespace(demo=False));terminal.launched=True
        terminal.execute('lens j residual-post')
        client.request.assert_called_with('llmdbg/neuronLens',mode='lens:j',view='residual-post')

    def test_external_host_survives_two_terminal_sessions(self):
        from dap_client import DebugHost
        with tempfile.TemporaryDirectory() as directory:
            host=DebugHost(Path(directory))
            try:
                for _ in range(2):
                    result=self.cli('--connect',f'127.0.0.1:{host.port}','-c','next')
                    self.assertEqual(result.returncode,0,result.stderr)
                    self.assertIsNone(host.process.poll())
            finally:host.close()


if __name__=='__main__':unittest.main()
