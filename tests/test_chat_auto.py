from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from llmdbg import AUTO_REPLY, qwen35
from llmdbg.chat_auto import AUTO_REPLY_MARKER, generate_reply, normalize_messages, prepare_auto_chat, resolve_auto_chat
from llmdbg.config import load_debug_config


class FakeExecution:
    def __init__(self):
        self.input_ids = torch.tensor([[1,2]])
        self.tokens = ('one','two')
        self.rendered_prompt = '<user>question</user><assistant>'
        self.progress = lambda message: None
        self.cache = object()
        self.auto_chat = None
        self.device = 'cpu'
        self.inputs_embeds = self.position_ids = self.visual_pos_masks = None
        self.deepstack_visual_embeds = self.per_layer_inputs = None
        self.interventions = {};self.ica_interventions = set();self.vector_patches = {}
        self.runtime = SimpleNamespace(final_norm=torch.nn.Identity(),
            tokenizer=SimpleNamespace(eos_token_id=7,decode=lambda ids,**kw: ''.join(map(str,ids)),
                                      encode=lambda text,**kw: [6]*len(text)),
            model=SimpleNamespace(config=SimpleNamespace(),get_output_embeddings=lambda:SimpleNamespace(weight=torch.eye(8))),
            is_multimodal=False)

    @property
    def token_count(self):
        return self.input_ids.shape[1]

    def append_generated_token(self, token):
        self.input_ids = torch.cat((self.input_ids,torch.tensor([[token]])),dim=1)
        self.tokens += (str(token),)

    def _prefill(self, count):
        token = {2:3,3:4}.get(count,7)
        self.runtime.final_norm(torch.nn.functional.one_hot(torch.tensor([[token]]),8).float())


class AutoReplyTest(unittest.TestCase):
    def test_partial_reply_stops_before_followup_and_can_resume_with_more_budget(self):
        execution=FakeExecution()
        execution.runtime.tokenizer.apply_chat_template=lambda messages,**kw: ''.join(
            '<'+m['role']+'>'+m['content']+'</'+m['role']+'>' for m in messages)+'<assistant>'
        messages=[{'role':'user','content':'one'},{'role':'assistant','content':AUTO_REPLY},
                  {'role':'user','content':'reveal'}]
        with patch('llmdbg.model_runtime.prepare_model_execution',return_value=execution):
            prepare_auto_chat('test/model',messages,max_new_tokens=1,revision='main')
        resolve_auto_chat(execution,allow_partial=True)
        self.assertIsNotNone(execution.auto_chat)
        self.assertEqual(execution.input_ids.tolist(),[[1,2,3]])
        self.assertNotIn('reveal',execution.rendered_prompt)
        with patch.object(execution,'_prefill',side_effect=AssertionError('partial tokens must not regenerate')):
            resolve_auto_chat(execution,allow_partial=True)
        execution.auto_chat['budget']=8
        resolve_auto_chat(execution,allow_partial=True)
        self.assertIsNone(execution.auto_chat)
        self.assertEqual(execution.input_ids[0,:4].tolist(),[1,2,3,4])
        self.assertIn('reveal',execution.rendered_prompt)

    def test_repeated_replay_and_last_message_edit_reuse_unpinned_resident_turns(self):
        from copy import copy,deepcopy
        base=FakeExecution()
        base.runtime.tokenizer.apply_chat_template=lambda messages,**kw: ''.join(
            '<'+m['role']+'>'+m['content']+'</'+m['role']+'>' for m in messages)+'<assistant>'
        messages=[{'role':'user','content':'one'},{'role':'assistant','content':AUTO_REPLY},
                  {'role':'user','content':'two'},{'role':'assistant','content':AUTO_REPLY},
                  {'role':'user','content':'reveal'}]
        with patch('llmdbg.model_runtime.prepare_model_execution',return_value=base):
            prepare_auto_chat('test/model',messages,max_new_tokens=8,revision='main')
        def clone(last=None):
            execution=copy(base)
            execution.input_ids=base.input_ids.clone()
            execution.auto_chat=deepcopy(base.auto_chat)
            if last is not None:execution.auto_chat['messages'][-1]['content']=last
            return execution
        with TemporaryDirectory() as directory,patch.dict(os.environ,{'LLMDBG_GENERATION_CACHE_DIR':directory}):
            first=resolve_auto_chat(clone())
            logs=[]
            for last in (None,'reveal differently'):
                repeated=clone(last);repeated.progress=logs.append
                with patch.object(repeated,'_prefill',side_effect=AssertionError('cached turns must not regenerate')):
                    resolve_auto_chat(repeated)
                if last is None:self.assertEqual(first.input_ids.tolist(),repeated.input_ids.tolist())
                else:self.assertIn(last,repeated.rendered_prompt)
            self.assertEqual(sum('resident memory' in line for line in logs),4)
            self.assertFalse(list(Path(directory).glob('*.json')))
            self.assertEqual(base.input_ids.tolist(),[[1,2]])
            changed=clone();changed.auto_chat['messages'][2]['content']='different middle question'
            with patch.object(changed,'_prefill',wraps=changed._prefill) as forward:
                resolve_auto_chat(changed)
                self.assertGreater(forward.call_count,0)

    def test_final_assistant_prefill_api_static_config_and_validation(self):
        messages=[{'role':'user','content':'Reveal.'},{'role':'assistant','content':'The first concept'}]
        model=qwen35('Qwen/Qwen3.5-2B')
        prompt=model.chat(messages,add_generation_prompt=False,continue_final_message=True)
        self.assertTrue(prompt.continue_final_message)
        for supplied,kwargs in ((messages,{'continue_final_message':True}),
                                (messages[:1],{'continue_final_message':True,'add_generation_prompt':False}),
                                (messages,{'continue_final_message':'yes','add_generation_prompt':False})):
            with self.assertRaises(ValueError):
                model.chat(supplied,**kwargs)
        with TemporaryDirectory() as directory:
            source=Path(directory)/'prefill.llmdbg.py'
            source.write_text('''from llmdbg import debug, qwen35
model = qwen35("Qwen/Qwen3.5-2B")
messages = [{"role":"user","content":"Reveal."},{"role":"assistant","content":"The first concept"}]
tokens = model.chat(messages, add_generation_prompt=False, continue_final_message=True)
with debug(model,tokens):
    x = model.token_embedding(tokens)
''')
            config=load_debug_config(str(source))
            self.assertTrue(config['continueFinalMessage'])
            self.assertFalse(config['addGenerationPrompt'])
            source.write_text(source.read_text().replace('add_generation_prompt=False','add_generation_prompt=True'))
            with self.assertRaisesRegex(ValueError,'add_generation_prompt=False'):
                load_debug_config(str(source))

    def test_auto_reply_prefill_continuation_applies_only_to_last_message(self):
        execution=FakeExecution()
        flags=[]
        def template(messages,**kwargs):
            flags.append(kwargs['continue_final_message'])
            rendered=''.join('<'+m['role']+'>'+m['content']+'</'+m['role']+'>' for m in messages)
            return rendered.removesuffix('</assistant>') if kwargs['continue_final_message'] else rendered+'<assistant>'
        execution.runtime.tokenizer.apply_chat_template=template
        messages=[{'role':'user','content':'one'},{'role':'assistant','content':AUTO_REPLY},
                  {'role':'user','content':'two'},{'role':'assistant','content':AUTO_REPLY},
                  {'role':'user','content':'reveal'},{'role':'assistant','content':'The first concept'}]
        def reply(execution,*args,**kwargs):
            execution.append_generated_token(3)
            return [3]
        with patch('llmdbg.model_runtime.prepare_model_execution',return_value=execution) as prepare, \
             patch('llmdbg.chat_auto.generate_reply',side_effect=reply):
            prepare_auto_chat('test/model',messages,max_new_tokens=8,add_generation_prompt=False,continue_final_message=True)
            self.assertFalse(prepare.call_args.kwargs['continue_final_message'])
            self.assertTrue(prepare.call_args.kwargs['add_generation_prompt'])
            resolve_auto_chat(execution)
        self.assertEqual(flags,[False,True])
        self.assertTrue(execution.rendered_prompt.endswith('<assistant>The first concept'))

    def test_program_and_static_config_serialize_the_same_marker(self):
        messages=[{'role':'user','content':'Choose.'},{'role':'assistant','content':AUTO_REPLY},{'role':'user','content':'Reveal.'}]
        chat=qwen35('Qwen/Qwen3.5-2B').chat(messages,max_new_tokens=12)
        self.assertEqual(chat.messages[1]['content'],AUTO_REPLY_MARKER)
        json.dumps(chat.messages)
        self.assertIs(messages[1]['content'],AUTO_REPLY)
        with TemporaryDirectory() as directory:
            source=Path(directory)/'auto.llmdbg.py'
            source.write_text('''from llmdbg import AUTO_REPLY, debug, qwen35
model = qwen35("Qwen/Qwen3.5-2B")
messages = [{"role":"user","content":"Choose."},
            {"role":"assistant","content":AUTO_REPLY},
            {"role":"user","content":"Reveal."}]
tokens = model.chat(messages, max_new_tokens=12, enable_thinking=False)
with debug(model,tokens):
    x = model.token_embedding(tokens)
''')
            config=load_debug_config(str(source))
            self.assertEqual(config['messages'],list(chat.messages))
            self.assertEqual(config['maxNewTokens'],12)
            self.assertFalse(config['enableThinking'])
            source.write_text(source.read_text().replace('AUTO_REPLY},','__import__("os").system("false")},'))
            with self.assertRaises(ValueError):
                load_debug_config(str(source))

    def test_invalid_placeholder_roles_and_missing_followups(self):
        for messages in ([{'role':'user','content':AUTO_REPLY}],
                         [{'role':'assistant','content':AUTO_REPLY},{'role':'user','content':'x'}],
                         [{'role':'user','content':'x'},{'role':'assistant','content':AUTO_REPLY}]):
            with self.assertRaises(ValueError):
                normalize_messages(messages)

    def test_generation_cache_and_eos_and_hook_cleanup(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ,{'LLMDBG_GENERATION_CACHE_DIR':directory}), \
             patch('llmdbg.generation_cache.identity',return_value=('key',True,{'schema':1,'vocabulary_size':8})):
            execution=FakeExecution()
            self.assertEqual(generate_reply(execution,8,'test/model','pinned'),[3,4])
            self.assertEqual(len(execution.runtime.final_norm._forward_hooks),0)
            replay=FakeExecution()
            with patch.object(replay,'_prefill',side_effect=AssertionError('must reuse disk IDs')):
                self.assertEqual(generate_reply(replay,8,'test/model','pinned'),[3,4])
            short=FakeExecution()
            with self.assertRaisesRegex(ValueError,'before EOS'):
                generate_reply(short,1,'test/model','pinned')
            self.assertEqual(len(short.runtime.final_norm._forward_hooks),0)

    def test_followup_appends_template_boundary_without_rebuilding_state(self):
        execution=FakeExecution()
        original_cache=execution.cache
        def template(messages,**kwargs):
            return ''.join('<'+m['role']+'>'+m['content']+'</'+m['role']+'>' for m in messages)+'<assistant>'
        execution.runtime.tokenizer.apply_chat_template=template
        messages=[{'role':'user','content':'question'},{'role':'assistant','content':AUTO_REPLY},
                  {'role':'user','content':'reveal'}]
        def reply(execution,*args,**kwargs):
            execution.append_generated_token(3)
            return [3]
        with patch('llmdbg.model_runtime.prepare_model_execution',return_value=execution) as prepare, \
             patch('llmdbg.chat_auto.generate_reply',side_effect=reply) as generate:
            result=prepare_auto_chat('test/model',messages,max_new_tokens=8,enable_thinking=False)
            generate.assert_not_called()
            self.assertEqual(result.input_ids.tolist(),[[1,2]])
            self.assertEqual(result.rendered_prompt,'<user>question</user><assistant>')
            resolve_auto_chat(result)
        self.assertIs(result,execution)
        self.assertIs(result.cache,original_cache)
        self.assertEqual(result.input_ids[0,:3].tolist(),[1,2,3])
        self.assertTrue(result.rendered_prompt.endswith('3</assistant><user>reveal</user><assistant>'))
        self.assertEqual(prepare.call_args.kwargs['messages'],messages[:1])
        self.assertIs(messages[1]['content'],AUTO_REPLY)

    def test_multiple_auto_turns_and_input_validation(self):
        execution=FakeExecution()
        execution.runtime.tokenizer.apply_chat_template=lambda messages,**kw: ''.join(
            '<'+m['role']+'>'+m['content']+'</'+m['role']+'>' for m in messages)+'<assistant>'
        messages=[{'role':'user','content':'one'},{'role':'assistant','content':AUTO_REPLY},
                  {'role':'user','content':'two'},{'role':'assistant','content':AUTO_REPLY},
                  {'role':'user','content':'three'}]
        with patch('llmdbg.model_runtime.prepare_model_execution',return_value=execution), \
             patch('llmdbg.chat_auto.generate_reply',side_effect=[[3],[4]]) as reply:
            result=prepare_auto_chat('test/model',messages,max_new_tokens=8)
            reply.assert_not_called()
            # Simulate the forward generator appending each intermediate body.
            def append(execution,*args,**kwargs):
                token=3 if execution.auto_chat['current']==1 else 4
                execution.append_generated_token(token)
                return [token]
            reply.side_effect=append
            resolve_auto_chat(result)
        self.assertEqual(reply.call_count,2)
        self.assertIn('3</assistant><user>two',result.rendered_prompt)
        self.assertIn('4</assistant><user>three',result.rendered_prompt)
        with self.assertRaisesRegex(ValueError,'max_new_tokens > 0'):
            prepare_auto_chat('test/model',messages,max_new_tokens=0)
        images=messages[:2]+[{'role':'user','content':[{'type':'image','url':'one.png'}]}]
        with self.assertRaisesRegex(ValueError,'before the first AUTO_REPLY'):
            prepare_auto_chat('test/model',images,max_new_tokens=8)

    def test_debugger_steps_one_reply_token_at_a_time_and_appends_followup_only_at_eos(self):
        import io
        from llmdbg.dap import LLMDebugAdapter, ProtocolStream
        execution=FakeExecution()
        execution.runtime.tokenizer.apply_chat_template=lambda messages,**kw: ''.join(
            '<'+m['role']+'>'+m['content']+'</'+m['role']+'>' for m in messages)+'<assistant>'
        messages=[{'role':'user','content':'one'},{'role':'assistant','content':AUTO_REPLY},
                  {'role':'user','content':'reveal'}]
        with patch('llmdbg.model_runtime.prepare_model_execution',return_value=execution), \
             patch.object(execution,'_prefill',side_effect=AssertionError('no eager forwards')):
            prepare_auto_chat('test/model',messages,max_new_tokens=8)
        self.assertEqual(execution.token_count,2)
        adapter=LLMDebugAdapter(ProtocolStream(io.BytesIO(),io.BytesIO()))
        adapter.execution=execution;adapter.max_new_tokens=8;adapter.generated_tokens=0
        for token in (3,4):
            adapter.predictions=[SimpleNamespace(token_id=token)]
            self.assertTrue(adapter._append_next_generated_token())
            self.assertNotIn('reveal',execution.rendered_prompt)
        self.assertEqual(execution.input_ids.tolist(),[[1,2,3,4]])
        adapter.predictions=[SimpleNamespace(token_id=7)]
        with patch('llmdbg.generation_cache.identity',return_value=('key',False,{})), \
             patch.object(execution,'_prefill',side_effect=AssertionError('EOS must append text only')):
            self.assertTrue(adapter._append_next_generated_token())
        self.assertIsNone(execution.auto_chat)
        self.assertIn('<user>reveal</user>',execution.rendered_prompt)
        self.assertEqual(adapter.prompt,execution.rendered_prompt)
        self.assertEqual(adapter.generated_tokens,0)
        self.assertEqual(execution.input_ids[0,:4].tolist(),[1,2,3,4])

    def test_resuming_a_partial_intermediate_reply_does_not_duplicate_cached_tokens(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ,{'LLMDBG_GENERATION_CACHE_DIR':directory}), \
             patch('llmdbg.generation_cache.identity',return_value=('key',True,{'schema':1,'vocabulary_size':8})):
            generate_reply(FakeExecution(),8,'test/model','pinned')
            partial=FakeExecution();partial.append_generated_token(3)
            with patch.object(partial,'_prefill',side_effect=AssertionError('must use cache')):
                self.assertEqual(generate_reply(partial,8,'test/model','pinned',start=2),[3,4])
            self.assertEqual(partial.input_ids.tolist(),[[1,2,3,4]])

    def test_python_debug_iterator_is_lazy_too(self):
        from llmdbg.program import DebugSession
        execution=FakeExecution()
        execution.runtime.tokenizer.apply_chat_template=lambda messages,**kw: ''.join(
            '<'+m['role']+'>'+m['content']+'</'+m['role']+'>' for m in messages)+'<assistant>'
        messages=[{'role':'user','content':'one'},{'role':'assistant','content':AUTO_REPLY},
                  {'role':'user','content':'reveal'}]
        with patch('llmdbg.model_runtime.prepare_model_execution',return_value=execution):
            prepare_auto_chat('test/model',messages,max_new_tokens=8)
        session=DebugSession.__new__(DebugSession)
        session.execution=execution;session.prompt=SimpleNamespace(max_new_tokens=8)
        execution.runtime.final_norm_name='norm'
        session._states={'norm':SimpleNamespace(output_vector=(1.,))}
        session._refresh=lambda:None
        execution.vocabulary_projection=lambda *args,**kwargs:(None,[SimpleNamespace(
            token_id={2:3,3:4}.get(execution.token_count,7))])
        iterator=session.tokens
        self.assertEqual(next(iterator),0)
        self.assertEqual(execution.token_count,2)
        self.assertNotIn('reveal',execution.rendered_prompt)
        with patch('llmdbg.generation_cache.identity',return_value=('key',False,{})):
            rest=list(iterator)
        self.assertGreater(len(rest),3)
        self.assertEqual(execution.input_ids[0,:4].tolist(),[1,2,3,4])
        self.assertIsNone(execution.auto_chat)


if __name__ == '__main__':
    unittest.main()
