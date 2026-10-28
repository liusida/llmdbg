"""Curated demos must remain standalone, parseable debugger programs."""
from pathlib import Path
import unittest

from llmdbg.chat_auto import is_auto_reply
from llmdbg.config import load_debug_config


class PlaygroundTest(unittest.TestCase):
    root = Path(__file__).resolve().parents[1] / 'playground'

    def test_curated_configs_and_source_plans(self):
        for filename, kind in {
            'GPT2-Text.llmdbg.py': 'gpt2',
            'Qwen3-Chat.llmdbg.py': 'qwen3',
            'Qwen35-Vision.llmdbg.py': 'qwen35',
            'Qwen35-Thinking.llmdbg.py': 'qwen35',
            'Qwen3-MultiTurn.llmdbg.py': 'qwen3',
        }.items():
            with self.subTest(filename=filename):
                config = load_debug_config(self.root / filename)
                self.assertEqual(config['modelKind'], kind)
                self.assertTrue(config['sourcePlans'])
                self.assertIn('mlp', config['sourcePlans'])
                if kind == 'qwen35':
                    self.assertIn('gated_linear_attention', config['sourcePlans'])
                self.assertNotIn('SecretConcept', filename)

    def test_vision_has_a_bundled_image_and_disables_thinking(self):
        config = load_debug_config(self.root / 'Qwen35-Vision.llmdbg.py')
        self.assertFalse(config['enableThinking'])
        for message in config['messages']:
            if isinstance(message['content'], list):
                for part in message['content']:
                    if part['type'] == 'image':
                        self.assertTrue((self.root / part['url']).is_file())

    def test_thinking_is_text_only_with_a_larger_budget(self):
        config = load_debug_config(self.root / 'Qwen35-Thinking.llmdbg.py')
        self.assertTrue(config['enableThinking'])
        self.assertEqual(config['maxNewTokens'], 4096)
        self.assertTrue(all(isinstance(m['content'], str) for m in config['messages']))

    def test_multi_turn_has_lazy_replies_and_an_assistant_prefill(self):
        config = load_debug_config(self.root / 'Qwen3-MultiTurn.llmdbg.py')
        self.assertEqual(sum(is_auto_reply(m['content']) for m in config['messages']), 2)
        self.assertEqual(config['messages'][-1]['role'], 'assistant')
        self.assertTrue(config['messages'][-1]['content'])
        self.assertFalse(config['addGenerationPrompt'])
        self.assertTrue(config['continueFinalMessage'])
