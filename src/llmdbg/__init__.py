"""LLM layer debugger."""

__version__ = "0.0.3a35"

from llmdbg.program import TensorHandle, debug, gpt2, qwen35

__all__ = ["TensorHandle", "debug", "gpt2", "qwen35", "__version__"]
