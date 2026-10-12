from common.constants import CLAUDE_LLM_GENERATOR
from llm.claude import ClaudeLLMGenerator

def get_llm_generator(llm_generator: str) -> ClaudeLLMGenerator:
	"""
	Input: llm_generator, the LLM provider name, e.g. "claude".
	Output: that provider's LLM generator, holding transforms_llm, question_llm and compute_cost().
	What it does: picks and builds the LLM generator for the provider. Raises ValueError if it is not supported.
	"""
	if llm_generator == CLAUDE_LLM_GENERATOR:
		return ClaudeLLMGenerator()

	raise ValueError(f"llm_generator ({llm_generator}) is not supported. Use \"{CLAUDE_LLM_GENERATOR}\".")
