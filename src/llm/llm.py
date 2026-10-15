from langchain_core.callbacks import BaseCallbackHandler

from common.constants import CHATGPT_LLM
from common.constants import CLAUDE_LLM
from common.constants import GEMINI_LLM
from common.constants import OLLAMA_LLM
from llm.chatgpt import ChatGPTLLM
from llm.claude import ClaudeLLM
from llm.gemini import GeminiLLM
from llm.ollama import OllamaLLM
from llm.schema import LLM

# 1. TokenCounter: counts the tokens of one LLM's calls, for any provider
# 2. get_llm:      builds the LLM of llm_args's provider

class TokenCounter(BaseCallbackHandler):
	"""
	Input: nothing when created.
	Output: an object holding the input tokens, output tokens and number of calls.
	What it does: adds up the tokens of every LangChain chat model call it is attached to.
	"""

	# Run on_llm_end in the calling thread, so totals are never updated from two threads at once
	run_inline = True

	def __init__(self):
		"""
		Input: nothing.
		Output: nothing.
		What it does: starts all counts at 0.
		"""
		super().__init__()
		self.input_tokens = 0
		self.output_tokens = 0
		self.number_of_calls = 0

	def on_llm_end(self, response, **kwargs) -> None:
		"""
		Input: response, the LangChain result of one finished call.
		Output: nothing.
		What it does: adds that call's input and output tokens to the totals.
		"""
		self.number_of_calls = self.number_of_calls + 1
		for generation_list in response.generations:
			for generation in generation_list:
				message = getattr(generation, "message", None)
				if message is None:
					continue
				usage_metadata = getattr(message, "usage_metadata", None)
				if usage_metadata is None:
					continue
				self.input_tokens = self.input_tokens + usage_metadata["input_tokens"]
				self.output_tokens = self.output_tokens + usage_metadata["output_tokens"]

def get_llm(llm_args: LLM) -> ChatGPTLLM | ClaudeLLM | GeminiLLM | OllamaLLM:
	"""
	Input: llm_args, the LLM settings of one layer, already checked by llm.validate.validate_llm.
	Output: a ChatGPTLLM, ClaudeLLM, GeminiLLM or OllamaLLM, with chat_model, json_chat_model, generate(system_prompt, user_prompt) and compute_cost().
	What it does: builds the LLM of llm_args's provider with its own token counter. Raises ValueError for another provider.
	"""
	if llm_args.llm == CLAUDE_LLM:
		return ClaudeLLM(llm_args, TokenCounter())

	if llm_args.llm == CHATGPT_LLM:
		return ChatGPTLLM(llm_args, TokenCounter())

	if llm_args.llm == GEMINI_LLM:
		return GeminiLLM(llm_args, TokenCounter())

	if llm_args.llm == OLLAMA_LLM:
		return OllamaLLM(llm_args, TokenCounter())

	raise ValueError(f"llm ({llm_args.llm}) is not supported. Use {CLAUDE_LLM}, {CHATGPT_LLM}, {GEMINI_LLM} or {OLLAMA_LLM}.")
