from langchain_core.callbacks import BaseCallbackHandler

from common.constants import CHATGPT_LLM
from common.constants import CLAUDE_LLM
from common.constants import GEMINI_LLM
from common.constants import OLLAMA_LLM
from llm.chatgpt import ChatGPTLLM
from llm.claude import ClaudeLLM
from llm.gemini import GeminiLLM
from llm.ollama import OllamaLLM
from common.args import LLMArgs

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
		What it does: starts all counts at 0. input_tokens counts every input token, cached ones included;
		cache_creation_tokens and cache_read_tokens count the part written to and read from the prompt cache.
		"""
		super().__init__()
		self.input_tokens = 0
		self.cache_creation_tokens = 0
		self.cache_read_tokens = 0
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
				# LangChain's input_tokens already includes the cached tokens, for every provider
				self.input_tokens = self.input_tokens + usage_metadata["input_tokens"]
				self.output_tokens = self.output_tokens + usage_metadata["output_tokens"]

				# The cached part, when the provider reports it. Claude may report its cache writes by lifetime
				# (ephemeral_5m_input_tokens, ephemeral_1h_input_tokens) instead of cache_creation, so all three are added
				input_token_details = usage_metadata.get("input_token_details")
				if input_token_details is None:
					continue
				for cache_creation_key in ("cache_creation", "ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"):
					cache_creation_tokens = input_token_details.get(cache_creation_key)
					if cache_creation_tokens is not None:
						self.cache_creation_tokens = self.cache_creation_tokens + cache_creation_tokens
				cache_read_tokens = input_token_details.get("cache_read")
				if cache_read_tokens is not None:
					self.cache_read_tokens = self.cache_read_tokens + cache_read_tokens

def get_llm(llm_args: LLMArgs) -> ChatGPTLLM | ClaudeLLM | GeminiLLM | OllamaLLM:
	"""
	Input: llm_args, the LLM settings of one layer, already checked by llm.validate.validate_llm.
	Output: a ChatGPTLLM, ClaudeLLM, GeminiLLM or OllamaLLM, with chat_model, json_chat_model, generate(system_prompt, user_prompt), generate_batch() and compute_cost().
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
