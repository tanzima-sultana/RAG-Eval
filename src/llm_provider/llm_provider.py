import litellm

from langchain_litellm import ChatLiteLLM
from langchain_core.callbacks import BaseCallbackHandler

from common.args import OllamaArgs
from llm_provider.validate import validate_llm
from schema.llm import LLMArgs
from schema.llm import LLMCost
from schema.llm import LLMResponse


class TokenCounter(BaseCallbackHandler):
	# Run on_llm_end in the calling thread, so the totals are never updated from two threads at once
	run_inline = True

	def __init__(self):
		"""
		Input: none.
		Output: nothing.
		What it does: starts every count at 0.
		"""
		super().__init__()
		self.number_of_calls = 0
		self.input_tokens = 0
		self.cache_creation_tokens = 0
		self.cache_read_tokens = 0
		self.output_tokens = 0

	def on_llm_end(self, response, **kwargs) -> None:
		"""
		Input: response, the LangChain result of one finished chat model call.
		Output: nothing.
		What it does: adds that call's input, cached and output tokens to the totals.
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
				# input_tokens already includes the cached tokens
				self.input_tokens = self.input_tokens + usage_metadata["input_tokens"]
				self.output_tokens = self.output_tokens + usage_metadata["output_tokens"]

				# The cached part, when the provider reports it
				input_token_details = usage_metadata.get("input_token_details")
				if input_token_details is None:
					continue
				cache_creation_tokens = input_token_details.get("cache_creation")
				if cache_creation_tokens is not None:
					self.cache_creation_tokens = self.cache_creation_tokens + cache_creation_tokens
				cache_read_tokens = input_token_details.get("cache_read")
				if cache_read_tokens is not None:
					self.cache_read_tokens = self.cache_read_tokens + cache_read_tokens


class LLMProvider:
	def __init__(self, llm_args: LLMArgs, ollama_args: OllamaArgs):
		"""
		Input: llm_args, the LLM settings of one layer; ollama_args, the Ollama settings, used only for an Ollama model.
		Output: nothing.
		What it does: validates the settings, so a paid model is never built without ALLOW_PAID_API=1, and builds a
		LangChain ChatLiteLLM for the model with a token counter attached. An Ollama model also gets the server URL
		and context length from ollama_args.
		"""
		validate_llm(llm_args, ollama_args)

		self.llm_args = llm_args
		self.token_counter = TokenCounter()

		# RAGAS wraps this chat model in LangchainLLMWrapper. generate() uses it too, so every call is counted the same way
		provider = llm_args.llm_model.split("/", 1)[0]
		if provider == "ollama" or provider == "ollama_chat":
			self.chat_model = ChatLiteLLM(
				model=llm_args.llm_model,
				max_tokens=llm_args.max_tokens,
				request_timeout=llm_args.timeout_seconds,
				api_base=ollama_args.ollama_api_base,
				model_kwargs={"num_ctx": ollama_args.ollama_context_length},
				callbacks=[self.token_counter]
			)
		else:
			self.chat_model = ChatLiteLLM(
				model=llm_args.llm_model,
				max_tokens=llm_args.max_tokens,
				request_timeout=llm_args.timeout_seconds,
				callbacks=[self.token_counter]
			)

	def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
		"""
		Input: system_prompt, the instructions ("" for none); user_prompt, the text sent to the model.
		Output: an LLMResponse with the reply text and the tokens of this call.
		What it does: sends the prompts to the model and returns its reply.
		"""
		messages = []
		if system_prompt != "":
			messages.append(("system", system_prompt))
		messages.append(("user", user_prompt))

		response = self.chat_model.invoke(messages)

		input_tokens = 0
		output_tokens = 0
		if response.usage_metadata is not None:
			input_tokens = response.usage_metadata["input_tokens"]
			output_tokens = response.usage_metadata["output_tokens"]

		return LLMResponse(
			llm_model=self.llm_args.llm_model,
			text=response.text,
			input_tokens=input_tokens,
			output_tokens=output_tokens
		)

	def compute_cost(self) -> LLMCost:
		"""
		Input: none.
		Output: an LLMCost with the calls, tokens and cost of every call made so far, in US dollars.
		What it does: prices the counted tokens with LiteLLM's price map, prints them and returns them.
		A local Ollama model costs 0.
		"""
		llm_model = self.llm_args.llm_model
		token_counter = self.token_counter

		# prompt_tokens is every input token. LiteLLM prices the cached part of it at the cache prices
		input_cost, output_cost = litellm.cost_per_token(
			model=llm_model,
			prompt_tokens=token_counter.input_tokens,
			completion_tokens=token_counter.output_tokens,
			cache_creation_input_tokens=token_counter.cache_creation_tokens,
			cache_read_input_tokens=token_counter.cache_read_tokens
		)
		total_cost = input_cost + output_cost

		print(f"{llm_model} calls : {token_counter.number_of_calls}")
		print(
			f"{llm_model} input tokens : {token_counter.input_tokens} "
			f"(cache written : {token_counter.cache_creation_tokens}, cache read : {token_counter.cache_read_tokens})"
		)
		print(f"{llm_model} output tokens : {token_counter.output_tokens}")
		print(f"{llm_model} cost : ${total_cost:.4f}")

		return LLMCost(
			llm_model=llm_model,
			number_of_calls=token_counter.number_of_calls,
			input_tokens=token_counter.input_tokens,
			cache_creation_tokens=token_counter.cache_creation_tokens,
			cache_read_tokens=token_counter.cache_read_tokens,
			output_tokens=token_counter.output_tokens,
			total_cost=total_cost
		)
