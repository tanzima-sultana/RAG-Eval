
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import BaseCallbackHandler

from common.constants import CLAUDE_PRICES_PER_MILLION_TOKENS
from common.constants import EVAL_LLM_MAX_TOKENS
from common.constants import EVAL_LLM_TIMEOUT_SECONDS

# 1. ClaudeTokenCounter: counts the tokens of one Claude model's calls
# 2. ClaudeLLM:          one Claude model; generate(prompt) and compute_cost(). Returned by llm.get_llm()

class ClaudeTokenCounter(BaseCallbackHandler):
	"""
	Input: nothing when created.
	Output: an object holding the input tokens, output tokens and number of Claude calls.
	What it does: adds up the tokens of every Claude call it is attached to.
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
		Input: response, the LangChain result of one finished Claude call.
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

class ClaudeLLM:
	"""
	Input: llm_model, a Claude model id, when created.
	Output: an object that sends a prompt to that Claude model and returns its text.
	What it does: calls one Claude model and counts its tokens, so the cost can be computed.
	"""

	def __init__(self, llm_model: str):
		"""
		Input: llm_model, a Claude model id that has a price in CLAUDE_PRICES_PER_MILLION_TOKENS.
		Output: nothing.
		What it does: builds a ChatAnthropic for that model with a token counter attached.
		"""
		self.llm_model = llm_model
		self.token_counter = ClaudeTokenCounter()
		# Without these, ChatAnthropic allows 128000 output tokens and no timeout
		self.claude_llm = ChatAnthropic(
			model=llm_model,
			max_tokens=EVAL_LLM_MAX_TOKENS,
			timeout=EVAL_LLM_TIMEOUT_SECONDS,
			callbacks=[self.token_counter]
		)

	def generate(self, prompt: str) -> str:
		"""
		Input: prompt, the text sent to Claude.
		Output: Claude's reply text.
		What it does: sends the prompt to the Claude model and returns the text of its reply.
		"""
		response = self.claude_llm.invoke(prompt)
		return response.text

	def compute_cost(self) -> float:
		"""
		Input: nothing.
		Output: the cost of all calls made so far, in US dollars.
		What it does: multiplies the counted tokens by the model's prices and prints the tokens and cost.
		"""
		if self.llm_model not in CLAUDE_PRICES_PER_MILLION_TOKENS:
			raise ValueError(f"No price for Claude model {self.llm_model} in CLAUDE_PRICES_PER_MILLION_TOKENS")

		model_prices = CLAUDE_PRICES_PER_MILLION_TOKENS[self.llm_model]
		input_cost = self.token_counter.input_tokens * model_prices["input"] / 1000000
		output_cost = self.token_counter.output_tokens * model_prices["output"] / 1000000
		total_cost = input_cost + output_cost

		print(f"{self.llm_model} calls : {self.token_counter.number_of_calls}")
		print(f"{self.llm_model} input tokens : {self.token_counter.input_tokens}, cost : ${input_cost:.4f}")
		print(f"{self.llm_model} output tokens : {self.token_counter.output_tokens}, cost : ${output_cost:.4f}")
		print(f"{self.llm_model} cost : ${total_cost:.4f}")
		return total_cost
