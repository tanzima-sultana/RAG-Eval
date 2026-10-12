
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import BaseCallbackHandler
from ragas.llms import LangchainLLMWrapper

from common.constants import CLAUDE_PRICES_PER_MILLION_TOKENS
from common.constants import EVAL_LLM_MAX_TOKENS
from common.constants import EVAL_LLM_TIMEOUT_SECONDS
from common.constants import CLAUDE_TRANSFORMS_MODEL
from common.constants import CLAUDE_QUESTION_MODEL

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

def get_ragas_claude_llm(model_name: str, token_counter: ClaudeTokenCounter) -> LangchainLLMWrapper:
	"""
	Input: model_name, a Claude model id; token_counter, a ClaudeTokenCounter.
	Output: a RAGAS LLM that calls that Claude model.
	What it does: builds a ChatAnthropic with token_counter attached and wraps it for RAGAS.
	"""
	# Without these, ChatAnthropic allows 128000 output tokens and no timeout
	claude_llm = ChatAnthropic(
		model=model_name,
		max_tokens=EVAL_LLM_MAX_TOKENS,
		timeout=EVAL_LLM_TIMEOUT_SECONDS,
		callbacks=[token_counter]
	)
	# RAGAS sets temperature 0.01 by default, which the Claude 5.5 models reject
	ragas_claude_llm = LangchainLLMWrapper(claude_llm, bypass_temperature=True)
	return ragas_claude_llm

def compute_claude_cost(model_name: str, token_counter: ClaudeTokenCounter) -> float:
	"""
	Input: model_name, the Claude model id; token_counter, the ClaudeTokenCounter of the calls.
	Output: the cost of those calls in US dollars.
	What it does: multiplies the counted tokens by the model's prices and prints the tokens and cost.
	"""
	if model_name not in CLAUDE_PRICES_PER_MILLION_TOKENS:
		raise ValueError(f"No price for Claude model {model_name} in CLAUDE_PRICES_PER_MILLION_TOKENS")

	model_prices = CLAUDE_PRICES_PER_MILLION_TOKENS[model_name]
	input_cost = token_counter.input_tokens * model_prices["input"] / 1000000
	output_cost = token_counter.output_tokens * model_prices["output"] / 1000000
	total_cost = input_cost + output_cost

	print(f"{model_name} calls : {token_counter.number_of_calls}")
	print(f"{model_name} input tokens : {token_counter.input_tokens}, cost : ${input_cost:.4f}")
	print(f"{model_name} output tokens : {token_counter.output_tokens}, cost : ${output_cost:.4f}")
	print(f"{model_name} cost : ${total_cost:.4f}")
	return total_cost

class ClaudeLLMGenerator:
	"""
	Input: nothing when created.
	Output: an object holding the two RAGAS Claude LLMs and their token counters.
	What it does: builds the knowledge graph LLM and the question LLM, and computes their cost.
	"""

	def __init__(self):
		"""
		Input: nothing.
		Output: nothing.
		What it does: builds transforms_llm and question_llm, each with its own token counter.
		"""
		# A cheaper model builds the knowledge graph, a stronger one writes the questions
		self.transforms_token_counter = ClaudeTokenCounter()
		self.transforms_llm = get_ragas_claude_llm(CLAUDE_TRANSFORMS_MODEL, self.transforms_token_counter)
		self.question_token_counter = ClaudeTokenCounter()
		self.question_llm = get_ragas_claude_llm(CLAUDE_QUESTION_MODEL, self.question_token_counter)

	def compute_cost(self) -> float:
		"""
		Input: nothing.
		Output: the total cost of both models' calls in US dollars.
		What it does: prints the tokens and cost of each model and the total cost.
		"""
		transforms_cost = compute_claude_cost(CLAUDE_TRANSFORMS_MODEL, self.transforms_token_counter)
		question_cost = compute_claude_cost(CLAUDE_QUESTION_MODEL, self.question_token_counter)
		total_cost = transforms_cost + question_cost
		print(f"Claude total cost : ${total_cost:.4f}")
		return total_cost
