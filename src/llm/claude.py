import json
import os
import urllib.error
import urllib.request

from langchain_anthropic import ChatAnthropic

from llm.schema import LLM

# 1. validate_claude: checks a Claude model can be used. Called by llm.validate.validate_llm
# 2. ClaudeLLM:       one Claude model; chat_model, json_chat_model, generate() and compute_cost(). Returned by llm.get_llm()

def validate_claude(llm_args: LLM) -> bool:
	"""
	Input: llm_args, with llm claude.
	Output: True. Raises ValueError for the first check that fails.
	What it does: checks paid calls are allowed, CLAUDE_API_KEY is set and Anthropic lists the model.
	"""
	# 1. Claude calls cost money, so a run must opt in with ALLOW_PAID_API=1
	if os.environ.get("ALLOW_PAID_API") != "1":
		raise ValueError(f"Claude model ({llm_args.llm_model}) calls a paid API. Run again with ALLOW_PAID_API=1 set.")

	# 2. Check the API key
	api_key = os.environ.get("CLAUDE_API_KEY", "")
	if api_key == "":
		raise ValueError("CLAUDE_API_KEY is not set. Add it to .env.")

	# 3. Check Anthropic lists the model. Listing models is free
	request = urllib.request.Request(
		"https://api.anthropic.com/v1/models?limit=1000",
		headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"}
	)
	try:
		with urllib.request.urlopen(request, timeout=10) as response:
			models_body = json.loads(response.read().decode("utf-8"))
	except urllib.error.URLError as error:
		raise ValueError(f"Could not list the Claude models: {error}") from error

	model_names = []
	for model in models_body["data"]:
		model_names.append(model["id"])
	if llm_args.llm_model not in model_names:
		raise ValueError(f"Claude model ({llm_args.llm_model}) is not one of the Anthropic models: {model_names}")

	# 4. Paid models need a price, so the cost of every run can be computed
	if llm_args.input_price == 0 or llm_args.output_price == 0:
		raise ValueError(f"Claude model ({llm_args.llm_model}) needs an input and output price above 0.")

	return True

class ClaudeLLM:
	"""
	Input: llm_args, the settings of one Claude model, and token_counter, when created.
	Output: an object that sends prompts to that Claude model and returns its text.
	What it does: calls one Claude model and counts its tokens, so the cost can be computed.
	"""

	def __init__(self, llm_args: LLM, token_counter):
		"""
		Input: llm_args, the Claude model settings; token_counter, the llm.TokenCounter that counts its tokens.
		Output: nothing.
		What it does: builds a ChatAnthropic for that model with the token counter attached.
		"""
		# Claude calls cost money, so a Claude LLM is never built without ALLOW_PAID_API=1, even if
		# validate_claude was not called first
		if os.environ.get("ALLOW_PAID_API") != "1":
			raise ValueError(f"Claude model ({llm_args.llm_model}) calls a paid API. Run again with ALLOW_PAID_API=1 set.")

		self.llm_args = llm_args
		self.token_counter = token_counter
		# The key is passed from CLAUDE_API_KEY, because ChatAnthropic only looks for ANTHROPIC_API_KEY by itself.
		# Without max_tokens and timeout, ChatAnthropic allows 128000 output tokens and no timeout
		self.chat_model = ChatAnthropic(
			model=llm_args.llm_model,
			api_key=os.environ.get("CLAUDE_API_KEY", ""),
			max_tokens=llm_args.max_tokens,
			timeout=llm_args.timeout_seconds,
			callbacks=[token_counter]
		)
		# Claude follows the JSON format RAGAS's prompts ask for without a JSON mode, so it is the same chat model
		self.json_chat_model = self.chat_model

	def generate(self, system_prompt: str, user_prompt: str) -> str:
		"""
		Input: system_prompt, the instructions ("" for none); user_prompt, the text sent to Claude.
		Output: Claude's reply text.
		What it does: sends the system prompt and user prompt to the Claude model and returns the text of its reply.
		"""
		messages = []
		if system_prompt != "":
			messages.append(("system", system_prompt))
		messages.append(("user", user_prompt))

		response = self.chat_model.invoke(messages)
		return response.text

	def compute_cost(self) -> float:
		"""
		Input: nothing.
		Output: the cost of all calls made so far, in US dollars.
		What it does: multiplies the counted tokens by the model's prices and prints the tokens and cost.
		"""
		llm_model = self.llm_args.llm_model
		input_cost = self.token_counter.input_tokens * self.llm_args.input_price / 1000000
		output_cost = self.token_counter.output_tokens * self.llm_args.output_price / 1000000
		total_cost = input_cost + output_cost

		print(f"{llm_model} calls : {self.token_counter.number_of_calls}")
		print(f"{llm_model} input tokens : {self.token_counter.input_tokens}, cost : ${input_cost:.4f}")
		print(f"{llm_model} output tokens : {self.token_counter.output_tokens}, cost : ${output_cost:.4f}")
		print(f"{llm_model} cost : ${total_cost:.4f}")
		return total_cost
