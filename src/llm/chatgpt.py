import json
import os
import urllib.error
import urllib.request

import httpx
from langchain_openai import ChatOpenAI

from llm.schema import LLM

# 1. validate_chatgpt: checks a ChatGPT model can be used. Called by llm.validate.validate_llm
# 2. ChatGPTLLM:       one ChatGPT model; chat_model, json_chat_model, generate() and compute_cost(). Returned by llm.get_llm()

def validate_chatgpt(llm_args: LLM) -> bool:
	"""
	Input: llm_args, with llm chatgpt.
	Output: True. Raises ValueError for the first check that fails.
	What it does: checks paid calls are allowed, CHATGPT_API_KEY is set and OpenAI lists the model.
	"""
	# 1. ChatGPT calls cost money, so a run must opt in with ALLOW_PAID_API=1
	if os.environ.get("ALLOW_PAID_API") != "1":
		raise ValueError(f"ChatGPT model ({llm_args.llm_model}) calls a paid API. Run again with ALLOW_PAID_API=1 set.")

	# 2. Check the API key
	api_key = os.environ.get("CHATGPT_API_KEY", "")
	if api_key == "":
		raise ValueError("CHATGPT_API_KEY is not set. Add it to .env.")

	# 3. Check OpenAI lists the model. Listing models is free
	request = urllib.request.Request(
		"https://api.openai.com/v1/models",
		headers={"Authorization": "Bearer " + api_key}
	)
	try:
		with urllib.request.urlopen(request, timeout=10) as response:
			models_body = json.loads(response.read().decode("utf-8"))
	except urllib.error.URLError as error:
		raise ValueError(f"Could not list the ChatGPT models: {error}") from error

	model_names = []
	for model in models_body["data"]:
		model_names.append(model["id"])
	if llm_args.llm_model not in model_names:
		raise ValueError(f"ChatGPT model ({llm_args.llm_model}) is not one of the OpenAI models.")

	# 4. Paid models need a price, so the cost of every run can be computed
	if llm_args.input_price == 0 or llm_args.output_price == 0:
		raise ValueError(f"ChatGPT model ({llm_args.llm_model}) needs an input and output price above 0.")

	return True

class ChatGPTLLM:
	"""
	Input: llm_args, the settings of one ChatGPT model, and token_counter, when created.
	Output: an object that sends prompts to that ChatGPT model and returns its text.
	What it does: calls one OpenAI model and counts its tokens, so the cost can be computed.
	"""

	def __init__(self, llm_args: LLM, token_counter):
		"""
		Input: llm_args, the ChatGPT model settings; token_counter, the llm.TokenCounter that counts its tokens.
		Output: nothing.
		What it does: builds two ChatOpenAI for that model with the token counter attached:
		chat_model for plain text replies and json_chat_model for replies that must be JSON.
		"""
		# ChatGPT calls cost money, so a ChatGPT LLM is never built without ALLOW_PAID_API=1, even if
		# validate_chatgpt was not called first
		if os.environ.get("ALLOW_PAID_API") != "1":
			raise ValueError(f"ChatGPT model ({llm_args.llm_model}) calls a paid API. Run again with ALLOW_PAID_API=1 set.")

		self.llm_args = llm_args
		self.token_counter = token_counter
		api_key = os.environ.get("CHATGPT_API_KEY", "")
		# RAGAS runs each step in a new event loop. ChatOpenAI keeps its HTTPS connections open between calls, and a
		# connection left from the previous step's closed loop fails the next call with "Event loop is closed".
		# With keep-alive off, every call opens its own connection, so none outlives its loop
		self.chat_model = ChatOpenAI(
			model=llm_args.llm_model,
			api_key=api_key,
			max_tokens=llm_args.max_tokens,
			timeout=llm_args.timeout_seconds,
			http_async_client=httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=0)),
			callbacks=[token_counter]
		)
		# The same model in OpenAI's JSON mode, so every reply is valid JSON. Used by RAGAS in QnA, whose prompts ask for JSON
		self.json_chat_model = ChatOpenAI(
			model=llm_args.llm_model,
			api_key=api_key,
			max_tokens=llm_args.max_tokens,
			timeout=llm_args.timeout_seconds,
			model_kwargs={"response_format": {"type": "json_object"}},
			http_async_client=httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=0)),
			callbacks=[token_counter]
		)

	def generate(self, system_prompt: str, user_prompt: str) -> str:
		"""
		Input: system_prompt, the instructions ("" for none); user_prompt, the text sent to ChatGPT.
		Output: ChatGPT's reply text.
		What it does: sends the system prompt and user prompt to the ChatGPT model and returns the text of its reply.
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
