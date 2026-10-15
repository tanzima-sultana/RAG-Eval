import json
import os
import urllib.error
import urllib.parse
import urllib.request

from langchain_google_genai import ChatGoogleGenerativeAI

from llm.schema import LLM

# 1. validate_gemini: checks a Gemini model can be used. Called by llm.validate.validate_llm
# 2. GeminiLLM:       one Gemini model; chat_model, json_chat_model, generate() and compute_cost(). Returned by llm.get_llm()

def validate_gemini(llm_args: LLM) -> bool:
	"""
	Input: llm_args, with llm gemini.
	Output: True. Raises ValueError for the first check that fails.
	What it does: checks paid calls are allowed, GEMINI_API_KEY is set and Google lists the model.
	"""
	# 1. Gemini calls cost money, so a run must opt in with ALLOW_PAID_API=1
	if os.environ.get("ALLOW_PAID_API") != "1":
		raise ValueError(f"Gemini model ({llm_args.llm_model}) calls a paid API. Run again with ALLOW_PAID_API=1 set.")

	# 2. Check the API key
	api_key = os.environ.get("GEMINI_API_KEY", "")
	if api_key == "":
		raise ValueError("GEMINI_API_KEY is not set. Add it to .env.")

	# 3. Check Google lists the model. Listing models is free. The list comes in pages, and each
	# name starts with "models/"
	model_names = []
	page_token = ""
	while True:
		query = {"key": api_key, "pageSize": "1000"}
		if page_token != "":
			query["pageToken"] = page_token
		url = "https://generativelanguage.googleapis.com/v1beta/models?" + urllib.parse.urlencode(query)
		try:
			with urllib.request.urlopen(url, timeout=10) as response:
				models_body = json.loads(response.read().decode("utf-8"))
		except urllib.error.URLError as error:
			raise ValueError(f"Could not list the Gemini models: {error}") from error

		for model in models_body.get("models", []):
			model_names.append(model["name"].removeprefix("models/"))

		page_token = models_body.get("nextPageToken", "")
		if page_token == "":
			break

	if llm_args.llm_model not in model_names:
		raise ValueError(f"Gemini model ({llm_args.llm_model}) is not one of the Gemini models: {model_names}")

	# 4. Paid models need a price, so the cost of every run can be computed
	if llm_args.input_price == 0 or llm_args.output_price == 0:
		raise ValueError(f"Gemini model ({llm_args.llm_model}) needs an input and output price above 0.")

	return True

class GeminiLLM:
	"""
	Input: llm_args, the settings of one Gemini model, and token_counter, when created.
	Output: an object that sends prompts to that Gemini model and returns its text.
	What it does: calls one Gemini model and counts its tokens, so the cost can be computed.
	"""

	def __init__(self, llm_args: LLM, token_counter):
		"""
		Input: llm_args, the Gemini model settings; token_counter, the llm.TokenCounter that counts its tokens.
		Output: nothing.
		What it does: builds two ChatGoogleGenerativeAI for that model with the token counter attached:
		chat_model for plain text replies and json_chat_model for replies that must be JSON.
		"""
		# Gemini calls cost money, so a Gemini LLM is never built without ALLOW_PAID_API=1, even if
		# validate_gemini was not called first
		if os.environ.get("ALLOW_PAID_API") != "1":
			raise ValueError(f"Gemini model ({llm_args.llm_model}) calls a paid API. Run again with ALLOW_PAID_API=1 set.")

		self.llm_args = llm_args
		self.token_counter = token_counter
		# The key is passed from GEMINI_API_KEY, so a GOOGLE_API_KEY set for something else is never used
		api_key = os.environ.get("GEMINI_API_KEY", "")
		self.chat_model = ChatGoogleGenerativeAI(
			model=llm_args.llm_model,
			google_api_key=api_key,
			max_output_tokens=llm_args.max_tokens,
			timeout=llm_args.timeout_seconds,
			callbacks=[token_counter]
		)
		# The same model with JSON output, so every reply is valid JSON. Used by RAGAS in QnA, whose prompts ask for JSON
		self.json_chat_model = ChatGoogleGenerativeAI(
			model=llm_args.llm_model,
			google_api_key=api_key,
			max_output_tokens=llm_args.max_tokens,
			timeout=llm_args.timeout_seconds,
			response_mime_type="application/json",
			callbacks=[token_counter]
		)

	def generate(self, system_prompt: str, user_prompt: str) -> str:
		"""
		Input: system_prompt, the instructions ("" for none); user_prompt, the text sent to Gemini.
		Output: Gemini's reply text.
		What it does: sends the system prompt and user prompt to the Gemini model and returns the text of its reply.
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
