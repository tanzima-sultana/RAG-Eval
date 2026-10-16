import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI

from common.args import LLMArgs
from llm.schema import LLMCost
from llm.schema import LLMResponse

# 1. validate_gemini: checks a Gemini model can be used. Called by llm.validate.validate_llm
# 2. GeminiLLM:       one Gemini model; chat_model, json_chat_model, generate(), generate_batch() and compute_cost().
#                     Returned by llm.get_llm()

def validate_gemini(llm_args: LLMArgs) -> bool:
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
	if llm_args.input_price == 0 or llm_args.output_price == 0 or llm_args.cache_write_price == 0 or llm_args.cache_read_price == 0:
		raise ValueError(f"Gemini model ({llm_args.llm_model}) needs an input, output, cache write and cache read price above 0.")

	return True

class GeminiLLM:
	"""
	Input: llm_args, the settings of one Gemini model, and token_counter, when created.
	Output: an object that sends prompts to that Gemini model and returns its text.
	What it does: calls one Gemini model and counts its tokens, so the cost can be computed.
	"""

	def __init__(self, llm_args: LLMArgs, token_counter):
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

	def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
		"""
		Input: system_prompt, the instructions ("" for none); user_prompt, the text sent to Gemini.
		Output: an LLMResponse with the reply text and the tokens of this call. Its request_id is "".
		What it does: sends the system prompt and user prompt to the Gemini model and returns the text of its reply.
		"""
		messages = []
		if system_prompt != "":
			messages.append(("system", system_prompt))
		messages.append(("user", user_prompt))

		response = self.chat_model.invoke(messages)

		# The provider reports the tokens of the call in usage_metadata, or None when it does not
		input_tokens = 0
		output_tokens = 0
		if response.usage_metadata is not None:
			input_tokens = response.usage_metadata["input_tokens"]
			output_tokens = response.usage_metadata["output_tokens"]

		llm_response = LLMResponse(
			llm=self.llm_args.llm,
			llm_model=self.llm_args.llm_model,
			request_id="",
			text=response.text,
			input_tokens=input_tokens,
			output_tokens=output_tokens
		)
		return llm_response

	def generate_batch(self, requests: dict, batch_id_file: str, batch_poll_seconds: int, batch_timeout_seconds: int) -> dict:
		"""
		Input: requests, {request_id: {"system_prompt", "user_prompt"}}; batch_id_file, where the batch id is kept;
		batch_poll_seconds, the seconds between status checks; batch_timeout_seconds, the seconds to wait in all.
		Output: {request_id: LLMResponse} for every request that succeeded.
		What it does: sends all requests inline as one Gemini batch job at the batch price, waits for it to end and
		collects the replies by request id. If batch_id_file exists, it waits for that batch job instead.
		"""
		client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))

		# 1. Resume the batch job of an earlier run, so the same requests are never paid twice
		if os.path.exists(batch_id_file):
			with open(batch_id_file, encoding="utf-8") as f:
				batch_name = f.read().strip()
			print(f"Resuming Gemini batch : {batch_name}")
		else:
			# 2. Send all requests inline as one batch job. Gemini has no custom id, so the request id goes in metadata
			inlined_requests = []
			for request_id in requests:
				request = requests[request_id]
				inlined_requests.append({
					"contents": [{"role": "user", "parts": [{"text": request["user_prompt"]}]}],
					"config": {
						"system_instruction": request["system_prompt"],
						"max_output_tokens": self.llm_args.max_tokens
					},
					"metadata": {"request_id": request_id}
				})
			batch_job = client.batches.create(model=self.llm_args.llm_model, src=inlined_requests)
			batch_name = batch_job.name

			# 3. Save the batch job name at once, so a run that stops while waiting can resume it
			os.makedirs(os.path.dirname(batch_id_file), exist_ok=True)
			with open(batch_id_file, "w", encoding="utf-8") as f:
				f.write(batch_name)
			print(f"Gemini batch sent : {batch_name}, {len(inlined_requests)} requests")

		# 4. Check the status until the batch job has ended
		end_states = ("JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED")
		start_time = time.time()
		while True:
			batch_job = client.batches.get(name=batch_name)
			if batch_job.state.name in end_states:
				break
			if time.time() - start_time > batch_timeout_seconds:
				raise TimeoutError(f"Gemini batch {batch_name} did not end in {batch_timeout_seconds}s. Run again to resume it.")
			print(f"Gemini batch {batch_name} : {batch_job.state.name}")
			time.sleep(batch_poll_seconds)

		# 5. Collect the reply text and tokens of every succeeded request
		replies = {}
		failed_count = 0
		if batch_job.dest is not None and batch_job.dest.inlined_responses is not None:
			for inlined_response in batch_job.dest.inlined_responses:
				if inlined_response.error is not None or inlined_response.response is None:
					failed_count = failed_count + 1
					continue
				request_id = inlined_response.metadata["request_id"]
				usage_metadata = inlined_response.response.usage_metadata
				# prompt_token_count includes the tokens Gemini read from its prompt cache, which it caches on its own
				input_tokens = usage_metadata.prompt_token_count
				output_tokens = usage_metadata.candidates_token_count
				cache_read_tokens = 0
				if usage_metadata.cached_content_token_count is not None:
					cache_read_tokens = usage_metadata.cached_content_token_count
				replies[request_id] = LLMResponse(
					llm=self.llm_args.llm,
					llm_model=self.llm_args.llm_model,
					request_id=request_id,
					text=inlined_response.response.text,
					input_tokens=input_tokens,
					output_tokens=output_tokens
				)
				self.token_counter.number_of_calls = self.token_counter.number_of_calls + 1
				self.token_counter.input_tokens = self.token_counter.input_tokens + input_tokens
				self.token_counter.cache_read_tokens = self.token_counter.cache_read_tokens + cache_read_tokens
				self.token_counter.output_tokens = self.token_counter.output_tokens + output_tokens

		print(f"Gemini batch {batch_name} : {batch_job.state.name}, {len(replies)} succeeded, {failed_count} failed")
		return replies

	def compute_cost(self) -> LLMCost:
		"""
		Input: nothing.
		Output: an LLMCost with the calls, tokens (cached ones apart) and cost of all calls made so far, in US dollars.
		What it does: multiplies the counted tokens by the model's prices, cached input tokens by the cache prices,
		and prints the tokens and cost.
		"""
		llm_model = self.llm_args.llm_model
		# The cached input tokens are priced at the cache prices, the rest at the input price
		uncached_input_tokens = self.token_counter.input_tokens - self.token_counter.cache_creation_tokens - self.token_counter.cache_read_tokens
		input_cost = uncached_input_tokens * self.llm_args.input_price / 1000000
		cache_write_cost = self.token_counter.cache_creation_tokens * self.llm_args.cache_write_price / 1000000
		cache_read_cost = self.token_counter.cache_read_tokens * self.llm_args.cache_read_price / 1000000
		output_cost = self.token_counter.output_tokens * self.llm_args.output_price / 1000000
		total_cost = input_cost + cache_write_cost + cache_read_cost + output_cost

		print(f"{llm_model} calls : {self.token_counter.number_of_calls}")
		print(f"{llm_model} input tokens : {self.token_counter.input_tokens}, of which not cached : {uncached_input_tokens}, cost : ${input_cost:.4f}")
		print(f"{llm_model} cache written tokens : {self.token_counter.cache_creation_tokens}, cost : ${cache_write_cost:.4f}")
		print(f"{llm_model} cache read tokens : {self.token_counter.cache_read_tokens}, cost : ${cache_read_cost:.4f}")
		print(f"{llm_model} output tokens : {self.token_counter.output_tokens}, cost : ${output_cost:.4f}")
		print(f"{llm_model} cost : ${total_cost:.4f}")

		llm_cost = LLMCost(
			llm=self.llm_args.llm,
			llm_model=llm_model,
			number_of_calls=self.token_counter.number_of_calls,
			input_tokens=self.token_counter.input_tokens,
			cache_creation_tokens=self.token_counter.cache_creation_tokens,
			cache_read_tokens=self.token_counter.cache_read_tokens,
			output_tokens=self.token_counter.output_tokens,
			input_cost=input_cost,
			cache_write_cost=cache_write_cost,
			cache_read_cost=cache_read_cost,
			output_cost=output_cost,
			total_cost=total_cost
		)
		return llm_cost
