import json
import os
import time
import urllib.error
import urllib.request

import httpx
import openai
from langchain_openai import ChatOpenAI

from llm.schema import LLMArgs
from llm.schema import LLMCost
from llm.schema import LLMResponse

# 1. validate_chatgpt: checks a ChatGPT model can be used. Called by llm.validate.validate_llm
# 2. ChatGPTLLM:       one ChatGPT model; chat_model, json_chat_model, generate(), generate_batch() and compute_cost().
#                      Returned by llm.get_llm()

def validate_chatgpt(llm_args: LLMArgs) -> bool:
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
	if llm_args.input_price == 0 or llm_args.output_price == 0 or llm_args.cache_write_price == 0 or llm_args.cache_read_price == 0:
		raise ValueError(f"ChatGPT model ({llm_args.llm_model}) needs an input, output, cache write and cache read price above 0.")

	return True

class ChatGPTLLM:
	"""
	Input: llm_args, the settings of one ChatGPT model, and token_counter, when created.
	Output: an object that sends prompts to that ChatGPT model and returns its text.
	What it does: calls one OpenAI model and counts its tokens, so the cost can be computed.
	"""

	def __init__(self, llm_args: LLMArgs, token_counter):
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

	def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
		"""
		Input: system_prompt, the instructions ("" for none); user_prompt, the text sent to ChatGPT.
		Output: an LLMResponse with the reply text and the tokens of this call. Its request_id is "".
		What it does: sends the system prompt and user prompt to the ChatGPT model and returns the text of its reply.
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
		What it does: uploads all requests as one JSONL file, runs them as one OpenAI batch at the batch price, waits
		for it to end and collects the replies by request id. If batch_id_file exists, it waits for that batch instead.
		"""
		client = openai.OpenAI(api_key=os.environ.get("CHATGPT_API_KEY", ""), timeout=self.llm_args.timeout_seconds)

		# 1. Resume the batch of an earlier run, so the same requests are never paid twice
		if os.path.exists(batch_id_file):
			with open(batch_id_file, encoding="utf-8") as f:
				batch_id = f.read().strip()
			print(f"Resuming ChatGPT batch : {batch_id}")
		else:
			# 2. Write one JSONL line per request, each with its request id as custom_id
			jsonl_lines = []
			for request_id in requests:
				request = requests[request_id]
				batch_request = {
					"custom_id": request_id,
					"method": "POST",
					"url": "/v1/chat/completions",
					"body": {
						"model": self.llm_args.llm_model,
						"max_completion_tokens": self.llm_args.max_tokens,
						"messages": [
							{"role": "system", "content": request["system_prompt"]},
							{"role": "user", "content": request["user_prompt"]}
						]
					}
				}
				jsonl_lines.append(json.dumps(batch_request))
			jsonl_bytes = ("\n".join(jsonl_lines) + "\n").encode("utf-8")

			# 3. Upload the file and start the batch
			input_file = client.files.create(file=("batch_requests.jsonl", jsonl_bytes), purpose="batch")
			batch = client.batches.create(input_file_id=input_file.id, endpoint="/v1/chat/completions", completion_window="24h")
			batch_id = batch.id

			# 4. Save the batch id at once, so a run that stops while waiting can resume it
			os.makedirs(os.path.dirname(batch_id_file), exist_ok=True)
			with open(batch_id_file, "w", encoding="utf-8") as f:
				f.write(batch_id)
			print(f"ChatGPT batch sent : {batch_id}, {len(jsonl_lines)} requests")

		# 5. Check the status until the batch has ended
		start_time = time.time()
		while True:
			batch = client.batches.retrieve(batch_id)
			if batch.status in ("completed", "failed", "expired", "cancelled"):
				break
			if time.time() - start_time > batch_timeout_seconds:
				raise TimeoutError(f"ChatGPT batch {batch_id} did not end in {batch_timeout_seconds}s. Run again to resume it.")
			print(f"ChatGPT batch {batch_id} : {batch.status}, {batch.request_counts.completed} done")
			time.sleep(batch_poll_seconds)

		# 6. Collect the reply text and tokens of every succeeded request. An expired batch can still have some replies
		replies = {}
		failed_count = 0
		if batch.output_file_id is not None:
			output_text = client.files.content(batch.output_file_id).text
			for output_line in output_text.split("\n"):
				if output_line.strip() == "":
					continue
				output = json.loads(output_line)
				if output["response"] is None or output["response"]["status_code"] != 200:
					failed_count = failed_count + 1
					continue
				body = output["response"]["body"]
				# prompt_tokens includes the tokens OpenAI read from its prompt cache, which it caches on its own
				input_tokens = body["usage"]["prompt_tokens"]
				output_tokens = body["usage"]["completion_tokens"]
				cache_read_tokens = 0
				prompt_tokens_details = body["usage"].get("prompt_tokens_details")
				if prompt_tokens_details is not None and prompt_tokens_details.get("cached_tokens") is not None:
					cache_read_tokens = prompt_tokens_details["cached_tokens"]
				replies[output["custom_id"]] = LLMResponse(
					llm=self.llm_args.llm,
					llm_model=self.llm_args.llm_model,
					request_id=output["custom_id"],
					text=body["choices"][0]["message"]["content"],
					input_tokens=input_tokens,
					output_tokens=output_tokens
				)
				self.token_counter.number_of_calls = self.token_counter.number_of_calls + 1
				self.token_counter.input_tokens = self.token_counter.input_tokens + input_tokens
				self.token_counter.cache_read_tokens = self.token_counter.cache_read_tokens + cache_read_tokens
				self.token_counter.output_tokens = self.token_counter.output_tokens + output_tokens

		print(f"ChatGPT batch {batch_id} : {batch.status}, {len(replies)} succeeded, {failed_count} failed")
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
