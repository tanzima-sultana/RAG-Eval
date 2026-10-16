import json
import os
import time
import urllib.error
import urllib.request

import anthropic
from langchain_anthropic import ChatAnthropic

from llm.schema import LLMArgs
from llm.schema import LLMCost
from llm.schema import LLMResponse

# 1. validate_claude: checks a Claude model can be used. Called by llm.validate.validate_llm
# 2. ClaudeLLM:       one Claude model; chat_model, json_chat_model, generate(), generate_batch() and compute_cost().
#                     Returned by llm.get_llm()

def validate_claude(llm_args: LLMArgs) -> bool:
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
	if llm_args.input_price == 0 or llm_args.output_price == 0 or llm_args.cache_write_price == 0 or llm_args.cache_read_price == 0:
		raise ValueError(f"Claude model ({llm_args.llm_model}) needs an input, output, cache write and cache read price above 0.")

	return True

class ClaudeLLM:
	"""
	Input: llm_args, the settings of one Claude model, and token_counter, when created.
	Output: an object that sends prompts to that Claude model and returns its text.
	What it does: calls one Claude model and counts its tokens, so the cost can be computed.
	"""

	def __init__(self, llm_args: LLMArgs, token_counter):
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

	def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
		"""
		Input: system_prompt, the instructions ("" for none); user_prompt, the text sent to Claude.
		Output: an LLMResponse with the reply text and the tokens of this call. Its request_id is "".
		What it does: sends the system prompt and user prompt to the Claude model and returns the text of its reply.
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
		What it does: sends all requests as one Message Batch at the batch price, with the system prompt marked for
		Claude's prompt cache, waits for it to end and collects the replies by request id. If batch_id_file exists, it
		waits for that batch instead of sending a new one.
		"""
		client = anthropic.Anthropic(api_key=os.environ.get("CLAUDE_API_KEY", ""), timeout=self.llm_args.timeout_seconds)

		# 1. Resume the batch of an earlier run, so the same requests are never paid twice
		if os.path.exists(batch_id_file):
			with open(batch_id_file, encoding="utf-8") as f:
				batch_id = f.read().strip()
			print(f"Resuming Claude batch : {batch_id}")
		else:
			# 2. Send all requests as one batch, each with its request id as custom_id.
			# The system prompt is marked for Claude's prompt cache, so the requests after the first read it at the cache
			# read price. Claude does not cache a system prompt below its minimum size, and then bills it as normal input
			batch_requests = []
			for request_id in requests:
				request = requests[request_id]
				batch_requests.append({
					"custom_id": request_id,
					"params": {
						"model": self.llm_args.llm_model,
						"max_tokens": self.llm_args.max_tokens,
						"system": [
							{
								"type": "text",
								"text": request["system_prompt"],
								"cache_control": {"type": "ephemeral", "ttl": self.llm_args.cache_ttl}
							}
						],
						"messages": [{"role": "user", "content": request["user_prompt"]}]
					}
				})
			batch = client.messages.batches.create(requests=batch_requests)
			batch_id = batch.id

			# 3. Save the batch id at once, so a run that stops while waiting can resume it
			os.makedirs(os.path.dirname(batch_id_file), exist_ok=True)
			with open(batch_id_file, "w", encoding="utf-8") as f:
				f.write(batch_id)
			print(f"Claude batch sent : {batch_id}, {len(batch_requests)} requests")

		# 4. Check the status until the batch has ended
		start_time = time.time()
		while True:
			batch = client.messages.batches.retrieve(batch_id)
			if batch.processing_status == "ended":
				break
			if time.time() - start_time > batch_timeout_seconds:
				raise TimeoutError(f"Claude batch {batch_id} did not end in {batch_timeout_seconds}s. Run again to resume it.")
			print(f"Claude batch {batch_id} : {batch.processing_status}, {batch.request_counts.succeeded} done")
			time.sleep(batch_poll_seconds)

		# 5. Collect the reply text and tokens of every succeeded request
		replies = {}
		failed_count = 0
		for entry in client.messages.batches.results(batch_id):
			if entry.result.type != "succeeded":
				failed_count = failed_count + 1
				continue
			reply_text = ""
			for content_block in entry.result.message.content:
				if content_block.type == "text":
					reply_text = reply_text + content_block.text
			# Claude's input_tokens leaves out the cached tokens, so they are added to get every input token
			usage = entry.result.message.usage
			cache_creation_tokens = 0
			if usage.cache_creation_input_tokens is not None:
				cache_creation_tokens = usage.cache_creation_input_tokens
			cache_read_tokens = 0
			if usage.cache_read_input_tokens is not None:
				cache_read_tokens = usage.cache_read_input_tokens
			input_tokens = usage.input_tokens + cache_creation_tokens + cache_read_tokens
			output_tokens = usage.output_tokens
			replies[entry.custom_id] = LLMResponse(
				llm=self.llm_args.llm,
				llm_model=self.llm_args.llm_model,
				request_id=entry.custom_id,
				text=reply_text,
				input_tokens=input_tokens,
				output_tokens=output_tokens
			)
			self.token_counter.number_of_calls = self.token_counter.number_of_calls + 1
			self.token_counter.input_tokens = self.token_counter.input_tokens + input_tokens
			self.token_counter.cache_creation_tokens = self.token_counter.cache_creation_tokens + cache_creation_tokens
			self.token_counter.cache_read_tokens = self.token_counter.cache_read_tokens + cache_read_tokens
			self.token_counter.output_tokens = self.token_counter.output_tokens + output_tokens

		print(f"Claude batch {batch_id} : {len(replies)} succeeded, {failed_count} failed")
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
