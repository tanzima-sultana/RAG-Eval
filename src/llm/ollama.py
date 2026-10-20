import json
import urllib.error
import urllib.request

from langchain_ollama import ChatOllama

from llm.schema import LLMArgs
from llm.schema import LLMCost
from llm.schema import LLMResponse

# 1. validate_ollama: checks an Ollama model can be used. Called by llm.validate.validate_llm
# 2. OllamaLLM:       one local Ollama model; chat_model, json_chat_model, generate(), generate_batch() and compute_cost().
#                     Returned by llm.get_llm()

# Ollama server, started by the Ollama install
OLLAMA_URL = "http://localhost:11434"
# Context window in tokens: the prompt and the reply must fit in it together. Ollama's default of 4096 cuts long
# RAGAS prompts, so the instructions at their start are lost. 32768 keeps llama3.1:8b fully on a 12 GB GPU
OLLAMA_CONTEXT_LENGTH = 32768

def validate_ollama(llm_args: LLMArgs) -> bool:
	"""
	Input: llm_args, with llm ollama.
	Output: True. Raises ValueError for the first check that fails.
	What it does: checks the prices are 0, the Ollama server is reachable and the model is pulled.
	"""
	# 1. Ollama runs locally, so it costs nothing
	if llm_args.input_price != 0 or llm_args.output_price != 0 or llm_args.cache_write_price != 0 or llm_args.cache_read_price != 0:
		raise ValueError(f"Ollama model ({llm_args.llm_model}) runs locally, so its input, output and cache prices must be 0.")

	# 2. Check the Ollama server is running
	try:
		with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=10) as response:
			tags_body = json.loads(response.read().decode("utf-8"))
	except urllib.error.URLError as error:
		raise ValueError(f"Ollama server is not reachable at {OLLAMA_URL}. Start it with: ollama serve") from error

	# 3. Check the model is pulled
	pulled_model_names = []
	for pulled_model in tags_body["models"]:
		pulled_model_names.append(pulled_model["name"])
	if llm_args.llm_model not in pulled_model_names:
		raise ValueError(f"Ollama model ({llm_args.llm_model}) is not pulled. Pull it with: ollama pull {llm_args.llm_model}")

	return True

class OllamaLLM:
	"""
	Input: llm_args, the settings of one Ollama model, and token_counter, when created.
	Output: an object that sends prompts to that local model and returns its text.
	What it does: calls one model on the local Ollama server through Ollama's own API. The calls stay on
	this machine and are free, so the cost is 0.
	"""

	def __init__(self, llm_args: LLMArgs, token_counter):
		"""
		Input: llm_args, the Ollama model settings; token_counter, the llm.TokenCounter that counts its tokens.
		Output: nothing.
		What it does: builds two ChatOllama for the local Ollama server, with the token counter attached:
		chat_model for plain text replies and json_chat_model for replies that must be JSON.
		"""
		self.llm_args = llm_args
		self.token_counter = token_counter
		# Temperature 0 so the same prompt gives the same reply on every run. num_predict is Ollama's name for the
		# maximum output tokens
		self.chat_model = ChatOllama(
			model=llm_args.llm_model,
			base_url=OLLAMA_URL,
			temperature=0,
			num_predict=llm_args.max_tokens,
			num_ctx=OLLAMA_CONTEXT_LENGTH,
			client_kwargs={"timeout": llm_args.timeout_seconds},
			callbacks=[token_counter]
		)
		# The same model in Ollama's JSON mode, so every reply is valid JSON. Used by RAGAS in QnA, whose prompts
		# ask for JSON, which small local models often do not follow on their own
		self.json_chat_model = ChatOllama(
			model=llm_args.llm_model,
			base_url=OLLAMA_URL,
			temperature=0,
			num_predict=llm_args.max_tokens,
			num_ctx=OLLAMA_CONTEXT_LENGTH,
			format="json",
			client_kwargs={"timeout": llm_args.timeout_seconds},
			callbacks=[token_counter]
		)

	def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
		"""
		Input: system_prompt, the instructions ("" for none); user_prompt, the text sent to the model.
		Output: an LLMResponse with the reply text and the tokens of this call. Its request_id is "".
		What it does: sends the system prompt and user prompt to the local model and returns the text of its reply.
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
		Input: requests, {request_id: {"system_prompt", "user_prompt"}}; batch_id_file, batch_poll_seconds and
		batch_timeout_seconds, not used.
		Output: {request_id: LLMResponse} for every request.
		What it does: Ollama has no batch API and is free, so it sends the requests one at a time. It takes the same
		inputs as the paid providers' generate_batch, so generation calls every provider the same way.
		"""
		replies = {}
		for request_id in requests:
			request = requests[request_id]
			llm_response = self.generate(request["system_prompt"], request["user_prompt"])
			llm_response.request_id = request_id
			replies[request_id] = llm_response
			if len(replies) % 20 == 0:
				print(f"Ollama replies : {len(replies)} of {len(requests)}")

		return replies

	def compute_cost(self) -> LLMCost:
		"""
		Input: nothing.
		Output: an LLMCost with the calls, tokens (cached ones apart) and cost of all calls made so far. The cost is 0 with the ollama prices of 0.
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
