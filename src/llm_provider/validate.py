import json
import os
import urllib.error
import urllib.request

import litellm

from common.args import OllamaArgs
from schema.llm import LLMArgs


def validate_llm(llm_args: LLMArgs, ollama_args: OllamaArgs) -> None:
	"""
	Input: llm_args, the LLM settings of one layer; ollama_args, the Ollama settings, used only for an Ollama model.
	Output: nothing.
	What it does: raises ValueError if the settings are not valid, or if the model cannot be used: a local Ollama model
	needs its server running and the model pulled; a paid model needs ALLOW_PAID_API=1, its API key and a LiteLLM price.
	"""
	llm_model = llm_args.llm_model

	# 1. Check the settings every provider shares
	if "/" not in llm_model:
		raise ValueError(
			f"llm_model ({llm_model}) must be a LiteLLM model string \"{{provider}}/{{model}}\", "
			"e.g. anthropic/claude-sonnet-5-5 or ollama_chat/llama3.1:8b."
		)

	if llm_args.max_tokens <= 0:
		raise ValueError(f"max_tokens ({llm_args.max_tokens}) must be greater than 0.")

	if llm_args.timeout_seconds <= 0:
		raise ValueError(f"timeout_seconds ({llm_args.timeout_seconds}) must be greater than 0.")

	provider, model_name = llm_model.split("/", 1)

	# 2. A local Ollama model: the server from the config must answer and the model must be pulled
	if provider == "ollama" or provider == "ollama_chat":
		if ollama_args.ollama_context_length <= 0:
			raise ValueError(f"ollama_context_length ({ollama_args.ollama_context_length}) must be greater than 0.")

		ollama_api_base = ollama_args.ollama_api_base
		try:
			with urllib.request.urlopen(ollama_api_base + "/api/tags", timeout=10) as response:
				tags_body = json.loads(response.read().decode("utf-8"))
		except urllib.error.URLError as error:
			raise ValueError(f"Ollama server is not reachable at {ollama_api_base}. Start it with: ollama serve") from error

		pulled_model_names = []
		for pulled_model in tags_body["models"]:
			pulled_model_names.append(pulled_model["name"])
		if model_name not in pulled_model_names:
			raise ValueError(f"Ollama model ({model_name}) is not pulled. Pull it with: ollama pull {model_name}")
		return

	# 3. Any other provider is paid: a run must opt in
	if os.environ.get("ALLOW_PAID_API") != "1":
		raise ValueError(f"{llm_model} calls a paid API. Run again with ALLOW_PAID_API=1 set.")

	# 4. LiteLLM must find the provider's API key in the environment
	environment = litellm.validate_environment(llm_model)
	if len(environment["missing_keys"]) > 0:
		raise ValueError(f"{llm_model} needs {environment['missing_keys']} set in the environment. Add it to .env.")

	# 5. The model needs a price, so every run's cost is known
	try:
		litellm.get_model_info(model=llm_model)
	except Exception as error:
		raise ValueError(f"{llm_model} is not in LiteLLM's price map, so its cost cannot be computed.") from error
