import json
import os
import urllib.error
import urllib.request

from common.constants import CLAUDE_LLM_GENERATOR
from common.constants import CLAUDE_PRICES_PER_MILLION_TOKENS
from common.constants import OLLAMA_LLM
from common.constants import OLLAMA_URL
from llm.claude import ClaudeLLM
from llm.ollama import OllamaLLM

def get_llm(llm: str, llm_model: str) -> ClaudeLLM | OllamaLLM:
	"""
	Input: llm, the LLM provider name ("claude" or "ollama"); llm_model, the model to run on that provider.
	Output: a ClaudeLLM or OllamaLLM for that model, with a generate(prompt) method.
	What it does: checks the provider is supported and the model can be used on it, then builds that provider's LLM.
	A Claude LLM is only built when the environment variable ALLOW_PAID_API is 1.
	Raises ValueError otherwise.
	"""
	if llm == CLAUDE_LLM_GENERATOR:
		# Claude calls cost money, so a run must opt in with ALLOW_PAID_API=1. The Claude Code paid API guard hook
		# asks the user before any command that sets it
		if os.environ.get("ALLOW_PAID_API") != "1":
			raise ValueError(
				f"Claude model ({llm_model}) calls a paid API. Run again with ALLOW_PAID_API=1 set, "
				"e.g. ALLOW_PAID_API=1 bash scripts/eval.sh"
			)
		# Only models with a price can be used, so the cost of every run can be computed
		if llm_model not in CLAUDE_PRICES_PER_MILLION_TOKENS:
			raise ValueError(f"Claude model ({llm_model}) has no price in CLAUDE_PRICES_PER_MILLION_TOKENS.")
		return ClaudeLLM(llm_model)

	if llm == OLLAMA_LLM:
		# The Ollama server must be running and the model must already be pulled
		try:
			with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=10) as response:
				tags_body = json.loads(response.read().decode("utf-8"))
		except urllib.error.URLError as error:
			raise ValueError(f"Ollama server is not reachable at {OLLAMA_URL}. Start it with: ollama serve") from error

		pulled_model_names = []
		for pulled_model in tags_body["models"]:
			pulled_model_names.append(pulled_model["name"])
		if llm_model not in pulled_model_names:
			raise ValueError(f"Ollama model ({llm_model}) is not pulled. Pull it with: ollama pull {llm_model}")
		return OllamaLLM(llm_model)

	raise ValueError(f"llm ({llm}) is not supported. Use \"{CLAUDE_LLM_GENERATOR}\" or \"{OLLAMA_LLM}\".")
