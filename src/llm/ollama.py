import json
import urllib.request

from common.constants import OLLAMA_URL
from common.constants import OLLAMA_TIMEOUT_SECONDS

class OllamaLLM:
	"""
	Input: llm_model, an Ollama model name, e.g. llama3.1:8b, when created.
	Output: an object that sends a prompt to that local model and returns its text.
	What it does: calls one model on the local Ollama server. Local calls are free, so its cost is always 0.
	"""

	def __init__(self, llm_model: str):
		"""
		Input: llm_model, an Ollama model name that is already pulled on the server.
		Output: nothing.
		What it does: keeps the model name for the calls.
		"""
		self.llm_model = llm_model

	def generate(self, prompt: str) -> str:
		"""
		Input: prompt, the text sent to the model.
		Output: the model's reply text.
		What it does: sends the prompt to the Ollama /api/generate endpoint and returns the reply.
		"""
		# Temperature 0 so the same prompt gives the same reply on every run
		request_body = {
			"model": self.llm_model,
			"prompt": prompt,
			"stream": False,
			"options": {"temperature": 0}
		}
		request = urllib.request.Request(
			OLLAMA_URL + "/api/generate",
			data=json.dumps(request_body).encode("utf-8"),
			headers={"Content-Type": "application/json"},
			method="POST"
		)
		with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
			response_body = json.loads(response.read().decode("utf-8"))

		return response_body["response"]

	def compute_cost(self) -> float:
		"""
		Input: nothing.
		Output: 0.0, because local Ollama calls are free.
		What it does: prints the cost of this model's calls, which is always 0.
		"""
		total_cost = 0.0
		print(f"{self.llm_model} cost : ${total_cost:.4f}")
		return total_cost
