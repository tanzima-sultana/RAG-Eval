import dataclasses

@dataclasses.dataclass
class LLM:
	"""
	The LLM settings of one layer, from the command line.

	Fields:
		llm:             The LLM provider: claude, chatgpt, gemini or ollama.
		llm_model:       The model the provider runs, e.g. claude-sonnet-5-5 or llama3.1:8b.
		input_price:     US dollars per million input tokens. 0 for ollama.
		output_price:    US dollars per million output tokens. 0 for ollama.
		max_tokens:      Maximum output tokens per call.
		timeout_seconds: Seconds before one call times out.
	"""
	llm: str
	llm_model: str
	input_price: float
	output_price: float
	max_tokens: int
	timeout_seconds: int
