import dataclasses


@dataclasses.dataclass
class LLMArgs:
	# A LiteLLM model string, "{provider}/{model}", e.g. anthropic/claude-sonnet-5-5 or ollama_chat/llama3.1:8b
	llm_model: str
	# Maximum output tokens per call
	max_tokens: int
	# Seconds before one call times out
	timeout_seconds: int


@dataclasses.dataclass
class LLMResponse:
	llm_model: str
	text: str
	input_tokens: int
	output_tokens: int


@dataclasses.dataclass
class LLMCost:
	llm_model: str
	number_of_calls: int
	# Every input token, cached ones included
	input_tokens: int
	# Of input_tokens, the ones written to and read from the provider's prompt cache
	cache_creation_tokens: int
	cache_read_tokens: int
	output_tokens: int
	# US dollars, from LiteLLM's price map
	total_cost: float
