import dataclasses

@dataclasses.dataclass
class LLMModel:
	"""
	Which LLM made an output. LLMCost and LLMResponse extend it, so every LLM output names its model.

	Fields:
		llm:       The LLM provider: claude, chatgpt, gemini or ollama.
		llm_model: The model the provider runs, e.g. claude-sonnet-5-5 or llama3.1:8b.
	"""
	llm: str
	llm_model: str

@dataclasses.dataclass
class LLMCost(LLMModel):
	"""
	The tokens and cost of all calls one LLM object made, returned by compute_cost().

	Fields:
		number_of_calls: Replies counted. Each request of a batch counts as one.
		input_tokens:    Input tokens of all calls.
		output_tokens:   Output tokens of all calls.
		input_cost:      US dollars for the input tokens.
		output_cost:     US dollars for the output tokens.
		total_cost:      input_cost plus output_cost.
	"""
	number_of_calls: int
	input_tokens: int
	output_tokens: int
	input_cost: float
	output_cost: float
	total_cost: float

@dataclasses.dataclass
class LLMResponse(LLMModel):
	"""
	One reply of the LLM, returned by generate() and, per request, by generate_batch().

	Fields:
		request_id:    The request id of a batch request. "" for a single generate() call.
		text:          The reply text.
		input_tokens:  Input tokens of this call. 0 when the provider does not report them.
		output_tokens: Output tokens of this call. 0 when the provider does not report them.
	"""
	request_id: str
	text: str
	input_tokens: int
	output_tokens: int
