import dataclasses

@dataclasses.dataclass
class Prompt:
	"""
	The prompt the LLM gets for one Context: the system prompt and the user prompt, ready to send to l9.
	The prompt list is in the same order as the context list, so prompt_list[i] belongs to context_list[i]
	and retrieved_list[i], which holds its setup.

	Fields:
		id:            Same as the Retrieved, Context, Query and EvalQnA id, linking the prompt to its eval question.
		system_prompt: The instructions for the LLM, the same for every prompt.
		user_prompt:   The context and the original question.
		token_count:   The context token count plus an estimate for the system prompt and the question, in
		               embedding-model tokens. An estimate of the prompt size, not the generation LLM's own count.
	"""
	id: str
	system_prompt: str
	user_prompt: str
	token_count: int
