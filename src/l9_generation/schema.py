import dataclasses

@dataclasses.dataclass
class Generated:
	"""
	The answer the LLM gave for one Prompt.
	The generated list is in the same order as the prompt list, so generated_list[i] belongs to prompt_list[i],
	context_list[i] and retrieved_list[i], which holds its setup. A Prompt with an empty context gets I_DONT_KNOW
	without calling the LLM.

	Fields:
		id:        Same as the Retrieved, Context, Prompt, Query and EvalQnA id, linking the answer to its eval question.
		qus:       The original question the LLM answered.
		answer:    The answer the LLM gave, or I_DONT_KNOW.
		llm:       The LLM provider that wrote the answer, e.g. claude or ollama.
		llm_model: The model that wrote the answer.
	"""
	id: str
	qus: str
	answer: str
	llm: str
	llm_model: str
