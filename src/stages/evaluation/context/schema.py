import dataclasses

@dataclasses.dataclass
class Context:
	"""
	The context the LLM gets for one Retrieved: the text of its chunks, ready to go into the prompt.
	The context list is in the same order as the retrieved list, so context_list[i] belongs to retrieved_list[i],
	which holds its setup. A Retrieved with no chunks gets an empty Context ("" and 0).

	Fields:
		id:          Same as the Retrieved, Query and EvalQnA id, linking the context to its eval question.
		context:     The text sent to the LLM.
		token_count: The sum of the chunk sizes of the chunks used, in embedding-model tokens. An estimate of the
		             prompt size, not the generation LLM's own count.
	"""
	id: str
	context: str
	token_count: int
