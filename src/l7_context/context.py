from stages.build.chunking.schema import Chunk
from stages.evaluation.retrieval.schema import Retrieved
from l7_context.schema import Context

def get_context_list(retrieved_outputs: list[Retrieved], chunks: dict[str, Chunk]) -> list[Context]:
	"""
	Input: retrieved_outputs, the filtered Retrieved of one chunk type; chunks, the chunks of that chunk type.
	Output: one Context per Retrieved, in the same order. A Retrieved with no chunks gets an empty Context.
	What it does: joins the text of each Retrieved's chunks, best first, into its context.
	"""
	context_list = []

	for retrieved in retrieved_outputs:
		# A Retrieved with no chunks keeps its place with an empty Context
		if len(retrieved.chunk_ids) == 0:
			context_list.append(Context(id=retrieved.id, context="", token_count=0))
			continue

		# 1. Get the text of each chunk, best first, and add up the chunk sizes as the token count
		chunk_texts = []
		token_count = 0
		for chunk_id in retrieved.chunk_ids:
			chunk = chunks[chunk_id]
			chunk_texts.append(chunk.text)
			token_count = token_count + chunk.size

		# 2. Join the texts into the context, one blank line between chunks
		context = "\n\n".join(chunk_texts)

		# 3. Keep the Context in the same place as its Retrieved
		context_list.append(Context(id=retrieved.id, context=context, token_count=token_count))

	return context_list
