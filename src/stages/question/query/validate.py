from common.args import QueryArgs

def validate_query_args(query_args: QueryArgs):
	"""
	Input: query_args, the query settings.
	Output: nothing. Raises ValueError for the first check that fails.
	What it does: checks embedding_model is not empty and n_multi_qus is at least 1.
	The query LLM is checked by llm.validate.validate_llm.
	"""

	# 1. Check embedding_model
	if query_args.embedding_model.strip() == "":
		raise ValueError(
			"embedding_model is empty. Pass --model_name with the SentenceTransformer model the build used, e.g. all-MiniLM-L6-v2."
		)

	# 2. Check n_multi_qus
	if query_args.n_multi_qus < 1:
		raise ValueError(
			f"n_multi_qus ({query_args.n_multi_qus}) must be at least 1"
		)
