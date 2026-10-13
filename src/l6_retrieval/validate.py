from common.args import RetrievalArgs

def validate_retrieval_args(retrieval_args: RetrievalArgs):
	"""
	Input: retrieval_args, the retrieval settings.
	Output: nothing. Raises ValueError if a setting is invalid.
	"""

	# 1. Check top_k
	if retrieval_args.top_k < 1:
		raise ValueError(
			f"top_k ({retrieval_args.top_k}) must be at least 1"
		)

	# 2. Check rerank_top_k
	if retrieval_args.rerank_top_k < 1:
		raise ValueError(
			f"rerank_top_k ({retrieval_args.rerank_top_k}) must be at least 1"
		)

	# 3. Check rerank_top_k is not more than top_k
	if retrieval_args.rerank_top_k > retrieval_args.top_k:
		raise ValueError(
			f"rerank_top_k ({retrieval_args.rerank_top_k}) must not be more than top_k ({retrieval_args.top_k}), "
			"since the reranker only reorders the top_k chunks the retriever returns"
		)
