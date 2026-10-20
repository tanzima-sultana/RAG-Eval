from common.args import RetrievalArgs
from common.constants import BM25_RETRIEVAL
from common.constants import DENSE_RETRIEVAL
from common.constants import HYBRID_RETRIEVAL
from common.constants import HYDE_QUERY
from common.constants import MULTI_QUERY
from common.constants import ORIGINAL_QUERY
from common.constants import REWRITTEN_QUERY

def validate_retrieval_args(retrieval_args: RetrievalArgs):
	"""
	Input: retrieval_args, the retrieval settings.
	Output: nothing. Raises ValueError if a setting is invalid.
	"""

	# 1. Check retrieval_types
	for retrieval_type in retrieval_args.retrieval_types:
		if retrieval_type not in (DENSE_RETRIEVAL, BM25_RETRIEVAL, HYBRID_RETRIEVAL):
			raise ValueError(
				f"retrieval_type ({retrieval_type}) is not supported. "
				f"Use \"{DENSE_RETRIEVAL}\", \"{BM25_RETRIEVAL}\" or \"{HYBRID_RETRIEVAL}\"."
			)

	# 2. Check query_types
	for query_type in retrieval_args.query_types:
		if query_type not in (ORIGINAL_QUERY, REWRITTEN_QUERY, MULTI_QUERY, HYDE_QUERY):
			raise ValueError(
				f"query_type ({query_type}) is not supported. "
				f"Use \"{ORIGINAL_QUERY}\", \"{REWRITTEN_QUERY}\", \"{MULTI_QUERY}\" or \"{HYDE_QUERY}\"."
			)

	# 3. Check top_k
	if retrieval_args.top_k < 1:
		raise ValueError(
			f"top_k ({retrieval_args.top_k}) must be at least 1"
		)

	# 4. Check rerank_top_k
	if retrieval_args.rerank_top_k < 1:
		raise ValueError(
			f"rerank_top_k ({retrieval_args.rerank_top_k}) must be at least 1"
		)

	# 5. Check rerank_top_k is not more than top_k
	if retrieval_args.rerank_top_k > retrieval_args.top_k:
		raise ValueError(
			f"rerank_top_k ({retrieval_args.rerank_top_k}) must not be more than top_k ({retrieval_args.top_k}), "
			"since the reranker only reorders the top_k chunks the retriever returns"
		)
