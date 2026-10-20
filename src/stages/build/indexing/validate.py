from common.constants import FAISS_FLATIP_INDEX, FAISS_IVF_INDEX, FAISS_HNSW_INDEX, BM25_INDEX, QDRANT_INDEX
from common.args import IndexArgs

def validate_index_args(index_args: IndexArgs):
	"""
	Input: index_args, an IndexArgs holding the indexing parameters.
	Output: nothing. Raises ValueError for the first check that fails.
	What it does: checks that index_type is supported, faiss_ivf has a
	positive ivf_nlist, faiss_hnsw has a positive hnsw_m and an
	hnsw_ef_construction at least hnsw_m, bm25_k1 is not negative, bm25_b
	is between 0 and 1, bm25_tokenizer is not empty, and qdrant_batch_size
	is positive.
	"""

	# 1. Check index_type
	if index_args.index_type not in (FAISS_FLATIP_INDEX, FAISS_IVF_INDEX, FAISS_HNSW_INDEX, BM25_INDEX, QDRANT_INDEX):
		raise ValueError(
			f"index_type ({index_args.index_type}) is not supported. "
			f"Use \"{FAISS_FLATIP_INDEX}\", \"{FAISS_IVF_INDEX}\", \"{FAISS_HNSW_INDEX}\", \"{BM25_INDEX}\" or \"{QDRANT_INDEX}\"."
		)

	# 2. Check ivf_nlist
	if index_args.index_type == FAISS_IVF_INDEX and index_args.ivf_nlist <= 0:
		raise ValueError(
			f"ivf_nlist ({index_args.ivf_nlist}) must be greater than 0 for {FAISS_IVF_INDEX}"
		)

	# 3. Check hnsw_m
	if index_args.index_type == FAISS_HNSW_INDEX and index_args.hnsw_m <= 0:
		raise ValueError(
			f"hnsw_m ({index_args.hnsw_m}) must be greater than 0 for {FAISS_HNSW_INDEX}"
		)

	# 4. Check hnsw_ef_construction
	if index_args.index_type == FAISS_HNSW_INDEX and index_args.hnsw_ef_construction < index_args.hnsw_m:
		raise ValueError(
			f"hnsw_ef_construction ({index_args.hnsw_ef_construction}) must be at least "
			f"hnsw_m ({index_args.hnsw_m}) for {FAISS_HNSW_INDEX}"
		)

	# 5. Check bm25_k1
	if index_args.bm25_k1 < 0.0:
		raise ValueError(
			f"bm25_k1 ({index_args.bm25_k1}) must not be negative"
		)

	# 6. Check bm25_b
	if index_args.bm25_b < 0.0 or index_args.bm25_b > 1.0:
		raise ValueError(
			f"bm25_b ({index_args.bm25_b}) must be between 0 and 1"
		)

	# 7. Check bm25_tokenizer
	if index_args.bm25_tokenizer.strip() == "":
		raise ValueError(
			"bm25_tokenizer is empty. Provide a tokenizer name, e.g. whitespace."
		)

	# 8. Check qdrant_batch_size
	if index_args.qdrant_batch_size <= 0:
		raise ValueError(
			f"qdrant_batch_size ({index_args.qdrant_batch_size}) must be greater than 0"
		)
