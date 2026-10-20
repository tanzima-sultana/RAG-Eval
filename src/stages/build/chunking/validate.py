from common.constants import NO_FRAMEWORK, LANGCHAIN_FRAMEWORK
from common.constants import FIXED_CHUNKING, SENTENCE_CHUNKING, SEMANTIC_CHUNKING
from common.args import ChunkingArgs

def validate_chunking_args(chunking_args: ChunkingArgs):
	"""
	Input: chunking_args, a ChunkingArgs holding the chunking parameters.
	Output: nothing. Raises ValueError for the first check that fails.
	What it does: checks that framework is "" or "langchain", model_name is
	not empty, chunk_type is supported, min_chunk_size is smaller than
	max_chunk_size, fixed_chunk_overlap is smaller than max_chunk_size, and
	semantic chunking has a non-zero semantic_threshold.
	"""

	# 1. Check framework
	if chunking_args.framework not in (NO_FRAMEWORK, LANGCHAIN_FRAMEWORK):
		raise ValueError(
			f"framework ({chunking_args.framework}) is not supported. "
			f"Use \"{LANGCHAIN_FRAMEWORK}\" or \"{NO_FRAMEWORK}\" for the built-in chunkers."
		)

	# 2. Check model_name
	if chunking_args.model_name.strip() == "":
		raise ValueError(
			"model_name is empty. Provide a SentenceTransformer model name, e.g. all-MiniLM-L6-v2."
		)

	# 3. Check chunk_type
	if chunking_args.chunk_type not in (FIXED_CHUNKING, SENTENCE_CHUNKING, SEMANTIC_CHUNKING):
		raise ValueError(
			f"chunk_type ({chunking_args.chunk_type}) is not supported yet. Provide a valid chunk_type."
		)

	# 4. Check min and max chunk size
	if chunking_args.min_chunk_size >= chunking_args.max_chunk_size:
		raise ValueError(
			f"min_chunk_size ({chunking_args.min_chunk_size}) must be smaller than max_chunk_size ({chunking_args.max_chunk_size})"
		)

	# 5. Check fixed chunk overlap
	if chunking_args.fixed_chunk_overlap >= chunking_args.max_chunk_size:
		raise ValueError(
			f"fixed_chunk_overlap ({chunking_args.fixed_chunk_overlap}) must be smaller than "
			f"max_chunk_size ({chunking_args.max_chunk_size}); otherwise the sliding window "
			"never advances."
		)

	# 6. Check semantic threshold
	if chunking_args.chunk_type == SEMANTIC_CHUNKING and chunking_args.semantic_threshold == 0.0:
		raise ValueError(
			"provide valid semantic threshold for semantic chunking"
		)
