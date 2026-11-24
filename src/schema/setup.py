import dataclasses


@dataclasses.dataclass(frozen=True)
class Setup:
	# One retrieval setup the benchmark scores and the pipeline runs. Frozen, so the evals can be grouped by it
	# fixed, sentence or semantic
	chunk_type: str
	# faiss_flatip, faiss_ivf, faiss_hnsw or qdrant for dense, bm25 for bm25, and both joined with "+" for hybrid,
	# e.g. faiss_flatip+bm25
	index_type: str
	# original, rewritten, multi or hyde. BM25 and hybrid have no hyde
	query_type: str
	# dense, bm25 or hybrid
	retrieval_type: str
	# True to rerank the top_k chunks with the cross-encoder and keep rerank_top_k
	reranking: bool


@dataclasses.dataclass(frozen=True)
class SetupRecord:
	# One best setup as the benchmark saves it: where its build outputs are and every setting that made its results,
	# so the pipeline rebuilds it exactly and only loads what the build saved
	setup: Setup
	# Where the setup's index is: a folder for FAISS and BM25, the collection name for Qdrant. Hybrid has two, the
	# dense one first and then BM25
	index_paths: tuple[str, ...]
	# The setup's chunk folder
	chunk_path: str
	# The build the setup was made from: its dataset, embedding model, input files and document list folder
	dataset_name: str
	embedding_model: str
	input_hash: str
	documents_path: str
	# The chunk settings of the setup's chunk type
	min_chunk_size: int
	max_chunk_size: int
	fixed_chunk_overlap: int
	semantic_threshold: float
	# The settings of each index, in the order of index_paths. Empty for faiss_flatip and qdrant
	index_settings: tuple[dict, ...]
	# The retrieval settings the setup was scored with
	top_k: int
	rerank_top_k: int
	# The query LLM settings that wrote the query texts the setup was scored with. ollama_context_length is None
	# when the query LLM is not an Ollama model
	query_llm_model: str
	query_llm_max_tokens: int
	n_multi_qus: int
	ollama_context_length: int | None
