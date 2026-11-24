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
class SetupArtifacts:
	# What the pipeline loads for one best setup
	# Where the setup's index is: a folder for FAISS and BM25, the collection name for Qdrant. Hybrid has two, the
	# dense one first and then BM25
	index_paths: tuple[str, ...]
	# The embedding model the indexes were built with, which embeds the query
	embedding_model: str


@dataclasses.dataclass(frozen=True)
class SetupParams:
	# Every setting that decides one best setup's results, flat
	chunk_type: str
	index_type: str
	query_type: str
	retrieval_type: str
	reranking: bool
	top_k: int
	rerank_top_k: int
	cross_encoder_model: str
	# The query LLM settings that write the query texts. ollama_context_length is None when the query LLM is not an
	# Ollama model
	query_llm_model: str
	query_llm_max_tokens: int
	n_multi_qus: int
	ollama_context_length: int | None


@dataclasses.dataclass(frozen=True)
class SetupLineage:
	# Where one best setup came from: the build manifest and the hashes of the build, eval dataset and benchmark
	manifest_path: str
	build_hash: str
	eval_dataset_hash: str
	benchmark_hash: str


@dataclasses.dataclass(frozen=True)
class SetupRecord:
	# One best setup as the benchmark saves it, grouped by kind: what to load, how to run it and where it came from
	artifacts: SetupArtifacts
	params: SetupParams
	lineage: SetupLineage
