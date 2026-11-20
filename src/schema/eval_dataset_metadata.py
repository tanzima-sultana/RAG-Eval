import dataclasses


@dataclasses.dataclass
class EvalDatasetMetadata:
	# What one saved eval dataset is and how it was made. Saved next to the eval dataset file

	# 1. Identity: which eval set this is
	# The 12-character hash of the eval dataset args, as in the file name
	eval_dataset_hash: str
	# The eval dataset file
	eval_dataset_path: str
	# The RAGAS knowledge graph file it was generated from
	knowledge_graph_path: str

	# 2. Integrity: the file is what it claims to be
	# SHA-256 of the eval dataset file
	content_hash: str
	# Eval samples actually saved, which can be fewer than requested
	n_evals_generated: int
	# When it was generated, UTC in ISO 8601
	created_at: str

	# 3. Source: which documents it came from
	dataset_name: str
	# Which saved document set was used
	dataset_size: int
	# The document list it read from, the same as documents_path in the eval dataset config
	documents_path: str
	# SHA-256 of the sorted document ids of that set
	documents_hash: str

	# 4. Generation settings: everything in the hash
	# The LLM that writes the questions and answers, and its maximum output tokens per call
	eval_dataset_llm_model: str
	eval_dataset_llm_max_tokens: int
	# The LLM that builds the RAGAS knowledge graph, and its maximum output tokens per call
	eval_dataset_kg_llm_model: str
	eval_dataset_kg_llm_max_tokens: int
	# The Ollama context length, only when one of the LLMs is an Ollama model, None otherwise
	ollama_context_length: int | None
	# The local embedding model RAGAS uses to build its knowledge graph
	embedding_model: str
	# Documents sampled for RAGAS
	n_sample_documents: int
	# Eval samples asked for
	n_evals_requested: int
	# Seed for the document sample
	sample_documents_seed: int
	# Only documents with more than this many tokens could be sampled
	min_document_tokens: int
	# Characters used to match each passage back to its document
	passage_match_chars: int

	# 5. Library version: generation is RAGAS code
	ragas_version: str
