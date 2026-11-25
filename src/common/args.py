import argparse
import dataclasses

import yaml

from schema.llm import LLMArgs


@dataclasses.dataclass
class IngestionArgs:
	dataset_size: int
	dataset_name: str
	input_path: str
	output_path: str
	document_type: str


def make_ingestion_args(parsed_args: argparse.Namespace) -> IngestionArgs:
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_args().
	Output: an IngestionArgs holding the ingestion settings.
	What it does: copies each ingestion setting into the matching IngestionArgs field.
	"""
	ingestion_args = IngestionArgs(
		dataset_size=parsed_args.dataset_size,
		dataset_name=parsed_args.dataset_name,
		input_path=parsed_args.input_path,
		output_path=parsed_args.output_path,
		document_type=parsed_args.document_type
	)
	return ingestion_args


@dataclasses.dataclass
class EmbedderArgs:
	# The one embedding model chunking, embedding, indexing and the vector DB share
	# huggingface
	embedding_provider: str
	model_name: str
	batch_size: int
	normalize_embeddings: bool


def make_embedder_args(parsed_args: argparse.Namespace) -> EmbedderArgs:
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_args().
	Output: an EmbedderArgs holding the embedder settings.
	What it does: copies each embedder setting into the matching EmbedderArgs field. normalize_embeddings is "true" or
	"false" on the command line, and becomes a bool.
	"""
	embedder_args = EmbedderArgs(
		embedding_provider=parsed_args.embedding_provider,
		model_name=parsed_args.model_name,
		batch_size=parsed_args.batch_size,
		normalize_embeddings=parsed_args.normalize_embeddings == "true"
	)
	return embedder_args


@dataclasses.dataclass
class ChunkingArgs:
	chunk_type: str
	min_chunk_size: int
	max_chunk_size: int
	fixed_chunk_overlap: int


def make_chunking_args(parsed_args: argparse.Namespace, chunk_type: str, fixed_chunk_overlap: int) -> ChunkingArgs:
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_args(); chunk_type, one of chunk_types;
	fixed_chunk_overlap, one of the fixed_chunk_overlap list.
	Output: a ChunkingArgs holding the chunking settings for that chunk type and overlap.
	What it does: copies each chunking setting, chunk_type and fixed_chunk_overlap into the matching ChunkingArgs field.
	"""
	chunking_args = ChunkingArgs(
		chunk_type=chunk_type,
		min_chunk_size=parsed_args.min_chunk_size,
		max_chunk_size=parsed_args.max_chunk_size,
		fixed_chunk_overlap=fixed_chunk_overlap
	)
	return chunking_args


@dataclasses.dataclass
class EmbeddingArgs:
	embedder_args: EmbedderArgs
	chunking_args: ChunkingArgs


def make_embedding_args(embedder_args: EmbedderArgs, chunking_args: ChunkingArgs) -> EmbeddingArgs:
	"""
	Input: embedder_args, the shared embedder settings; chunking_args, the chunking settings of one chunk type.
	Output: an EmbeddingArgs holding them.
	What it does: holds embedder_args and chunking_args in one EmbeddingArgs.
	"""
	embedding_args = EmbeddingArgs(
		embedder_args=embedder_args,
		chunking_args=chunking_args
	)
	return embedding_args


@dataclasses.dataclass
class FaissIVFArgs:
	# nlist of the FAISS IVF index = ivf_nlist_multiplier * sqrt(number of chunks), capped at number of chunks // 39
	ivf_nlist_multiplier: float


def make_faiss_ivf_args(parsed_args: argparse.Namespace, ivf_nlist_multiplier: float) -> FaissIVFArgs:
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_args(); ivf_nlist_multiplier, one of the
	ivf_nlist_multiplier list.
	Output: a FaissIVFArgs for one FAISS IVF index.
	What it does: puts the multiplier into a FaissIVFArgs.
	"""
	faiss_ivf_args = FaissIVFArgs(ivf_nlist_multiplier=ivf_nlist_multiplier)
	return faiss_ivf_args


@dataclasses.dataclass
class FaissHNSWArgs:
	hnsw_m: int
	hnsw_ef_construction: int


def make_faiss_hnsw_args(parsed_args: argparse.Namespace) -> FaissHNSWArgs:
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_args().
	Output: a FaissHNSWArgs for the FAISS HNSW index.
	What it does: copies hnsw_m and hnsw_ef_construction into a FaissHNSWArgs. Raises ValueError if either is missing
	from the faiss_hnsw block of the config.
	"""
	if parsed_args.hnsw_m is None:
		raise ValueError("hnsw_m is missing. Give it in the faiss_hnsw block of the faiss_indexing config.")
	if parsed_args.hnsw_ef_construction is None:
		raise ValueError("hnsw_ef_construction is missing. Give it in the faiss_hnsw block of the faiss_indexing config.")

	faiss_hnsw_args = FaissHNSWArgs(
		hnsw_m=parsed_args.hnsw_m,
		hnsw_ef_construction=parsed_args.hnsw_ef_construction
	)
	return faiss_hnsw_args


def parse_faiss_indexing_args(parser: argparse.ArgumentParser) -> None:
	"""
	Input: parser, the argument parser of the build.
	Output: nothing.
	What it does: adds the FAISS indexing arguments to parser. faiss_types may be empty or missing. The settings of an
	index type are in its own block of the config and are needed only when that type is built, so they are optional.
	"""
	parser.add_argument(
		"--faiss_types",
		type=str,
		nargs="*",
		default=[],
		help="FAISS index types to build, one after another: faiss_flatip, faiss_ivf and/or faiss_hnsw. Empty builds none")
	parser.add_argument(
		"--faiss_ivf.ivf_nlist_multiplier",
		dest="ivf_nlist_multiplier",
		type=float,
		nargs="+",
		default=None,
		help="One or more multipliers; each builds a FAISS IVF index with nlist = multiplier * sqrt(chunks), capped at chunks // 39")
	parser.add_argument(
		"--faiss_hnsw.hnsw_m",
		dest="hnsw_m",
		type=int,
		default=None,
		help="Number of neighbours per node of the FAISS HNSW index")
	parser.add_argument(
		"--faiss_hnsw.hnsw_ef_construction",
		dest="hnsw_ef_construction",
		type=int,
		default=None,
		help="Candidate list size used while building the FAISS HNSW index")


@dataclasses.dataclass
class Bm25IndexingArgs:
	bm25_k1: float
	bm25_b: float
	bm25_tokenizer: str


def make_bm25_indexing_args(parsed_args: argparse.Namespace) -> Bm25IndexingArgs:
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_args().
	Output: a Bm25IndexingArgs holding the BM25 indexing settings.
	What it does: copies each BM25 setting into the matching Bm25IndexingArgs field.
	"""
	bm25_indexing_args = Bm25IndexingArgs(
		bm25_k1=parsed_args.bm25_k1,
		bm25_b=parsed_args.bm25_b,
		bm25_tokenizer=parsed_args.bm25_tokenizer
	)
	return bm25_indexing_args


def parse_bm25_indexing_args(parser: argparse.ArgumentParser) -> None:
	"""
	Input: parser, the argument parser of the build.
	Output: nothing.
	What it does: adds the BM25 indexing arguments to parser. BM25 is always built, so all three are required.
	"""
	parser.add_argument(
		"--bm25_k1",
		type=float,
		required=True,
		help="BM25 term frequency saturation")
	parser.add_argument(
		"--bm25_b",
		type=float,
		required=True,
		help="BM25 document length normalization")
	parser.add_argument(
		"--bm25_tokenizer",
		type=str,
		required=True,
		help="Tokenizer used by BM25: whitespace")


@dataclasses.dataclass
class QdrantArgs:
	qdrant_url: str
	qdrant_batch_size: int


def make_qdrant_args(parsed_args: argparse.Namespace) -> QdrantArgs:
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_args().
	Output: a QdrantArgs for the Qdrant collection.
	What it does: copies qdrant_url and qdrant_batch_size into a QdrantArgs. Raises ValueError if either is missing
	from the vectordb_indexing config.
	"""
	if parsed_args.qdrant_url is None:
		raise ValueError("qdrant_url is missing. Give it in the vectordb_indexing config.")
	if parsed_args.qdrant_batch_size is None:
		raise ValueError("qdrant_batch_size is missing. Give it in the vectordb_indexing config.")

	qdrant_args = QdrantArgs(
		qdrant_url=parsed_args.qdrant_url,
		qdrant_batch_size=parsed_args.qdrant_batch_size
	)
	return qdrant_args


def parse_vectordb_args(parser: argparse.ArgumentParser) -> None:
	"""
	Input: parser, the argument parser of the build.
	Output: nothing.
	What it does: adds the vector DB arguments to parser. vectordb_types may be empty or missing, so the Qdrant
	settings are optional.
	"""
	parser.add_argument(
		"--vectordb_types",
		type=str,
		nargs="*",
		default=[],
		help="Vector DB types to build, one after another: qdrant. Empty builds none")
	parser.add_argument(
		"--qdrant_url",
		type=str,
		default=None,
		help="URL of the Qdrant server, e.g. http://localhost:6333")
	parser.add_argument(
		"--qdrant_batch_size",
		type=int,
		default=None,
		help="Number of points uploaded to Qdrant in one request")


@dataclasses.dataclass
class OllamaArgs:
	ollama_api_base: str
	ollama_context_length: int


def make_ollama_args(parsed_args: argparse.Namespace) -> OllamaArgs:
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_args().
	Output: an OllamaArgs holding the Ollama settings.
	What it does: copies each Ollama setting into the matching OllamaArgs field.
	"""
	ollama_args = OllamaArgs(
		ollama_api_base=parsed_args.ollama_api_base,
		ollama_context_length=parsed_args.ollama_context_length
	)
	return ollama_args


@dataclasses.dataclass
class EvalDatasetArgs:
	# Where the built documents are: the root output folder, the dataset name and which saved document set
	output_path: str
	dataset_name: str
	dataset_size: int
	# The document list the build saved, doc_ids_{input hash}.json; its first dataset_size documents are used
	documents_path: str
	eval_dataset_llm_args: LLMArgs
	eval_dataset_kg_llm_args: LLMArgs
	eval_dataset_embedding_model: str
	n_sample_documents: int
	n_evals_requested: int
	# Seed for the document sample, so the same documents are picked every run
	sample_documents_seed: int
	# Only documents with more than this many tokens can be sampled for RAGAS
	min_document_tokens: int
	# Characters used to match each generated passage back to its document
	passage_match_chars: int


def make_eval_dataset_config_args(parsed_args: argparse.Namespace) -> EvalDatasetArgs:
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_eval_dataset_config().
	Output: an EvalDatasetArgs holding the eval dataset settings.
	What it does: builds each LLM's LLMArgs from its own block, and copies the other settings.
	"""
	eval_dataset_llm_args = LLMArgs(
		llm_model=parsed_args.eval_dataset_llm_model,
		max_tokens=parsed_args.eval_dataset_llm_max_tokens,
		timeout_seconds=parsed_args.eval_dataset_llm_timeout_seconds
	)
	eval_dataset_kg_llm_args = LLMArgs(
		llm_model=parsed_args.eval_dataset_kg_llm_model,
		max_tokens=parsed_args.eval_dataset_kg_llm_max_tokens,
		timeout_seconds=parsed_args.eval_dataset_kg_llm_timeout_seconds
	)
	eval_dataset_args = EvalDatasetArgs(
		output_path=parsed_args.documents_output_path,
		dataset_name=parsed_args.documents_dataset_name,
		dataset_size=parsed_args.documents_dataset_size,
		documents_path=parsed_args.documents_documents_path,
		eval_dataset_llm_args=eval_dataset_llm_args,
		eval_dataset_kg_llm_args=eval_dataset_kg_llm_args,
		eval_dataset_embedding_model=parsed_args.eval_dataset_embedding_model,
		n_sample_documents=parsed_args.eval_dataset_n_sample_documents,
		n_evals_requested=parsed_args.eval_dataset_n_evals_requested,
		sample_documents_seed=parsed_args.eval_dataset_sample_documents_seed,
		min_document_tokens=parsed_args.eval_dataset_min_document_tokens,
		passage_match_chars=parsed_args.eval_dataset_passage_match_chars
	)
	return eval_dataset_args


def parse_eval_dataset_config() -> argparse.Namespace:
	"""
	Input: none. Reads --config and any override flags from the command line.
	Output: an argparse.Namespace with one attribute per setting, named {section}_{key}.
	What it does: reads the eval dataset config file, turns each {section: {key: value}} into the flag
	--{section}.{key} and parses them. Flags given after --config override the file.
	"""
	# 1. Get the config file path. The other command line flags are overrides
	config_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
	config_parser.add_argument("--config", type=str, required=True, help="YAML config file, e.g. config/eval_dataset_config.yaml")
	config_args, override_arguments = config_parser.parse_known_args()

	with open(config_args.config, encoding="utf-8") as config_file:
		config = yaml.safe_load(config_file)

	# 2. Turn each setting of the file into its flag. The section is part of the flag, so two LLM blocks can share key names
	config_arguments = []
	for section in config:
		section_settings = config[section]
		if not isinstance(section_settings, dict):
			raise ValueError(f"{section} in {config_args.config} must be a section holding settings.")
		for key in section_settings:
			config_arguments.append(f"--{section}.{key}")
			config_arguments.append(str(section_settings[key]))

	# 3. Parse the file's flags and then the overrides, so an override replaces the file's value
	parser = argparse.ArgumentParser(description="Eval dataset parameters", allow_abbrev=False)
	parser.add_argument("--documents.output_path", dest="documents_output_path", type=str, required=True,
		help="Root output folder the build saved the documents in")
	parser.add_argument("--documents.dataset_name", dest="documents_dataset_name", type=str, required=True,
		help="Dataset name, e.g. pdfa-eng-wds")
	parser.add_argument("--documents.dataset_size", dest="documents_dataset_size", type=int, required=True,
		help="Which saved document set to generate from, e.g. 5000")
	parser.add_argument("--documents.documents_path", dest="documents_documents_path", type=str, required=True,
		help="The document list the build saved, doc_ids_{input hash}.json; its first dataset_size documents are used")
	parser.add_argument("--eval_dataset_llm.model", dest="eval_dataset_llm_model", type=str, required=True,
		help="LiteLLM model that writes the eval questions and answers, e.g. anthropic/claude-sonnet-5-5")
	parser.add_argument("--eval_dataset_llm.max_tokens", dest="eval_dataset_llm_max_tokens", type=int, required=True,
		help="Maximum output tokens per call of the eval dataset LLM")
	parser.add_argument("--eval_dataset_llm.timeout_seconds", dest="eval_dataset_llm_timeout_seconds", type=int, required=True,
		help="Seconds before one call of the eval dataset LLM times out")
	parser.add_argument("--eval_dataset_kg_llm.model", dest="eval_dataset_kg_llm_model", type=str, required=True,
		help="LiteLLM model that builds the RAGAS knowledge graph, e.g. anthropic/claude-haiku-5-5")
	parser.add_argument("--eval_dataset_kg_llm.max_tokens", dest="eval_dataset_kg_llm_max_tokens", type=int, required=True,
		help="Maximum output tokens per call of the knowledge graph LLM")
	parser.add_argument("--eval_dataset_kg_llm.timeout_seconds", dest="eval_dataset_kg_llm_timeout_seconds", type=int, required=True,
		help="Seconds before one call of the knowledge graph LLM times out")
	parser.add_argument("--eval_dataset.embedding_model", dest="eval_dataset_embedding_model", type=str, required=True,
		help="Local embedding model RAGAS uses to build its knowledge graph")
	parser.add_argument("--eval_dataset.n_sample_documents", dest="eval_dataset_n_sample_documents", type=int, required=True,
		help="Documents randomly picked for RAGAS to generate the eval dataset from")
	parser.add_argument("--eval_dataset.n_evals_requested", dest="eval_dataset_n_evals_requested", type=int, required=True,
		help="Eval samples to generate")
	parser.add_argument("--eval_dataset.sample_documents_seed", dest="eval_dataset_sample_documents_seed", type=int, required=True,
		help="Seed for the document sample, so the same documents are picked every run")
	parser.add_argument("--eval_dataset.min_document_tokens", dest="eval_dataset_min_document_tokens", type=int, required=True,
		help="Only documents with more than this many tokens can be sampled for RAGAS")
	parser.add_argument("--eval_dataset.passage_match_chars", dest="eval_dataset_passage_match_chars", type=int, required=True,
		help="Characters used to match each generated passage back to its document")
	parser.add_argument("--ollama.api_base", dest="ollama_api_base", type=str, required=True,
		help="URL of the local Ollama server, used only for an Ollama model")
	parser.add_argument("--ollama.context_length", dest="ollama_context_length", type=int, required=True,
		help="Context window in tokens of an Ollama model, used only for an Ollama model")

	parsed_args = parser.parse_args(config_arguments + override_arguments)
	return parsed_args


@dataclasses.dataclass
class QueryLLMArgs:
	# A LiteLLM model string, "{provider}/{model}", e.g. ollama_chat/llama3.1:8b
	llm_model: str
	# Maximum output tokens per call
	max_tokens: int
	# Seconds before one call times out
	timeout_seconds: int
	# The Ollama server and context length, only when the query LLM is an Ollama model, None otherwise
	ollama_api_base: str | None
	ollama_context_length: int | None


def make_query_llm_args(parsed_args: argparse.Namespace) -> QueryLLMArgs:
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_benchmark_config().
	Output: a QueryLLMArgs holding the query LLM settings.
	What it does: copies the settings of the query_llm block, and copies the Ollama settings only when the query LLM is
	an Ollama model.
	"""
	ollama_providers = ["ollama", "ollama_chat"]
	query_llm_provider = parsed_args.query_llm_model.split("/", 1)[0]
	ollama_api_base = None
	ollama_context_length = None
	if query_llm_provider in ollama_providers:
		ollama_api_base = parsed_args.query_llm_ollama_api_base
		ollama_context_length = parsed_args.query_llm_ollama_context_length

	query_llm_args = QueryLLMArgs(
		llm_model=parsed_args.query_llm_model,
		max_tokens=parsed_args.query_llm_max_tokens,
		timeout_seconds=parsed_args.query_llm_timeout_seconds,
		ollama_api_base=ollama_api_base,
		ollama_context_length=ollama_context_length
	)
	return query_llm_args


@dataclasses.dataclass
class QueryArgs:
	# Everything the query layer needs for one query type, and what is saved for the API
	query_llm_args: QueryLLMArgs
	# The one query type this QueryArgs makes: original, rewritten, multi or hyde
	query_type: str
	# Phrasings the query LLM writes for each question, used only by multi
	n_multi_qus: int


def make_query_args(parsed_args: argparse.Namespace, query_llm_args: QueryLLMArgs, query_type: str) -> QueryArgs:
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_benchmark_config(); query_llm_args, the query LLM
	settings; query_type, the one query type this QueryArgs makes.
	Output: a QueryArgs holding the query settings of that query type.
	What it does: copies n_multi_qus from the parsed args and keeps the query LLM settings and the query type.
	"""
	query_args = QueryArgs(
		query_llm_args=query_llm_args,
		query_type=query_type,
		n_multi_qus=parsed_args.n_multi_qus
	)
	return query_args


@dataclasses.dataclass
class QueryBenchmarkArgs:
	# The query settings only the benchmark uses. The API does not need them
	# The query types to make for every eval question: original, rewritten, multi and/or hyde
	query_types: list[str]
	# True to split the queries into val and test, False to use all of them as both
	query_split: bool
	val_queries_size: int
	test_queries_size: int


def make_query_benchmark_args(parsed_args: argparse.Namespace) -> QueryBenchmarkArgs:
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_benchmark_config().
	Output: a QueryBenchmarkArgs holding the benchmark's query settings.
	What it does: copies each setting of the query block that only the benchmark uses.
	"""
	query_benchmark_args = QueryBenchmarkArgs(
		query_types=parsed_args.query_types,
		query_split=parsed_args.query_split == "true",
		val_queries_size=parsed_args.val_queries_size,
		test_queries_size=parsed_args.test_queries_size
	)
	return query_benchmark_args


def make_manifest_chunking_args(chunk_entry, embedding_model: str) -> ChunkingArgs:
	"""
	Input: chunk_entry, one ChunkEntry of the build manifest; embedding_model, the manifest's embedding model.
	Output: a ChunkingArgs holding the settings the build made these chunks with.
	What it does: copies the chunk entry's settings and the embedding model into the matching ChunkingArgs field.
	"""
	chunking_args = ChunkingArgs(
		model_name=embedding_model,
		chunk_type=chunk_entry.chunk_type,
		min_chunk_size=chunk_entry.min_chunk_size,
		max_chunk_size=chunk_entry.max_chunk_size,
		fixed_chunk_overlap=chunk_entry.fixed_chunk_overlap,
		semantic_threshold=chunk_entry.semantic_threshold
	)
	return chunking_args


def parse_benchmark_config() -> argparse.Namespace:
	"""
	Input: none. Reads --config and any override flags from the command line.
	Output: an argparse.Namespace with one attribute per setting.
	What it does: reads the benchmark config file, turns each {section: {key: value}} into the flag
	--{section}.{key} and parses them. Flags given after --config override the file.
	"""
	# 1. Get the config file path. The other command line flags are overrides
	config_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
	config_parser.add_argument("--config", type=str, required=True, help="YAML config file, e.g. config/benchmark_config.yaml")
	config_args, override_arguments = config_parser.parse_known_args()

	with open(config_args.config, encoding="utf-8") as config_file:
		config = yaml.safe_load(config_file)

	# 2. Turn each setting of the file into its flag. A list is passed as one flag followed by its values
	config_arguments = []
	for section in config:
		section_settings = config[section]
		if not isinstance(section_settings, dict):
			raise ValueError(f"{section} in {config_args.config} must be a section holding settings.")
		for key in section_settings:
			config_arguments.append(f"--{section}.{key}")
			setting_value = section_settings[key]
			if isinstance(setting_value, list):
				for item in setting_value:
					config_arguments.append(str(item))
			else:
				config_arguments.append(str(setting_value))

	# 3. Parse the file's flags and then the overrides, so an override replaces the file's value
	parser = argparse.ArgumentParser(description="Benchmark parameters", allow_abbrev=False)
	parser.add_argument("--inputs.manifest_path", dest="manifest_path", type=str, required=True,
		help="The build manifest; every chunk type and index it lists is benchmarked")
	parser.add_argument("--inputs.eval_dataset_metadata_path", dest="eval_dataset_metadata_path", type=str, required=True,
		help="The eval dataset metadata; the eval set it names is used")
	parser.add_argument("--inputs.output_path", dest="output_path", type=str, required=True,
		help="Root output folder")
	parser.add_argument("--query.n_multi_qus", dest="n_multi_qus", type=int, required=True,
		help="Phrasings the query LLM writes for each eval question")
	parser.add_argument("--query.query_types", dest="query_types", type=str, nargs="*", required=True,
		help="Query types to make for every eval question: original, rewritten, multi and/or hyde. Must always hold original")
	# The config's true is read as the string "True", so the value is lowercased first
	parser.add_argument("--query.query_split", dest="query_split", type=str.lower, required=True, choices=["true", "false"],
		help="true to split the queries into val and test, false to use all of them as both")
	parser.add_argument("--query.val_queries_size", dest="val_queries_size", type=int, required=True,
		help="Used only with query_split true: queries for retrieval and retrieval evaluation")
	parser.add_argument("--query.test_queries_size", dest="test_queries_size", type=int, required=True,
		help="Used only with query_split true: queries for answer generation and answer evaluation")
	parser.add_argument("--query_llm.model", dest="query_llm_model", type=str, required=True,
		help="LiteLLM model that transforms the eval questions, e.g. ollama_chat/llama3.1:8b")
	parser.add_argument("--query_llm.max_tokens", dest="query_llm_max_tokens", type=int, required=True,
		help="Maximum output tokens per call of the query LLM")
	parser.add_argument("--query_llm.timeout_seconds", dest="query_llm_timeout_seconds", type=int, required=True,
		help="Seconds before one call of the query LLM times out")
	parser.add_argument("--query_llm.ollama_api_base", dest="query_llm_ollama_api_base", type=str, required=True,
		help="URL of the local Ollama server, used only for an Ollama model")
	parser.add_argument("--query_llm.ollama_context_length", dest="query_llm_ollama_context_length", type=int, required=True,
		help="Context window in tokens of an Ollama model, used only for an Ollama model")
	parsed_args = parser.parse_args(config_arguments + override_arguments)
	return parsed_args



def parse_embedder_args(parser: argparse.ArgumentParser) -> None:
	"""
	Input: parser, the argument parser of the build.
	Output: nothing.
	What it does: adds the embedder arguments to parser.
	"""
	parser.add_argument(
		"--embedding_provider",
		type=str,
		required=True,
		help="Embedding provider: huggingface")
	parser.add_argument(
		"--model_name",
		type=str,
		required=True,
		help="Embedding model shared by chunking, embedding, indexing and the vector DB")
	parser.add_argument(
		"--batch_size",
		type=int,
		required=True,
		help="Number of texts embedded per batch")
	parser.add_argument(
		"--normalize_embeddings",
		type=str.lower,
		required=True,
		choices=["true", "false"],
		help="true to normalize each vector to length 1, false not to")


def parse_chunking_args(parser: argparse.ArgumentParser) -> None:
	"""
	Input: parser, the argument parser of main.py.
	Output: nothing.
	What it does: adds the chunking arguments to parser.
	"""
	parser.add_argument(
		"--chunk_types",
		type=str,
		nargs="+",
		required=True,
		help="Chunk types to build, one after another: fixed, sentence and/or semantic")
	parser.add_argument(
		"--min_chunk_size",
		type=int,
		required=True,
		help="Minimum chunk size")
	parser.add_argument(
		"--max_chunk_size",
		type=int,
		required=True,
		help="Maximum chunk size")
	parser.add_argument(
		"--fixed_chunk_overlap",
		type=int,
		nargs="+",
		required=True,
		help="Overlaps between consecutive token windows, one or more, each built for every chunk type")


def parse_ingestion_args(parser: argparse.ArgumentParser) -> None:
	"""
	Input: parser, the argument parser of main.py.
	Output: nothing.
	What it does: adds the ingestion arguments to parser.
	"""
	parser.add_argument(
		"--dataset_size",
		type=int,
		required=True,
		help="Number of documents to ingest")
	parser.add_argument(
		"--dataset_name",
		type=str,
		required=True,
		help="Name of the dataset, used as its output folder name")
	parser.add_argument(
		"--input_path",
		type=str,
		required=True,
		help="Local file or folder to ingest")
	parser.add_argument(
		"--output_path",
		type=str,
		required=True,
		help="Root output folder")
	parser.add_argument(
		"--document_type",
		type=str,
		required=True,
		help="Type of documents to keep, e.g. pdf")


def parse_args() -> argparse.Namespace:
	"""
	Input: none. Reads --config and any override flags from the command line.
	Output: an argparse.Namespace with one attribute per build setting.
	What it does: reads the build config file, turns each setting into its flag and parses the build sections:
	ingestion, embedder, chunking, faiss_indexing, bm25_indexing and vectordb_indexing. Flags given after --config override the file.
	"""
	# 1. Get the config file path. The other command line flags are overrides
	config_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
	config_parser.add_argument("--config", type=str, required=True, help="YAML config file, e.g. config/build_config.yaml")
	config_args, override_arguments = config_parser.parse_known_args()

	with open(config_args.config, encoding="utf-8") as config_file:
		config = yaml.safe_load(config_file)

	# 2. Turn each setting of the file into its flag
	config_arguments = []
	for key in config:
		value = config[key]
		if isinstance(value, dict):
			for setting_key in value:
				setting_value = value[setting_key]
				# A nested block, e.g. faiss_hnsw: {hnsw_m: 32}, gives one flag per setting: --faiss_hnsw.hnsw_m 32. A list
				# in it is passed as one flag followed by its values, e.g. --faiss_ivf.ivf_nlist_multiplier 4 8 16
				if isinstance(setting_value, dict):
					for block_setting_key in setting_value:
						block_setting_value = setting_value[block_setting_key]
						config_arguments.append(f"--{setting_key}.{block_setting_key}")
						if isinstance(block_setting_value, list):
							for item in block_setting_value:
								config_arguments.append(str(item))
						else:
							config_arguments.append(str(block_setting_value))
					continue
				config_arguments.append("--" + setting_key)
				# A list is passed as one flag followed by its values, e.g. --chunk_types fixed sentence
				if isinstance(setting_value, list):
					for item in setting_value:
						config_arguments.append(str(item))
				else:
					config_arguments.append(str(setting_value))
		else:
			config_arguments.append("--" + key)
			config_arguments.append(str(value))

	# 3. Parse the file's flags and then the overrides, so an override replaces the file's value
	parser = argparse.ArgumentParser(description="Build parameters", allow_abbrev=False)
	parse_ingestion_args(parser)
	parse_embedder_args(parser)
	parse_chunking_args(parser)
	parse_faiss_indexing_args(parser)
	parse_bm25_indexing_args(parser)
	parse_vectordb_args(parser)

	parsed_args = parser.parse_args(config_arguments + override_arguments)
	return parsed_args
