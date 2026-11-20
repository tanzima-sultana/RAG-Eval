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
class ChunkingArgs:
	model_name: str
	chunk_type: str
	min_chunk_size: int
	max_chunk_size: int
	fixed_chunk_overlap: int
	semantic_threshold: float


def make_chunking_args(parsed_args: argparse.Namespace, chunk_type: str) -> ChunkingArgs:
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_args(); chunk_type, one of chunk_types.
	Output: a ChunkingArgs holding the chunking settings for that chunk type.
	What it does: copies each chunking setting and chunk_type into the matching ChunkingArgs field.
	"""
	chunking_args = ChunkingArgs(
		model_name=parsed_args.model_name,
		chunk_type=chunk_type,
		min_chunk_size=parsed_args.min_chunk_size,
		max_chunk_size=parsed_args.max_chunk_size,
		fixed_chunk_overlap=parsed_args.fixed_chunk_overlap,
		semantic_threshold=parsed_args.semantic_threshold
	)
	return chunking_args


@dataclasses.dataclass
class EmbeddingArgs:
	chunking_args: ChunkingArgs


def make_embedding_args(chunking_args: ChunkingArgs) -> EmbeddingArgs:
	"""
	Input: chunking_args, the chunking settings of one chunk type.
	Output: an EmbeddingArgs holding them.
	What it does: wraps chunking_args into an EmbeddingArgs.
	"""
	embedding_args = EmbeddingArgs(
		chunking_args=chunking_args
	)
	return embedding_args


@dataclasses.dataclass
class IndexArgs:
	chunking_args: ChunkingArgs
	index_type: str
	ivf_nlist: int
	hnsw_m: int
	hnsw_ef_construction: int
	bm25_k1: float
	bm25_b: float
	bm25_tokenizer: str
	# The hash of the input files the index is built from, so each input set gets its own index
	input_hash: str


def make_index_args(parsed_args: argparse.Namespace, chunking_args: ChunkingArgs, index_type: str, input_hash: str) -> IndexArgs:
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_args(); chunking_args, the chunking settings of one
	chunk type; index_type, one of index_types; input_hash, the hash of the input files.
	Output: an IndexArgs holding the indexing settings for that index type.
	What it does: copies chunking_args, index_type, each indexing setting and input_hash into the matching IndexArgs field.
	"""
	index_args = IndexArgs(
		chunking_args=chunking_args,
		index_type=index_type,
		ivf_nlist=parsed_args.ivf_nlist,
		hnsw_m=parsed_args.hnsw_m,
		hnsw_ef_construction=parsed_args.hnsw_ef_construction,
		bm25_k1=parsed_args.bm25_k1,
		bm25_b=parsed_args.bm25_b,
		bm25_tokenizer=parsed_args.bm25_tokenizer,
		input_hash=input_hash
	)
	return index_args


@dataclasses.dataclass
class VectorDbArgs:
	chunking_args: ChunkingArgs
	vectordb_type: str
	qdrant_url: str
	qdrant_batch_size: int
	# The hash of the input files the collection is built from, so each input set gets its own collection
	input_hash: str


def make_vectordb_args(parsed_args: argparse.Namespace, chunking_args: ChunkingArgs, vectordb_type: str, input_hash: str) -> VectorDbArgs:
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_args(); chunking_args, the chunking settings of one
	chunk type; vectordb_type, one of vectordb_types; input_hash, the hash of the input files.
	Output: a VectorDbArgs holding the vector DB settings for that vector DB type.
	What it does: copies chunking_args, vectordb_type, each vector DB setting and input_hash into the matching VectorDbArgs field.
	"""
	vectordb_args = VectorDbArgs(
		chunking_args=chunking_args,
		vectordb_type=vectordb_type,
		qdrant_url=parsed_args.qdrant_url,
		qdrant_batch_size=parsed_args.qdrant_batch_size,
		input_hash=input_hash
	)
	return vectordb_args


def parse_vectordb_args(parser: argparse.ArgumentParser) -> None:
	"""
	Input: parser, the argument parser of main.py.
	Output: nothing.
	What it does: adds the vector DB arguments to parser.
	"""
	parser.add_argument(
		"--vectordb_types",
		type=str,
		nargs="+",
		required=True,
		help="Vector DB types to build, one after another: qdrant")
	parser.add_argument(
		"--qdrant_url",
		type=str,
		required=True,
		help="URL of the Qdrant server, e.g. http://localhost:6333")
	parser.add_argument(
		"--qdrant_batch_size",
		type=int,
		required=True,
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
	config_parser.add_argument("--config", type=str, required=True, help="YAML config file, e.g. eval_dataset_config.yaml")
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


def make_eval_dataset_args(parsed_args: argparse.Namespace) -> EvalDatasetArgs:
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_args() from the build config.
	Output: an EvalDatasetArgs holding the eval dataset settings, which the benchmark uses to find the saved eval dataset.
	What it does: builds the LLMArgs of the two eval dataset models with the shared LLM settings, and copies the other settings.
	"""
	eval_dataset_llm_args = LLMArgs(
		llm_model=parsed_args.eval_dataset_llm_model,
		max_tokens=parsed_args.llm_max_tokens,
		timeout_seconds=parsed_args.llm_timeout_seconds
	)
	eval_dataset_kg_llm_args = LLMArgs(
		llm_model=parsed_args.eval_dataset_kg_llm_model,
		max_tokens=parsed_args.llm_max_tokens,
		timeout_seconds=parsed_args.llm_timeout_seconds
	)
	eval_dataset_args = EvalDatasetArgs(
		output_path=parsed_args.output_path,
		dataset_name=parsed_args.dataset_name,
		dataset_size=parsed_args.dataset_size,
		eval_dataset_llm_args=eval_dataset_llm_args,
		eval_dataset_kg_llm_args=eval_dataset_kg_llm_args,
		eval_dataset_embedding_model=parsed_args.eval_dataset_embedding_model,
		n_sample_documents=parsed_args.n_sample_documents,
		n_evals_requested=parsed_args.n_evals_requested
	)
	return eval_dataset_args


def parse_eval_dataset_args(parser: argparse.ArgumentParser) -> None:
	"""
	Input: parser, the argument parser of main.py.
	Output: nothing.
	What it does: adds the eval dataset arguments to parser.
	"""
	parser.add_argument(
		"--eval_dataset_llm_model",
		type=str,
		required=True,
		help="LiteLLM model that writes the eval questions and answers, e.g. anthropic/claude-sonnet-5-5")
	parser.add_argument(
		"--eval_dataset_kg_llm_model",
		type=str,
		required=True,
		help="LiteLLM model that builds the RAGAS knowledge graph, e.g. anthropic/claude-haiku-5-5")
	parser.add_argument(
		"--eval_dataset_embedding_model",
		type=str,
		required=True,
		help="Local embedding model RAGAS uses to build its knowledge graph")
	parser.add_argument(
		"--n_sample_documents",
		type=int,
		required=True,
		help="Documents randomly picked for RAGAS to generate the eval dataset from")
	parser.add_argument(
		"--n_evals_requested",
		type=int,
		required=True,
		help="eval samples to generate")


@dataclasses.dataclass
class QueryArgs:
	query_llm_args: LLMArgs
	# The chunking model_name, so the query vectors match the chunk vectors
	embedding_model: str
	n_multi_qus: int
	query_types: list[str]
	# True to split the queries into val and test, False to use all of them as both
	query_split: bool
	val_queries_size: int
	test_queries_size: int


def make_query_args(parsed_args: argparse.Namespace) -> QueryArgs:
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_args().
	Output: a QueryArgs holding the query transformation settings.
	What it does: builds the query LLM's LLMArgs with the shared LLM settings, and takes the embedding model from
	the chunking model_name.
	"""
	query_llm_args = LLMArgs(
		llm_model=parsed_args.query_llm_model,
		max_tokens=parsed_args.llm_max_tokens,
		timeout_seconds=parsed_args.llm_timeout_seconds
	)
	query_args = QueryArgs(
		query_llm_args=query_llm_args,
		embedding_model=parsed_args.model_name,
		n_multi_qus=parsed_args.n_multi_qus,
		query_types=parsed_args.query_types,
		query_split=parsed_args.query_split == "true",
		val_queries_size=parsed_args.val_queries_size,
		test_queries_size=parsed_args.test_queries_size
	)
	return query_args


def parse_query_args(parser: argparse.ArgumentParser) -> None:
	"""
	Input: parser, the argument parser of main.py.
	Output: nothing.
	What it does: adds the query transformation arguments to parser.
	"""
	parser.add_argument(
		"--query_llm_model",
		type=str,
		required=True,
		help="LiteLLM model that transforms the eval questions, e.g. ollama_chat/llama3.1:8b")
	parser.add_argument(
		"--n_multi_qus",
		type=int,
		required=True,
		help="Phrasings the query LLM writes for each eval question")
	parser.add_argument(
		"--query_types",
		type=str,
		nargs="+",
		required=True,
		help="Query types to make and search with: original, rewritten, multi and/or hyde")
	# The config's true is read as the string "True", so the value is lowercased first
	parser.add_argument(
		"--query_split",
		type=str.lower,
		required=True,
		choices=["true", "false"],
		help="true to split the queries into val and test, false to use all of them as both")
	parser.add_argument(
		"--val_queries_size",
		type=int,
		required=True,
		help="Used only with query_split true: queries for retrieval and retrieval evaluation")
	parser.add_argument(
		"--test_queries_size",
		type=int,
		required=True,
		help="Used only with query_split true: queries for answer generation and answer evaluation")


@dataclasses.dataclass
class RetrievalArgs:
	# The chunk type's index settings, used to name the retrieved folders. Its index_type is empty: each retrieval
	# names its own index type
	index_args: IndexArgs
	retrieval_types: list[str]
	top_k: int
	reranking: bool
	rerank_top_k: int


def make_retrieval_args(parsed_args: argparse.Namespace, chunking_args: ChunkingArgs) -> RetrievalArgs:
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_args(); chunking_args, the chunking settings of one
	chunk type.
	Output: a RetrievalArgs holding the retrieval settings for that chunk type.
	What it does: builds the chunk type's index settings with an empty index type, and copies each retrieval setting.
	"""
	index_args = make_index_args(parsed_args, chunking_args, "")
	retrieval_args = RetrievalArgs(
		index_args=index_args,
		retrieval_types=parsed_args.retrieval_types,
		top_k=parsed_args.top_k,
		reranking=parsed_args.reranking == "true",
		rerank_top_k=parsed_args.rerank_top_k
	)
	return retrieval_args


def parse_retrieval_args(parser: argparse.ArgumentParser) -> None:
	"""
	Input: parser, the argument parser of main.py.
	Output: nothing.
	What it does: adds the retrieval arguments to parser.
	"""
	parser.add_argument(
		"--retrieval_types",
		type=str,
		nargs="+",
		required=True,
		help="Retrieval types to run: dense, bm25 and/or hybrid")
	parser.add_argument(
		"--top_k",
		type=int,
		required=True,
		help="Chunks each retriever returns for a query")
	# The config's false is read as the string "False", so the value is lowercased first
	parser.add_argument(
		"--reranking",
		type=str.lower,
		required=True,
		choices=["true", "false"],
		help="true to also rerank the top_k chunks with the cross-encoder, false not to")
	parser.add_argument(
		"--rerank_top_k",
		type=int,
		required=True,
		help="Chunks kept after the cross-encoder reranks the top_k chunks")
	parser.add_argument(
		"--n_top_setups",
		type=int,
		required=True,
		help="Best setups by retrieval eval, plain and reranked ranked together, that go on to the answer step")


def parse_llm_provider_args(parser: argparse.ArgumentParser) -> None:
	"""
	Input: parser, the argument parser of main.py.
	Output: nothing.
	What it does: adds the LLM provider arguments to parser: the settings every LLM call shares, and the Ollama ones.
	"""
	parser.add_argument(
		"--llm_max_tokens",
		type=int,
		required=True,
		help="Maximum output tokens per LLM call")
	parser.add_argument(
		"--llm_timeout_seconds",
		type=int,
		required=True,
		help="Seconds before one LLM call times out")
	parser.add_argument(
		"--ollama_api_base",
		type=str,
		required=True,
		help="URL of the local Ollama server, e.g. http://localhost:11434")
	parser.add_argument(
		"--ollama_context_length",
		type=int,
		required=True,
		help="Context window in tokens of an Ollama model; the prompt and the reply must fit in it together")


def parse_indexing_args(parser: argparse.ArgumentParser) -> None:
	"""
	Input: parser, the argument parser of main.py.
	Output: nothing.
	What it does: adds the indexing arguments to parser.
	"""
	parser.add_argument(
		"--index_types",
		type=str,
		nargs="+",
		required=True,
		help="Index types to build, one after another: faiss_flatip, faiss_ivf, faiss_hnsw and/or bm25")
	parser.add_argument(
		"--ivf_nlist",
		type=int,
		required=True,
		help="Number of clusters of the FAISS IVF index")
	parser.add_argument(
		"--hnsw_m",
		type=int,
		required=True,
		help="Number of neighbours per node of the FAISS HNSW index")
	parser.add_argument(
		"--hnsw_ef_construction",
		type=int,
		required=True,
		help="Candidate list size used while building the FAISS HNSW index")
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


def parse_chunking_args(parser: argparse.ArgumentParser) -> None:
	"""
	Input: parser, the argument parser of main.py.
	Output: nothing.
	What it does: adds the chunking arguments to parser.
	"""
	parser.add_argument(
		"--model_name",
		type=str,
		required=True,
		help="SentenceTransformer model used for token counts and semantic embeddings")
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
		required=True,
		help="Overlap between consecutive fixed-size chunks")
	parser.add_argument(
		"--semantic_threshold",
		type=float,
		required=True,
		help="Similarity threshold used by semantic chunking")


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
	ingestion, chunking, indexing and vectordb. Flags given after --config override the file.
	"""
	# 1. Get the config file path. The other command line flags are overrides
	config_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
	config_parser.add_argument("--config", type=str, required=True, help="YAML config file, e.g. build_config.yaml")
	config_args, override_arguments = config_parser.parse_known_args()

	with open(config_args.config, encoding="utf-8") as config_file:
		config = yaml.safe_load(config_file)

	# 2. Turn each setting of the file into its flag
	config_arguments = []
	for key in config:
		value = config[key]
		if isinstance(value, dict):
			for setting_key in value:
				config_arguments.append("--" + setting_key)
				setting_value = value[setting_key]
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
	parse_chunking_args(parser)
	parse_indexing_args(parser)
	parse_vectordb_args(parser)

	parsed_args = parser.parse_args(config_arguments + override_arguments)
	return parsed_args
