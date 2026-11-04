"""
args.py

The settings of each layer as dataclasses, and the command line arguments of build.py and eval.py that fill them.
The arguments come from the YAML file config.yaml (--config): build.py reads its build part and eval.py
its build and eval parts, so each setting is written once. Any flag given after --config overrides its value.
"""

import argparse
import dataclasses

import yaml

from common import constants
from llm.schema import LLMArgs

@dataclasses.dataclass
class IngestArgs:
	dataset_size: int 
	input_path: str
	output_path: str 
	document_type: str
	framework: str

@dataclasses.dataclass
class ChunkingArgs:
	framework: str
	model_name: str
	chunk_type: str
	min_chunk_size: int
	max_chunk_size: int
	fixed_chunk_overlap: int
	semantic_threshold: float

@dataclasses.dataclass
class IndexArgs:
	# faiss_flatip, faiss_ivf, faiss_hnsw, bm25 or qdrant
	index_type: str
	ivf_nlist: int
	hnsw_m: int
	hnsw_ef_construction: int
	bm25_k1: float
	bm25_b: float
	bm25_tokenizer: str
	qdrant_batch_size: int

@dataclasses.dataclass
class EvalQnAArgs:
	eval_qna_model: str
	n_sample_documents: int
	n_sample_evals: int
	framework: str

@dataclasses.dataclass
class QueryArgs:
	embedding_model: str
	n_multi_qus: int

@dataclasses.dataclass
class RetrievalArgs:
	framework: str
	retrieval_types: list[str]
	query_types: list[str]
	top_k: int
	reranking: bool
	rerank_top_k: int

@dataclasses.dataclass
class GenerationArgs:
	llm_args: LLMArgs
	batch_poll_seconds: int
	batch_timeout_seconds: int
	framework: str

@dataclasses.dataclass
class AnswerEvalArgs:
	ans_eval_framework: str
	llm_args: LLMArgs
	embedding_model: str
	framework: str

@dataclasses.dataclass
class AnalysisArgs:
	n_compare_runs: int
	n_worst_answers: int
	framework: str

# ------ Command line arguments
def parse_config_arguments(parser: argparse.ArgumentParser, config_parts: list[str]) -> argparse.Namespace:
	"""
	Input: parser, the argument parser of build.py or eval.py with all its arguments added; config_parts, the top-level
	parts of the YAML file to read: ["build"] for build.py, ["build", "eval"] for eval.py.
	Output: an argparse.Namespace with one attribute per argument.
	What it does: reads those parts of the YAML file given by --config, turns every setting into its command line flag
	and parses them with parser, so the usual type and required checks apply. Flags given after --config override the
	file. Every argument of parser must be in the file, so the file is the full record of a run.
	"""
	# 1. Get the config file path. The other command line flags are overrides, kept for step 3
	config_parser = argparse.ArgumentParser(add_help=False)
	config_parser.add_argument("--config", type=str, required=True, help="YAML config file, e.g. config.yaml")
	config_args, override_arguments = config_parser.parse_known_args()

	with open(config_args.config, encoding="utf-8") as f:
		config = yaml.safe_load(f)

	# 2. Turn each setting of the parts into its flag. Each part holds sections, whose names are only for reading
	config_arguments = []
	config_keys = []
	for part_name in config_parts:
		if part_name not in config or not isinstance(config[part_name], dict):
			parser.error(f"{config_args.config}: part {part_name} is missing")
		part = config[part_name]

		for section_name in part:
			section = part[section_name]
			if not isinstance(section, dict):
				parser.error(f"{config_args.config}: section {part_name}.{section_name} must hold settings as key: value lines")

			for key in section:
				value = section[key]
				config_keys.append(key)
				config_arguments.append("--" + key)
				if value is None:
					parser.error(f"{config_args.config}: {key} has no value")
				# A list is passed as one flag followed by its values, e.g. --chunk_types fixed sentence
				if isinstance(value, list):
					for item in value:
						config_arguments.append(str(item))
				# YAML reads true and false as booleans, and the arguments take them as the strings "true" and "false"
				elif isinstance(value, bool):
					if value:
						config_arguments.append("true")
					else:
						config_arguments.append("false")
				else:
					config_arguments.append(str(value))

	# 3. Parse the file's flags and then the overrides, so an override replaces the file's value.
	# A misspelled key is not an argument of parser, so it stops with "unrecognized arguments". The parsers are made
	# with allow_abbrev=False, so a shortened or misspelled key such as chunk_type is never taken as chunk_types
	parsed_args = parser.parse_args(config_arguments + override_arguments)

	# 4. Every argument must come from the file, so no setting falls back to a default without being written down
	for argument_name in vars(parsed_args):
		if argument_name not in config_keys:
			parser.error(f"{config_args.config}: {argument_name} is missing")

	return parsed_args

def parse_build_args():
	"""
	Input: none. Reads --config and any override flags passed to build.py.
	Output: an argparse.Namespace with one attribute per argument.
	What it does: defines the ingestion, chunking and indexing arguments and parses them from the build part of the
	config file.
	"""
	parser = argparse.ArgumentParser(description="RAG-Eval build parameters", allow_abbrev=False)
	add_ingest_arguments(parser)
	add_chunking_arguments(parser)
	add_index_arguments(parser)
	return parse_config_arguments(parser, ["build"])

def parse_eval_args():
	"""
	Input: none. Reads --config and any override flags passed to eval.py.
	Output: an argparse.Namespace with one attribute per argument.
	What it does: defines the arguments of the build to evaluate, which must have the values the build was run
	with, and the LLM, eval QnA, query, retrieval, generation, answer evaluation and analysis arguments, and parses them from the build and
	eval parts of the config file, so eval always uses the settings the build was made with.
	"""
	parser = argparse.ArgumentParser(description="RAG-Eval evaluation parameters", allow_abbrev=False)
	add_ingest_arguments(parser)
	add_chunking_arguments(parser)
	add_index_arguments(parser)
	add_llm_arguments(parser)
	add_eval_qna_arguments(parser)
	add_query_arguments(parser)
	add_retrieval_arguments(parser)
	add_generation_arguments(parser)
	add_answer_eval_arguments(parser)
	add_analysis_arguments(parser)
	return parse_config_arguments(parser, ["build", "eval"])

def add_ingest_arguments(parser: argparse.ArgumentParser) -> None:
	"""
	Input: parser, the argument parser of build.py or eval.py.
	Output: nothing.
	What it does: adds the ingestion arguments, which build.py and eval.py both take.
	"""
	parser.add_argument(
		"--dataset_size",
		type=int,
		required=False,
		default=0,
		help="Number of documents to ingest")
	parser.add_argument(
		"--input_path",
		type=str,
		required=False,
		default="",
		help="Input dataset local file or folder path")
	parser.add_argument(
		"--output_path",
		type=str,
		required=False,
		default="",
		help="Directory where ingested documents are written")
	parser.add_argument(
		"--document_type",
		type=str,
		required=True,
		help="Type of documents to ingest, e.g. pdf")
	parser.add_argument(
		"--framework",
		type=str,
		required=False,
		default="",
		help="Framework for ingestion, chunking, embedding, indexing and retrieval: langchain, or empty for the built-in code")

def add_chunking_arguments(parser: argparse.ArgumentParser) -> None:
	"""
	Input: parser, the argument parser of build.py or eval.py.
	Output: nothing.
	What it does: adds the chunking and embedding arguments, which build.py and eval.py both take.
	"""
	parser.add_argument(
		"--model_name",
		type=str,
		required=False,
		default="",
		help="SentenceTransformer model used for token counts and semantic embeddings")
	parser.add_argument(
		"--chunk_types",
		type=str,
		nargs="+",
		required=False,
		default=[constants.FIXED_CHUNKING, constants.SENTENCE_CHUNKING, constants.SEMANTIC_CHUNKING],
		help="Chunking strategies to build, one after another: fixed, sentence and/or semantic")
	parser.add_argument(
		"--min_chunk_size",
		type=int,
		required=False,
		default=constants.MIN_CHUNK_SIZE,
		help="Minimum chunk size")
	parser.add_argument(
		"--max_chunk_size",
		type=int,
		required=False,
		default=constants.MAX_CHUNK_SIZE,
		help="Maximum chunk size")
	parser.add_argument(
		"--fixed_chunk_overlap",
		type=int,
		required=False,
		default=constants.FIXED_CHUNK_OVERLAP,
		help="Overlap between consecutive fixed-size chunks")
	parser.add_argument(
		"--semantic_threshold",
		type=float,
		required=False,
		default=constants.SEMANTIC_THRESHOLD,
		help="Similarity threshold used by semantic chunking")

def add_index_arguments(parser: argparse.ArgumentParser) -> None:
	"""
	Input: parser, the argument parser of build.py or eval.py.
	Output: nothing.
	What it does: adds the indexing arguments, which build.py and eval.py both take.
	"""
	parser.add_argument(
		"--index_types",
		type=str,
		nargs="*",
		required=False,
		default=[constants.FAISS_FLATIP_INDEX, constants.FAISS_IVF_INDEX, constants.FAISS_HNSW_INDEX, constants.BM25_INDEX, constants.QDRANT_INDEX],
		help="Indexes to build, one after another: any of faiss_flatip, faiss_ivf, faiss_hnsw, bm25 and qdrant, or none")
	parser.add_argument(
		"--ivf_nlist",
		type=int,
		required=False,
		default=constants.IVF_NLIST,
		help="Number of IVF clusters for faiss_ivf")
	parser.add_argument(
		"--hnsw_m",
		type=int,
		required=False,
		default=constants.HNSW_M,
		help="Number of neighbours per node in the HNSW graph")
	parser.add_argument(
		"--hnsw_ef_construction",
		type=int,
		required=False,
		default=constants.HNSW_EF_CONSTRUCTION,
		help="HNSW candidate list size used while building the graph")
	parser.add_argument(
		"--bm25_k1",
		type=float,
		required=False,
		default=constants.BM25_K1,
		help="BM25 term frequency saturation")
	parser.add_argument(
		"--bm25_b",
		type=float,
		required=False,
		default=constants.BM25_B,
		help="BM25 document length normalization")
	parser.add_argument(
		"--bm25_tokenizer",
		type=str,
		required=False,
		default=constants.BM25_TOKENIZER,
		help="Tokenizer used to split text into BM25 terms")
	parser.add_argument(
		"--qdrant_batch_size",
		type=int,
		required=False,
		default=constants.QDRANT_BATCH_SIZE,
		help="Number of vectors sent to Qdrant per upsert")

def add_llm_arguments(parser: argparse.ArgumentParser) -> None:
	"""
	Input: parser, the argument parser of eval.py.
	Output: nothing.
	What it does: adds the LLM call arguments, the same for every layer that uses an LLM.
	"""
	parser.add_argument(
		"--llm_max_tokens",
		type=int,
		required=True,
		help="Maximum output tokens per LLM call, the same for every layer")
	parser.add_argument(
		"--llm_timeout_seconds",
		type=int,
		required=True,
		help="Seconds before one LLM call times out, the same for every layer")
	parser.add_argument(
		"--llm_cache_ttl",
		type=str,
		required=True,
		help="How long Claude keeps a cached system prompt: 5m or 1h, the same for every layer")

def add_eval_qna_arguments(parser: argparse.ArgumentParser) -> None:
	"""
	Input: parser, the argument parser of eval.py.
	Output: nothing.
	What it does: adds the eval QnA arguments: the two QnA LLMs, the embedding model and the sample sizes.
	"""
	parser.add_argument(
		"--qna_llm",
		type=str,
		required=True,
		help="LLM provider RAGAS uses to write the eval questions and answers: claude, chatgpt, gemini or ollama")
	parser.add_argument(
		"--qna_llm_model",
		type=str,
		required=True,
		help="Model the QnA LLM provider runs")
	parser.add_argument(
		"--qna_input_price",
		type=float,
		required=True,
		help="QnA model price in US dollars per million input tokens, 0 for ollama")
	parser.add_argument(
		"--qna_output_price",
		type=float,
		required=True,
		help="QnA model price in US dollars per million output tokens, 0 for ollama")
	parser.add_argument(
		"--qna_cache_write_price",
		type=float,
		required=True,
		help="QnA model price in US dollars per million input tokens written to the prompt cache, 0 for ollama")
	parser.add_argument(
		"--qna_cache_read_price",
		type=float,
		required=True,
		help="QnA model price in US dollars per million input tokens read from the prompt cache, 0 for ollama")
	parser.add_argument(
		"--qna_kg_llm",
		type=str,
		required=True,
		help="LLM provider RAGAS uses to build its knowledge graph: claude, chatgpt, gemini or ollama")
	parser.add_argument(
		"--qna_kg_llm_model",
		type=str,
		required=True,
		help="Model the knowledge graph LLM provider runs")
	parser.add_argument(
		"--qna_kg_input_price",
		type=float,
		required=True,
		help="Knowledge graph model price in US dollars per million input tokens, 0 for ollama")
	parser.add_argument(
		"--qna_kg_output_price",
		type=float,
		required=True,
		help="Knowledge graph model price in US dollars per million output tokens, 0 for ollama")
	parser.add_argument(
		"--qna_kg_cache_write_price",
		type=float,
		required=True,
		help="Knowledge graph model price in US dollars per million input tokens written to the prompt cache, 0 for ollama")
	parser.add_argument(
		"--qna_kg_cache_read_price",
		type=float,
		required=True,
		help="Knowledge graph model price in US dollars per million input tokens read from the prompt cache, 0 for ollama")
	parser.add_argument(
		"--eval_qna_model",
		type=str,
		required=False,
		default=constants.EVAL_QNA_MODEL,
		help="Local HuggingFace embedding model RAGAS uses to build its knowledge graph")
	parser.add_argument(
		"--n_sample_documents",
		type=int,
		required=False,
		default=constants.SAMPLE_DOCUMENTS,
		help="Number of documents randomly picked for RAGAS to generate the eval QnA from")
	parser.add_argument(
		"--n_sample_evals",
		type=int,
		required=False,
		default=constants.N_EVAL_SAMPLES,
		help="Number of eval QnA items to load or generate")

def add_query_arguments(parser: argparse.ArgumentParser) -> None:
	"""
	Input: parser, the argument parser of eval.py.
	Output: nothing.
	What it does: adds the query arguments: the query LLM and the number of multi-query phrasings.
	"""
	parser.add_argument(
		"--query_llm",
		type=str,
		required=True,
		help="LLM provider that transforms the eval questions: claude, chatgpt, gemini or ollama")
	parser.add_argument(
		"--query_llm_model",
		type=str,
		required=True,
		help="Model the query LLM provider runs")
	parser.add_argument(
		"--query_input_price",
		type=float,
		required=True,
		help="Query model price in US dollars per million input tokens, 0 for ollama")
	parser.add_argument(
		"--query_output_price",
		type=float,
		required=True,
		help="Query model price in US dollars per million output tokens, 0 for ollama")
	parser.add_argument(
		"--query_cache_write_price",
		type=float,
		required=True,
		help="Query model price in US dollars per million input tokens written to the prompt cache, 0 for ollama")
	parser.add_argument(
		"--query_cache_read_price",
		type=float,
		required=True,
		help="Query model price in US dollars per million input tokens read from the prompt cache, 0 for ollama")
	parser.add_argument(
		"--n_multi_qus",
		type=int,
		required=False,
		default=constants.N_MULTI_QUS,
		help="Number of phrasings the query LLM writes for each eval question")

def add_retrieval_arguments(parser: argparse.ArgumentParser) -> None:
	"""
	Input: parser, the argument parser of eval.py.
	Output: nothing.
	What it does: adds the retrieval, reranking and tuning arguments.
	"""
	parser.add_argument(
		"--retrieval_types",
		type=str,
		nargs="+",
		required=False,
		default=[constants.DENSE_RETRIEVAL, constants.BM25_RETRIEVAL, constants.HYBRID_RETRIEVAL],
		help="Retrieval types to run, one after another: dense, bm25 and/or hybrid")
	parser.add_argument(
		"--query_types",
		type=str,
		nargs="+",
		required=False,
		default=[constants.ORIGINAL_QUERY, constants.REWRITTEN_QUERY, constants.MULTI_QUERY, constants.HYDE_QUERY],
		help="Query types to search with: original, rewritten, multi and/or hyde")
	parser.add_argument(
		"--top_k",
		type=int,
		required=False,
		default=constants.TOP_K,
		help="Number of chunks each retriever returns for a query")
	parser.add_argument(
		"--reranking",
		type=str,
		required=False,
		choices=["true", "false"],
		default="false",
		help="true to also rerank the top_k chunks of every retrieval with the cross-encoder, false not to")
	parser.add_argument(
		"--rerank_top_k",
		type=int,
		required=False,
		default=constants.RERANK_TOP_K,
		help="Number of chunks kept after the cross-encoder reranks the top_k chunks")
	parser.add_argument(
		"--tuning_queries_size",
		type=int,
		required=True,
		help="Number of eval questions used for tuning; the rest are held out for one final check")
	parser.add_argument(
		"--query_split",
		type=str,
		required=True,
		choices=["tuning", "held_out"],
		help="Queries this run uses: tuning while tuning, held_out for the one final check")
	parser.add_argument(
		"--n_retrieval_evaluation",
		type=int,
		required=False,
		default=constants.N_RETRIEVAL_EVALUATION,
		help="Number of best plain setups and of best reranked setups whose retrieved outputs go on to the next layers")

def add_generation_arguments(parser: argparse.ArgumentParser) -> None:
	"""
	Input: parser, the argument parser of eval.py.
	Output: nothing.
	What it does: adds the generation arguments: the generation LLM and the batch settings.
	"""
	parser.add_argument(
		"--generation_llm",
		type=str,
		required=True,
		help="LLM provider that answers the prompts: claude, chatgpt, gemini or ollama")
	parser.add_argument(
		"--generation_llm_model",
		type=str,
		required=True,
		help="Model the generation LLM provider runs")
	parser.add_argument(
		"--generation_input_price",
		type=float,
		required=True,
		help="Generation model price in US dollars per million input tokens, 0 for ollama")
	parser.add_argument(
		"--generation_output_price",
		type=float,
		required=True,
		help="Generation model price in US dollars per million output tokens, 0 for ollama")
	parser.add_argument(
		"--generation_cache_write_price",
		type=float,
		required=True,
		help="Generation model price in US dollars per million input tokens written to the prompt cache, 0 for ollama")
	parser.add_argument(
		"--generation_cache_read_price",
		type=float,
		required=True,
		help="Generation model price in US dollars per million input tokens read from the prompt cache, 0 for ollama")
	parser.add_argument(
		"--batch_poll_seconds",
		type=int,
		required=True,
		help="Seconds between status checks of a generation batch")
	parser.add_argument(
		"--batch_timeout_seconds",
		type=int,
		required=True,
		help="Seconds to wait for a generation batch before stopping; a later run resumes the same batch")

def add_answer_eval_arguments(parser: argparse.ArgumentParser) -> None:
	"""
	Input: parser, the argument parser of eval.py.
	Output: nothing.
	What it does: adds the answer evaluation arguments: the framework, the judge LLM and the embedding model.
	"""
	parser.add_argument(
		"--ans_eval_framework",
		type=str,
		required=True,
		help="Framework that scores the answers, e.g. ragas")
	parser.add_argument(
		"--ans_eval_llm",
		type=str,
		required=True,
		help="LLM provider that judges the answers: claude, chatgpt, gemini or ollama")
	parser.add_argument(
		"--ans_eval_llm_model",
		type=str,
		required=True,
		help="Model the answer evaluation LLM provider runs")
	parser.add_argument(
		"--ans_eval_input_price",
		type=float,
		required=True,
		help="Answer evaluation model price in US dollars per million input tokens, 0 for ollama")
	parser.add_argument(
		"--ans_eval_output_price",
		type=float,
		required=True,
		help="Answer evaluation model price in US dollars per million output tokens, 0 for ollama")
	parser.add_argument(
		"--ans_eval_cache_write_price",
		type=float,
		required=True,
		help="Answer evaluation model price in US dollars per million input tokens written to the prompt cache, 0 for ollama")
	parser.add_argument(
		"--ans_eval_cache_read_price",
		type=float,
		required=True,
		help="Answer evaluation model price in US dollars per million input tokens read from the prompt cache, 0 for ollama")
	parser.add_argument(
		"--ans_eval_embedding_model",
		type=str,
		required=True,
		help="Local HuggingFace embedding model the answer evaluation framework uses, e.g. all-MiniLM-L6-v2")

def add_analysis_arguments(parser: argparse.ArgumentParser) -> None:
	"""
	Input: parser, the argument parser of eval.py.
	Output: nothing.
	What it does: adds the analysis arguments: how many runs to compare and how many worst answers to list.
	"""
	parser.add_argument(
		"--n_compare_runs",
		type=int,
		required=True,
		help="Most recent runs compared in the analysis report, the current run included")
	parser.add_argument(
		"--n_worst_answers",
		type=int,
		required=True,
		help="Lowest scoring answers listed in the analysis report")

# ------ Layer settings from the parsed arguments
def make_ingest_args(parsed_args):
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_args().
	Output: an IngestArgs object holding the same values.
	What it does: copies each parsed command line value into the matching
	IngestArgs field.
	"""
	ingest_args = IngestArgs(
		dataset_size=parsed_args.dataset_size,
		input_path=parsed_args.input_path,
		output_path=parsed_args.output_path,
		document_type=parsed_args.document_type,
		framework=parsed_args.framework
	)
	return ingest_args

def make_chunking_args(parsed_args, chunk_type: str):
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_args(); chunk_type, one of parsed_args.chunk_types.
	Output: a ChunkingArgs object holding the chunking values for chunk_type.
	What it does: copies each parsed chunking command line value into the
	matching ChunkingArgs field.
	"""
	chunking_args = ChunkingArgs(
		framework=parsed_args.framework,
		model_name=parsed_args.model_name,
		chunk_type=chunk_type,
		min_chunk_size=parsed_args.min_chunk_size,
		max_chunk_size=parsed_args.max_chunk_size,
		fixed_chunk_overlap=parsed_args.fixed_chunk_overlap,
		semantic_threshold=parsed_args.semantic_threshold
	)
	return chunking_args

def make_index_args(parsed_args, index_type: str):
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_args(); index_type, one of
	parsed_args.index_types.
	Output: an IndexArgs object holding the indexing values for index_type.
	What it does: copies each parsed indexing command line value into the
	matching IndexArgs field.
	"""
	index_args = IndexArgs(
		index_type=index_type,
		ivf_nlist=parsed_args.ivf_nlist,
		hnsw_m=parsed_args.hnsw_m,
		hnsw_ef_construction=parsed_args.hnsw_ef_construction,
		bm25_k1=parsed_args.bm25_k1,
		bm25_b=parsed_args.bm25_b,
		bm25_tokenizer=parsed_args.bm25_tokenizer,
		qdrant_batch_size=parsed_args.qdrant_batch_size
	)
	return index_args

def make_eval_qna_args(parsed_args):
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_args().
	Output: an EvalQnAArgs object holding the QnA generator settings.
	What it does: copies the parsed eval embedding model name, number of sample documents, number of eval QnA items
	and framework into the matching EvalQnAArgs fields. The QnA LLMs are made with make_llm_args.
	"""
	eval_qna_args = EvalQnAArgs(
		eval_qna_model=parsed_args.eval_qna_model,
		n_sample_documents=parsed_args.n_sample_documents,
		n_sample_evals=parsed_args.n_sample_evals,
		framework=parsed_args.framework
	)
	return eval_qna_args

def make_query_args(parsed_args):
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_args().
	Output: a QueryArgs object holding the query transformation settings.
	What it does: copies the build's embedding model (model_name) and the number of multi-query phrasings into
	the matching QueryArgs fields. The query LLM is made with make_llm_args.
	"""
	query_args = QueryArgs(
		embedding_model=parsed_args.model_name,
		n_multi_qus=parsed_args.n_multi_qus
	)
	return query_args

def make_retrieval_args(parsed_args):
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_args().
	Output: a RetrievalArgs object holding the retrieval settings.
	What it does: copies the parsed framework, retrieval_types, query_types, top_k, reranking and rerank_top_k into the matching RetrievalArgs fields.
	"""
	retrieval_args = RetrievalArgs(
		framework=parsed_args.framework,
		retrieval_types=parsed_args.retrieval_types,
		query_types=parsed_args.query_types,
		top_k=parsed_args.top_k,
		reranking=parsed_args.reranking == "true",
		rerank_top_k=parsed_args.rerank_top_k
	)
	return retrieval_args

def make_llm_args(llm: str, llm_model: str, input_price: float, output_price: float, cache_write_price: float, cache_read_price: float, max_tokens: int, timeout_seconds: int, cache_ttl: str) -> LLMArgs:
	"""
	Input: llm, the LLM provider; llm_model, its model; input_price, output_price, cache_write_price and
	cache_read_price, in US dollars per million tokens; max_tokens, the maximum output tokens per call;
	timeout_seconds, the seconds before one call times out; cache_ttl, how long Claude keeps a cached prompt.
	Output: an LLMArgs object holding these settings.
	What it does: copies the LLM settings of one layer into an LLMArgs object.
	"""
	llm_args = LLMArgs(
		llm=llm,
		llm_model=llm_model,
		input_price=input_price,
		output_price=output_price,
		cache_write_price=cache_write_price,
		cache_read_price=cache_read_price,
		max_tokens=max_tokens,
		timeout_seconds=timeout_seconds,
		cache_ttl=cache_ttl
	)
	return llm_args

def make_generation_args(parsed_args) -> GenerationArgs:
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_args().
	Output: a GenerationArgs object holding the generation LLM and the batch settings.
	What it does: makes the generation LLM with make_llm_args and copies batch_poll_seconds, batch_timeout_seconds and
	framework.
	"""
	generation_llm_args = make_llm_args(
		parsed_args.generation_llm,
		parsed_args.generation_llm_model,
		parsed_args.generation_input_price,
		parsed_args.generation_output_price,
		parsed_args.generation_cache_write_price,
		parsed_args.generation_cache_read_price,
		parsed_args.llm_max_tokens,
		parsed_args.llm_timeout_seconds,
		parsed_args.llm_cache_ttl
	)
	generation_args = GenerationArgs(
		llm_args=generation_llm_args,
		batch_poll_seconds=parsed_args.batch_poll_seconds,
		batch_timeout_seconds=parsed_args.batch_timeout_seconds,
		framework=parsed_args.framework
	)
	return generation_args

def make_answer_eval_args(parsed_args) -> AnswerEvalArgs:
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_args().
	Output: an AnswerEvalArgs object holding the answer evaluation framework, the judge LLM and the embedding model.
	What it does: makes the judge LLM with make_llm_args and copies ans_eval_framework and ans_eval_embedding_model.
	"""
	ans_eval_llm_args = make_llm_args(
		parsed_args.ans_eval_llm,
		parsed_args.ans_eval_llm_model,
		parsed_args.ans_eval_input_price,
		parsed_args.ans_eval_output_price,
		parsed_args.ans_eval_cache_write_price,
		parsed_args.ans_eval_cache_read_price,
		parsed_args.llm_max_tokens,
		parsed_args.llm_timeout_seconds,
		parsed_args.llm_cache_ttl
	)
	answer_eval_args = AnswerEvalArgs(
		ans_eval_framework=parsed_args.ans_eval_framework,
		llm_args=ans_eval_llm_args,
		embedding_model=parsed_args.ans_eval_embedding_model,
		framework=parsed_args.framework
	)
	return answer_eval_args

def make_analysis_args(parsed_args) -> AnalysisArgs:
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_args().
	Output: an AnalysisArgs object holding n_compare_runs, n_worst_answers and framework.
	What it does: copies the parsed n_compare_runs, n_worst_answers and framework into the matching AnalysisArgs fields.
	"""
	analysis_args = AnalysisArgs(
		n_compare_runs=parsed_args.n_compare_runs,
		n_worst_answers=parsed_args.n_worst_answers,
		framework=parsed_args.framework
	)
	return analysis_args
