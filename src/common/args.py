"""
args.py

The settings of each layer as dataclasses, and the command line arguments of build.py and eval.py that fill them.
"""

import argparse
import dataclasses

from common import constants

@dataclasses.dataclass
class IngestArgs:
	dataset_size: int 
	input_path: str
	output_path: str 
	download_workers: int 
	max_batch_rows: int 
	hugging_face_split: str 
	parse_workers: int 
	parse_timeout_seconds: int 
	dedup_threshold: float 
	pdf_parser: str 

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
	eval_qna_llm: str
	eval_qna_model: str
	n_sample_documents: int
	n_sample_evals: int

@dataclasses.dataclass
class QueryArgs:
	embedding_model: str
	query_llm: str
	query_llm_model: str
	n_multi_qus: int

@dataclasses.dataclass
class RetrievalArgs:
	framework: str
	retrieval_types: list[str]
	top_k: int
	reranking: bool
	rerank_top_k: int

# ------ Command line arguments
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
		help="Input dataset local path or Hugging Face dataset id")
	parser.add_argument(
		"--output_path",
		type=str,
		required=False,
		default="",
		help="Directory where ingested documents are written")
	parser.add_argument(
		"--download_workers",
		type=int,
		required=False,
		default=constants.DOWNLOAD_WORKERS,
		help="Number of parallel download workers")
	parser.add_argument(
		"--max_batch_rows",
		type=int,
		required=False,
		default=constants.MAX_BATCH_ROWS,
		help="Maximum number of source rows per batch")
	parser.add_argument(
		"--hugging_face_split",
		type=str,
		required=False,
		default=constants.HUGGING_FACE_SPLIT,
		help="Hugging Face dataset split to read")
	parser.add_argument(
		"--parse_workers",
		type=int,
		required=False,
		default=constants.PARSE_WORKERS,
		help="Number of parallel parse workers")
	parser.add_argument(
		"--parse_timeout_seconds",
		type=int,
		required=False,
		default=constants.PARSE_TIMEOUT_SECONDS,
		help="Per-document parse timeout in seconds")
	parser.add_argument(
		"--dedup_threshold",
		type=float,
		required=False,
		default=constants.DEDUP_THRESHOLD,
		help="Near-duplicate similarity threshold")
	parser.add_argument(
		"--pdf_parser",
		type=str,
		required=False,
		default=constants.PDF_PARSER,
		help="Library used to parse PDF files: pypdf or pypdfium2")

def add_chunking_arguments(parser: argparse.ArgumentParser) -> None:
	"""
	Input: parser, the argument parser of build.py or eval.py.
	Output: nothing.
	What it does: adds the chunking and embedding arguments, which build.py and eval.py both take.
	"""
	parser.add_argument(
		"--framework",
		type=str,
		required=False,
		default="",
		help="Chunking framework: langchain, or empty for the built-in chunkers")
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

def parse_build_args():
	"""
	Input: none. Reads the command line arguments passed to build.py.
	Output: an argparse.Namespace with one attribute per argument.
	What it does: defines the ingestion, chunking and indexing arguments and parses them.
	"""
	parser = argparse.ArgumentParser(description="RAG-Eval build parameters")
	add_ingest_arguments(parser)
	add_chunking_arguments(parser)
	add_index_arguments(parser)
	return parser.parse_args()

def parse_eval_args():
	"""
	Input: none. Reads the command line arguments passed to eval.py.
	Output: an argparse.Namespace with one attribute per argument.
	What it does: defines the arguments of the build to evaluate, which must have the values the build was run
	with, and the eval QnA, query, retrieval and tuning arguments, and parses them.
	"""
	parser = argparse.ArgumentParser(description="RAG-Eval evaluation parameters")
	add_ingest_arguments(parser)
	add_chunking_arguments(parser)
	add_index_arguments(parser)
	parser.add_argument(
		"--n_sample_evals",
		type=int,
		required=False,
		default=constants.N_EVAL_SAMPLES,
		help="Number of eval QnA items to load or generate")
	parser.add_argument(
		"--eval_qna_llm",
		type=str,
		required=False,
		default=constants.EVAL_QNA_LLM,
		help="LLM provider RAGAS uses to generate the eval QnA: claude")
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
		"--query_llm",
		type=str,
		required=False,
		default=constants.QUERY_LLM,
		help="Local LLM provider that transforms the eval questions: ollama")
	parser.add_argument(
		"--query_llm_model",
		type=str,
		required=False,
		default=constants.QUERY_LLM_MODEL,
		help="Local model the query LLM provider runs, e.g. llama3.1:8b")
	parser.add_argument(
		"--n_multi_qus",
		type=int,
		required=False,
		default=constants.N_MULTI_QUS,
		help="Number of phrasings the query LLM writes for each eval question")
	parser.add_argument(
		"--retrieval_types",
		type=str,
		nargs="+",
		required=False,
		default=[constants.DENSE_RETRIEVAL, constants.BM25_RETRIEVAL, constants.HYBRID_RETRIEVAL],
		help="Retrieval types to run, one after another: dense, bm25 and/or hybrid")
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
		"--n_retrieval_quality",
		type=int,
		required=False,
		default=constants.N_RETRIEVAL_QUALITY,
		help="Number of best plain setups and of best reranked setups whose retrieved outputs go on to the next layers")

	return parser.parse_args()

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
		download_workers=parsed_args.download_workers,
		max_batch_rows=parsed_args.max_batch_rows,
		hugging_face_split=parsed_args.hugging_face_split,
		parse_workers=parsed_args.parse_workers,
		parse_timeout_seconds=parsed_args.parse_timeout_seconds,
		dedup_threshold=parsed_args.dedup_threshold,
		pdf_parser=parsed_args.pdf_parser
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
	What it does: copies the parsed LLM generator, eval embedding
	model name, number of sample documents and number of eval QnA items into
	the matching EvalQnAArgs fields.
	"""
	eval_qna_args = EvalQnAArgs(
		eval_qna_llm=parsed_args.eval_qna_llm,
		eval_qna_model=parsed_args.eval_qna_model,
		n_sample_documents=parsed_args.n_sample_documents,
		n_sample_evals=parsed_args.n_sample_evals
	)
	return eval_qna_args


def make_query_args(parsed_args):
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_args().
	Output: a QueryArgs object holding the query transformation settings.
	What it does: copies the build's embedding model (model_name), the parsed
	query LLM provider, its model and the number of multi-query phrasings into
	the matching QueryArgs fields.
	"""
	query_args = QueryArgs(
		embedding_model=parsed_args.model_name,
		query_llm=parsed_args.query_llm,
		query_llm_model=parsed_args.query_llm_model,
		n_multi_qus=parsed_args.n_multi_qus
	)
	return query_args


def make_retrieval_args(parsed_args):
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_args().
	Output: a RetrievalArgs object holding the retrieval settings.
	What it does: copies the parsed framework, retrieval_types, top_k, reranking and rerank_top_k into the matching RetrievalArgs fields.
	"""
	retrieval_args = RetrievalArgs(
		framework=parsed_args.framework,
		retrieval_types=parsed_args.retrieval_types,
		top_k=parsed_args.top_k,
		reranking=parsed_args.reranking == "true",
		rerank_top_k=parsed_args.rerank_top_k
	)
	return retrieval_args
