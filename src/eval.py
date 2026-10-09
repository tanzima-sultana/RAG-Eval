import argparse
import time

from common import constants
from common import paths
from common.args import IngestArgs
from common.args import ChunkingArgs
from common.args import IndexArgs
from l1_ingestion.ingestion import ingestion
from l2_chunking.cache import load_chunks
from l4_indexing import cache as index_cache

def parse_args():
	"""
	Input: none. Reads the command line arguments passed to eval.py.
	Output: an argparse.Namespace with one attribute per argument.
	What it does: defines the arguments that identify which build to
	evaluate and parses them. They must have the same values the build was
	run with, because the chunk, embedding and index folders and the
	Qdrant collection name are built from them. Defaults match build.py.
	"""
	parser = argparse.ArgumentParser(description="RAG-Eval evaluation parameters")

	parser.add_argument(
		"--dataset_size",
		type=int,
		required=False,
		default=0,
		help="Number of documents the build was made with")
	parser.add_argument(
		"--input_path",
		type=str,
		required=False,
		default="",
		help="Input dataset local path or Hugging Face dataset id used by the build")
	parser.add_argument(
		"--output_path",
		type=str,
		required=False,
		default="",
		help="Base output folder the build wrote to")
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
	parser.add_argument(
		"--framework",
		type=str,
		required=False,
		default="",
		help="Framework the build used: langchain, or empty for the built-in layers")
	parser.add_argument(
		"--model_name",
		type=str,
		required=False,
		default="",
		help="SentenceTransformer model the build used")
	parser.add_argument(
		"--chunk_type",
		type=str,
		required=False,
		default=constants.FIXED_CHUNKING,
		help="Chunking strategy the build used: fixed, sentence or semantic")
	parser.add_argument(
		"--min_chunk_size",
		type=int,
		required=False,
		default=constants.MIN_CHUNK_SIZE,
		help="Minimum chunk size the build used")
	parser.add_argument(
		"--max_chunk_size",
		type=int,
		required=False,
		default=constants.MAX_CHUNK_SIZE,
		help="Maximum chunk size the build used")
	parser.add_argument(
		"--fixed_chunk_overlap",
		type=int,
		required=False,
		default=constants.FIXED_CHUNK_OVERLAP,
		help="Fixed chunk overlap the build used")
	parser.add_argument(
		"--semantic_threshold",
		type=float,
		required=False,
		default=constants.SEMANTIC_THRESHOLD,
		help="Semantic threshold the build used")
	parser.add_argument(
		"--faiss_index_type",
		type=str,
		required=False,
		default=constants.FAISS_FLATIP_INDEX,
		help="FAISS index type the build used: faiss_flatip, faiss_ivf or faiss_hnsw")
	parser.add_argument(
		"--ivf_nlist",
		type=int,
		required=False,
		default=constants.IVF_NLIST,
		help="Number of IVF clusters the build used")
	parser.add_argument(
		"--hnsw_m",
		type=int,
		required=False,
		default=constants.HNSW_M,
		help="HNSW neighbours per node the build used")
	parser.add_argument(
		"--hnsw_ef_construction",
		type=int,
		required=False,
		default=constants.HNSW_EF_CONSTRUCTION,
		help="HNSW build candidate list size the build used")
	parser.add_argument(
		"--bm25_k1",
		type=float,
		required=False,
		default=constants.BM25_K1,
		help="BM25 k1 the build used")
	parser.add_argument(
		"--bm25_b",
		type=float,
		required=False,
		default=constants.BM25_B,
		help="BM25 b the build used")
	parser.add_argument(
		"--bm25_tokenizer",
		type=str,
		required=False,
		default=constants.BM25_TOKENIZER,
		help="BM25 tokenizer the build used")

	return parser.parse_args()


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


def make_chunking_args(parsed_args):
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_args().
	Output: a ChunkingArgs object holding the chunking values of the build.
	What it does: copies each parsed chunking value into the matching
	ChunkingArgs field.
	"""
	chunking_args = ChunkingArgs(
		framework=parsed_args.framework,
		model_name=parsed_args.model_name,
		chunk_type=parsed_args.chunk_type,
		min_chunk_size=parsed_args.min_chunk_size,
		max_chunk_size=parsed_args.max_chunk_size,
		fixed_chunk_overlap=parsed_args.fixed_chunk_overlap,
		semantic_threshold=parsed_args.semantic_threshold
	)
	return chunking_args


def make_index_args(parsed_args):
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_args().
	Output: an IndexArgs object holding the index values of the build.
	What it does: copies each parsed index value into the matching
	IndexArgs field. qdrant_batch_size only matters while building, so it
	is set to its default.
	"""
	index_args = IndexArgs(
		faiss_index_type=parsed_args.faiss_index_type,
		ivf_nlist=parsed_args.ivf_nlist,
		hnsw_m=parsed_args.hnsw_m,
		hnsw_ef_construction=parsed_args.hnsw_ef_construction,
		bm25_k1=parsed_args.bm25_k1,
		bm25_b=parsed_args.bm25_b,
		bm25_tokenizer=parsed_args.bm25_tokenizer,
		qdrant_batch_size=constants.QDRANT_BATCH_SIZE
	)
	return index_args

if __name__ == "__main__":
	print("\n ------------- RAG Eval -------- \n")
	start = time.time()

	args = parse_args()
	print(args)

	ingest_args = make_ingest_args(args)
	chunking_args = make_chunking_args(args)
	index_args = make_index_args(args)
	output_path = paths.get_output_path(ingest_args.input_path, ingest_args.output_path)

	# 1. Load the l1 documents
	# The same ingestion call as build.py; it returns the documents from the l1 cache
	print("\n\nLoading documents")
	s1 = time.time()
	documents = ingestion(ingest_args)
	e1 = time.time() - s1
	print("\nLoading documents time : ", e1)

	# 2. Load the l2 chunks of those documents
	# Load-only; raises an error if a document has no chunk file
	print("\n\nLoading chunks")
	s2 = time.time()
	chunk_path = paths.get_chunk_path(output_path, chunking_args)
	doc_ids = []
	for doc_id in documents:
		doc_ids.append(doc_id)
	chunks = load_chunks(chunk_path, doc_ids)
	e2 = time.time() - s2
	print(f"Chunk folder : {chunk_path}")
	print(f"Chunks loaded : {len(chunks)} from {len(doc_ids)} documents")
	print("\nLoading chunks time : ", e2)

	# 3. Load the built indexes
	# Raises an error if any index is missing or was built with a different dataset size
	print("\n\nLoading indexes")
	s3 = time.time()
	faiss_index, bm25_index, qdrant_index = index_cache.load_index(output_path, chunking_args, index_args, args.dataset_size)
	e3 = time.time() - s3
	print("\nLoading indexes time : ", e3)

	# ---------
	end = time.time()
	elapsed_time = end - start
	print("\n\nElapsed time : ", elapsed_time)
