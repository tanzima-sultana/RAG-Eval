import argparse
import os
import time

from common import constants
from common import paths
from common.args import IngestArgs
from common.args import ChunkingArgs
from common.args import IndexArgs
from l1_ingestion.ingestion import ingestion
from l2_chunking.chunking import Chunking
from l3_embedding.embedding import Embedding
from l4_indexing.indexing import Indexing

def parse_args():
	"""
	Input: none. Reads the command line arguments passed to rag.py.
	Output: an argparse.Namespace with one attribute per IngestArgs field.
	What it does: defines the ingestion command line parameters and parses
	them. Defaults match the defaults in IngestArgs.
	"""
	parser = argparse.ArgumentParser(description="RAG-Eval Parameters")

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
		"--chunk_type",
		type=str,
		required=False,
		default=constants.FIXED_CHUNKING,
		help="Chunking strategy: fixed, sentence or semantic")
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
	parser.add_argument(
		"--faiss_index_type",
		type=str,
		required=False,
		default=constants.FAISS_FLATIP_INDEX,
		help="FAISS index type: faiss_flatip, faiss_ivf or faiss_hnsw")
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
	Output: a ChunkingArgs object holding the chunking values.
	What it does: copies each parsed chunking command line value into the
	matching ChunkingArgs field.
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
	Output: an IndexArgs object holding the indexing values.
	What it does: copies each parsed indexing command line value into the
	matching IndexArgs field.
	"""
	index_args = IndexArgs(
		faiss_index_type=parsed_args.faiss_index_type,
		ivf_nlist=parsed_args.ivf_nlist,
		hnsw_m=parsed_args.hnsw_m,
		hnsw_ef_construction=parsed_args.hnsw_ef_construction,
		bm25_k1=parsed_args.bm25_k1,
		bm25_b=parsed_args.bm25_b,
		bm25_tokenizer=parsed_args.bm25_tokenizer,
		qdrant_batch_size=parsed_args.qdrant_batch_size
	)
	return index_args

if __name__ == "__main__":
	print("\n ------------- RAG -------- \n")
	start = time.time()

	args = parse_args()
	print(args)

	# 1. Ingestion
	print("\n\nIngestion")
	ingest_args = make_ingest_args(args)

	s1 = time.time()
	documents = ingestion(ingest_args)
	e1 = time.time() - s1
	print("\nIngestion time : ", e1)

	output_path = paths.get_output_path(ingest_args.input_path, ingest_args.output_path)

	# 2. Chunking
	print("\n\nChunking")
	chunking_args = make_chunking_args(args)

	s2 = time.time()
	chunks = Chunking(chunking_args).compute_chunks(documents, output_path)
	e2 = time.time() - s2
	print("\nChunking time : ", e2)

	# 3. Embedding
	print("\n\nEmbedding")

	s3 = time.time()
	embeddings = Embedding(chunking_args).compute_embeddings(chunks, output_path)
	e3 = time.time() - s3
	print("\nEmbedding time : ", e3)

	# 4. Indexing
	print("\n\nIndexing")
	index_args = make_index_args(args)

	s4 = time.time()
	faiss_index, bm25_index, qdrant_index = Indexing(index_args, chunking_args).compute_index(chunks, embeddings, output_path)
	e4 = time.time() - s4
	print("\nIndexing time : ", e4)

	# ---------
	end = time.time()
	elapsed_time = end - start
	print("\n\nElapsed time : ", elapsed_time)
