import argparse
import time

from common.args import IngestArgs
from l1_ingestion.ingestion import ingestion


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
		default=1,
		help="Number of parallel download workers")
	parser.add_argument(
		"--max_batch_rows",
		type=int,
		required=False,
		default=1,
		help="Maximum number of source rows per batch")
	parser.add_argument(
		"--hugging_face_split",
		type=str,
		required=False,
		default="train",
		help="Hugging Face dataset split to read")
	parser.add_argument(
		"--parse_workers",
		type=int,
		required=False,
		default=16,
		help="Number of parallel parse workers")
	parser.add_argument(
		"--parse_timeout_seconds",
		type=int,
		required=False,
		default=0,
		help="Per-document parse timeout in seconds")
	parser.add_argument(
		"--dedup_threshold",
		type=float,
		required=False,
		default=0.0,
		help="Near-duplicate similarity threshold")
	parser.add_argument(
		"--pdf_parser",
		type=str,
		required=False,
		default="pypdf",
		help="Library used to parse PDF files: pypdf or pypdfium2")

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


if __name__ == "__main__":
	print("\n ------------- RAG -------- \n")
	start = time.time()

	args = parse_args()
	print(args)

	# 1. Ingestion
	ingest_args = make_ingest_args(args)

	s1 = time.time()
	documents = ingestion(ingest_args)
	e1 = time.time() - s1
	print("\n------------------ Ingestion time : ", e1)

	# ---------
	end = time.time()
	elapsed_time = end - start
	print("\n\nElapsed time : ", elapsed_time)
