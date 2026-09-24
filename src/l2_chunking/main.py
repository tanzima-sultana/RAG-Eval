import argparse
import sys
import time
import os 

from common import paths
from common.chunk_config import ChunkConfig
from l2_chunking.chunking import Chunking
from l2_chunking.summary import print_chunk_statistics


def parse_args():
	parser = argparse.ArgumentParser(description="Chunking Parameters")

	parser.add_argument(
		"--dataset_name",
		type=str,
		required=True,
		help="Dataset name")
	parser.add_argument(
		"--dataset_size",
		type=int,
		required=True,
		help="Dataset size")
	parser.add_argument(
		"--chunk_type",
		type=str,
		required=True,
		help="Chunking type")
	parser.add_argument(
		"--min_chunk_size",
		type=int,
		required=True,
		help="Min chunk size")
	parser.add_argument(
		"--max_chunk_size",
		type=int,
		required=True,
		help="Max chunk size")
	parser.add_argument(
		"--fixed_chunk_overlap",
		type=int,
		required=True,
		help="Overlap for fixed chunking")
	parser.add_argument(
		"--semantic_threshold",
		type=float,
		required=False,
		default=0.0,
		help="Semantic threshold")
	parser.add_argument(
		"--model_name",
		type=str,
		required=True,
		help="Transformer model name")

	return parser.parse_args()

if __name__ == "__main__":
	print("\n ------------- L2 - Chunking -------- \n")
	start = time.time()

	args = parse_args()
	print(args)

	dataset_size = args.dataset_size
	model_name = args.model_name

	chunk_config = ChunkConfig(
		dataset_name=args.dataset_name,
		chunk_type=args.chunk_type,
		min_chunk_size=args.min_chunk_size,
		max_chunk_size=args.max_chunk_size,
		fixed_chunk_overlap=args.fixed_chunk_overlap,
		semantic_threshold=args.semantic_threshold,
	)

	chunker = Chunking(chunk_config, dataset_size, model_name)
	chunks = chunker.compute_chunks()

	print_chunk_statistics(chunks, paths.get_chunk_path(chunk_config))

	end = time.time()
	elapsed_seconds = end - start
	elapsed_minutes = int(elapsed_seconds // 60)
	remaining_seconds = elapsed_seconds - elapsed_minutes * 60
	print("\n----- L2 total time -----")
	print(f"Total : {elapsed_seconds:.3f}s ({elapsed_minutes}m {remaining_seconds:.1f}s)")

	# huggingface `datasets`/fsspec leaves a background thread alive after
	# streaming a dataset, which crashes normal interpreter shutdown with
	# "Fatal Python error: PyGILState_Release" (known upstream bug,
	# huggingface/datasets#7357). All work is done by this point, so skip
	# Python's normal finalization/thread-teardown instead of hitting it.
	sys.stdout.flush()
	sys.stderr.flush()
	os._exit(0)