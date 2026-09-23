import argparse
import sys
import time
import os 

from common import paths
from common import timing
from common.chunk_config import ChunkConfig
from l4_indexing.indexing import Indexing

def parse_args():
	parser = argparse.ArgumentParser(description="Indexing Parameters")

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
		"--index_type",
		type=str,
		required=True,
		help="Index type")
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

	return parser.parse_args()


if __name__ == "__main__":
	print("\n ------------- L3 - Embedding -------- \n")
	timing.reset()
	start = time.time()

	args = parse_args()
	print(args)

	index_type = args.index_type
	dataset_size = args.dataset_size

	chunk_config = ChunkConfig(
		dataset_name=args.dataset_name,
		chunk_type=args.chunk_type,
		min_chunk_size=args.min_chunk_size,
		max_chunk_size=args.max_chunk_size,
		fixed_chunk_overlap=args.fixed_chunk_overlap,
		semantic_threshold=args.semantic_threshold,
	)

	indexer = Indexing(chunk_config, dataset_size)
	index = indexer.compute_index(index_type)

	end = time.time()
	elapsed_time = end - start
	print("\n\nEmbedding Elapsed time : ", elapsed_time)
	timing.print_summary()

	# huggingface `datasets`/fsspec leaves a background thread alive after
	# streaming a dataset, which crashes normal interpreter shutdown with
	# "Fatal Python error: PyGILState_Release" (known upstream bug,
	# huggingface/datasets#7357). All work is done by this point, so skip
	# Python's normal finalization/thread-teardown instead of hitting it.
	sys.stdout.flush()
	sys.stderr.flush()
	os._exit(0)