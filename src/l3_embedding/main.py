import argparse
import sys
import time
import os 

from common import paths
from common import timing

from l3_embedding.embedding import Embedding

def parse_args():
	parser = argparse.ArgumentParser(description="Embedding Parameters")

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
	print("\n ------------- L3 - Embedding -------- \n")
	timing.reset()
	start = time.time()

	args = parse_args()
	print(args)

	dataset_name = args.dataset_name
	dataset_size = args.dataset_size

	chunk_type = args.chunk_type
	min_chunk_size = args.min_chunk_size
	max_chunk_size = args.max_chunk_size
	fixed_chunk_overlap = args.fixed_chunk_overlap
	semantic_threshold = args.semantic_threshold
	model_name = args.model_name

	embedder = Embedding(dataset_name, dataset_size, min_chunk_size, max_chunk_size, fixed_chunk_overlap, semantic_threshold, model_name)
	embeddings = embedder.compute_embeddings(chunk_type)

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