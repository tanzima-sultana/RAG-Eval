import argparse
import sys
import time
import os 

from common import timing
from common.constants import SEMANTIC_CHUNKING
from l2_chunking.chunking import Chunking


def parse_args():
    parser = argparse.ArgumentParser(description="Chunking Parameters")

    parser.add_argument(
        "--chunk_type",
        type=str,
        required=True,
        help="Chunking type")
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

def validate(parsed_args):
    chunk_type = parsed_args.chunk_type
    semantic_threshold = parsed_args.semantic_threshold

    if chunk_type == SEMANTIC_CHUNKING and semantic_threshold == 0.0:
        sys.exit("Error: provide valid semantic threshold for semantic chunking")

if __name__ == "__main__":
    print("\n ------------- L2 - Chunking -------- \n")
    timing.reset()
    start = time.time()

    args = parse_args()
    print(args)
    validate(args)
    
    chunk_type = args.chunk_type
    max_chunk_size = args.max_chunk_size
    fixed_chunk_overlap = args.fixed_chunk_overlap
    semantic_threshold = args.semantic_threshold
    model_name = args.model_name

    chunker = Chunking(max_chunk_size, fixed_chunk_overlap, semantic_threshold, model_name)
    chunks = chunker.compute_chunks(chunk_type)

    end = time.time()
    elapsed_time = end - start
    print("\n\nChunking Elapsed time : ", elapsed_time)
    timing.print_summary()

    # huggingface `datasets`/fsspec leaves a background thread alive after
    # streaming a dataset, which crashes normal interpreter shutdown with
    # "Fatal Python error: PyGILState_Release" (known upstream bug,
    # huggingface/datasets#7357). All work is done by this point, so skip
    # Python's normal finalization/thread-teardown instead of hitting it.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)