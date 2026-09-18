import argparse
import sys
import time
import os 

from common import paths
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
    min_chunk_size = args.min_chunk_size
    max_chunk_size = args.max_chunk_size
    fixed_chunk_overlap = args.fixed_chunk_overlap
    semantic_threshold = args.semantic_threshold
    model_name = args.model_name

    chunker = Chunking(min_chunk_size, max_chunk_size, fixed_chunk_overlap, semantic_threshold, model_name)
    chunks = chunker.compute_chunks(chunk_type)

    end = time.time()
    elapsed_time = end - start
    print("\n\nChunking Elapsed time : ", elapsed_time)
    timing.print_summary()

    # Chunk-size and output-file-size statistics
    total_chunk_count = len(chunks)

    total_token_size = 0
    for chunk_id in chunks:
        total_token_size += chunks[chunk_id].size
    if total_chunk_count > 0:
        average_chunk_token_size = total_token_size / total_chunk_count
    else:
        average_chunk_token_size = 0

    output_path = paths.get_chunk_path(chunk_type)
    chunk_file_names = os.listdir(output_path)
    total_file_byte_size = 0
    for chunk_file_name in chunk_file_names:
        file_path = os.path.join(output_path, chunk_file_name)
        total_file_byte_size += os.path.getsize(file_path)
    chunk_file_count = len(chunk_file_names)
    if chunk_file_count > 0:
        average_file_byte_size = total_file_byte_size / chunk_file_count
    else:
        average_file_byte_size = 0

    print("\n----- Chunk statistics -----")
    print("Total chunks produced   :", total_chunk_count)
    print("Average chunk size (tokens):", average_chunk_token_size)
    print("Total chunk output size (bytes):", total_file_byte_size)
    print("Average chunk file size (bytes):", average_file_byte_size)
    print("-------------------------------------------")

    # huggingface `datasets`/fsspec leaves a background thread alive after
    # streaming a dataset, which crashes normal interpreter shutdown with
    # "Fatal Python error: PyGILState_Release" (known upstream bug,
    # huggingface/datasets#7357). All work is done by this point, so skip
    # Python's normal finalization/thread-teardown instead of hitting it.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)