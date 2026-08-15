import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import BUILD_LOG_PATH
from src.dataset import load_dataset
from src.local.chunking import chunk_dataset


class _Tee:
    """Duplicates writes to multiple streams, so print() reaches both the
    console and the build log file."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


def parse_args():
    parser = argparse.ArgumentParser(description="Build phase of the RAG pipeline")
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=1000,
        help="Number of rows to include in the processed dataset",
    )
    parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=500,
        help="Maximum number of characters per chunk",
    )
    parser.add_argument(
        "--fixed-chunk-overlap",
        type=int,
        default=50,
        help="Character overlap between consecutive fixed-size chunks",
    )
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.75,
        help="Minimum cosine similarity between consecutive sentences to keep them in the same semantic chunk",
    )
    return parser.parse_args()


def run_build(args):
    dataset_path = load_dataset(args.dataset_size)
    print(f"Processed dataset saved to: {dataset_path}")

    chunk_paths = chunk_dataset(
        dataset_path,
        max_chunk_size=args.max_chunk_size,
        fixed_chunk_overlap=args.fixed_chunk_overlap,
        semantic_threshold=args.semantic_threshold,
    )
    for strategy, path in chunk_paths.items():
        print(f"{strategy} chunks saved to: {path}")


def main():
    args = parse_args()

    original_stdout = sys.stdout
    with open(BUILD_LOG_PATH, "w") as log_file:
        sys.stdout = _Tee(original_stdout, log_file)
        try:
            run_build(args)
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    main()
