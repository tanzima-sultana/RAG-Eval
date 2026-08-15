import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Build phase of the RAG pipeline")
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=1000,
        help="Number of rows to include in the processed dataset",
    )
    args = parser.parse_args()

    dataset_path = load_dataset(args.dataset_size)
    print(f"Processed dataset saved to: {dataset_path}")


if __name__ == "__main__":
    main()
