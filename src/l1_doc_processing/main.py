
import argparse
import sys
import time
import os 

from common import timing
from l1_doc_processing.doc_processing import DocProcessing

def parse_args():
    parser = argparse.ArgumentParser(description="Document processing Parameters")

    parser.add_argument(
        "--doc_type",
        type=str,
        required=True,
        help="Document type")
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name")
    parser.add_argument(
        "--dataset_size",
        type=int,
        required=True,
        help="Number of documents in the dataset to process")
    parser.add_argument(
        "--dataset_url",
        type=str,
        required=False,
        default=None,
        help="Dataset web URL")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=False,
        default=None,
        help="Dataset local path")

    return parser.parse_args()


def validate(parsed_args):
    url = parsed_args.dataset_url is not None
    path = parsed_args.dataset_path is not None

    if url and path:
        sys.exit("Error: provide only one of --dataset_url or --dataset_path, not both.")

    if not url and not path:
        sys.exit("Error: at least one of --dataset_url or --dataset_path is required.")

if __name__ == "__main__":
    print("\n ------------- L1 - Document processing -------- \n")
    timing.reset()
    start = time.time()

    args = parse_args()
    print(args)
    validate(args)
    
    doc_type = args.doc_type
    dataset_name = args.dataset_name
    dataset_size = args.dataset_size
    dataset_url = args.dataset_url
    dataset_path = args.dataset_path

    processor = DocProcessing(doc_type, dataset_name, dataset_size, dataset_url, dataset_path)
    documents = processor.process()
    size = len(documents.values())

    end = time.time()
    elapsed_time = end - start
    print("\n\nDocument processing Elapsed time : ", elapsed_time)
    timing.print_summary()

    # huggingface `datasets`/fsspec leaves a background thread alive after
    # streaming a dataset, which crashes normal interpreter shutdown with
    # "Fatal Python error: PyGILState_Release" (known upstream bug,
    # huggingface/datasets#7357). All work is done by this point, so skip
    # Python's normal finalization/thread-teardown instead of hitting it.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)