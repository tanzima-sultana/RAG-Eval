import json
import os
import time

from common import constants
from common.args import make_query_args
from common.args import parse_benchmark_config
from evaluation.eval_dataset.cache import load_eval_dataset
from rag.serving.query.query import QueryTransformation
from schema.eval_dataset_metadata import EvalDatasetMetadata


def run_benchmark(args) -> None:
	"""
	Input: args, the parsed args of the benchmark config.
	Output: nothing.
	What it does: reads the dataset name from the first entry of the build manifest the config names, then reads the
	eval dataset metadata the config names and checks it is of the same dataset, then loads the saved eval samples the
	metadata points to, then makes the queries of every eval question, one query type at a time. The other benchmark
	steps are added later.
	"""
	# 1. The build manifest: only the dataset name of its first entry
	print("\n\nBuild manifest")
	s1 = time.time()

	with open(args.manifest_path, encoding="utf-8") as f:
		manifest_dicts = json.load(f)
	manifest_dataset_name = manifest_dicts[0]["dataset_name"]
	print(f"Build manifest : {args.manifest_path}")
	print(f"Dataset name : {manifest_dataset_name}")

	e1 = time.time() - s1
	print("\nBuild manifest time : ", e1)

	# 2. The eval dataset metadata, which must be of the same dataset as the build manifest
	print("\n\nEval dataset")
	s2 = time.time()

	with open(args.eval_dataset_metadata_path, encoding="utf-8") as f:
		eval_dataset_metadata_dict = json.load(f)
	eval_dataset_metadata = EvalDatasetMetadata(**eval_dataset_metadata_dict)
	print(f"Eval dataset metadata : {args.eval_dataset_metadata_path}")

	if manifest_dataset_name != eval_dataset_metadata.dataset_name:
		raise ValueError(
			f"The build manifest is of dataset {manifest_dataset_name}, but the eval dataset metadata is of dataset "
			f"{eval_dataset_metadata.dataset_name}."
		)

	# 3. The eval samples the metadata names
	eval_samples = load_eval_dataset(eval_dataset_metadata.eval_dataset_path)
	if eval_samples is None:
		raise FileNotFoundError(
			f"No saved eval dataset at {eval_dataset_metadata.eval_dataset_path}. Run scripts/eval_dataset.sh first."
		)
	print(f"Eval dataset : {len(eval_samples)} from {eval_dataset_metadata.eval_dataset_path}")

	e2 = time.time() - s2
	print("\nEval dataset time : ", e2)

	# 4. The queries of every eval question, one query type at a time. Original must always be one of the types
	print("\n\nQuery transformation")
	if len(args.query_types) == 0:
		raise ValueError("query_types is empty. It must hold at least original.")
	if constants.ORIGINAL_QUERY not in args.query_types:
		raise ValueError(f"query_types ({args.query_types}) must always hold \"{constants.ORIGINAL_QUERY}\".")

	s3 = time.time()
	eval_sample_ids = []
	eval_questions = []
	for eval_sample in eval_samples:
		eval_sample_ids.append(eval_sample.id)
		eval_questions.append(eval_sample.qus)
	query_path = os.path.join(args.output_path, "queries")

	# Against each eval sample id, one (queries, hashes, paths) entry per query type, in the order of query_types
	queries_by_eval_sample_id = {}
	for eval_sample_id in eval_sample_ids:
		queries_by_eval_sample_id[eval_sample_id] = []

	for query_type in args.query_types:
		print(f"\nQuery type : {query_type}")
		query_args = make_query_args(args, query_type)
		query_transformation = QueryTransformation(query_args, query_path)
		query_results = query_transformation.query_batch(eval_sample_ids, eval_questions)

		query_count = 0
		for eval_sample_id in eval_sample_ids:
			queries, query_hashes, query_file_paths = query_results[eval_sample_id]
			queries_by_eval_sample_id[eval_sample_id].append((queries, query_hashes, query_file_paths))
			query_count += len(queries)
		print(f"Queries : {query_count}")

	e3 = time.time() - s3
	print("\nQuery transformation time : ", e3)


if __name__ == "__main__":
	print("\n ------------- Benchmark -------- \n")
	start = time.time()

	"""
	Input: --config and any override flags from the command line.
	Output: nothing.
	What it does: parses the benchmark config and runs the benchmark.
	"""
	args = parse_benchmark_config()
	run_benchmark(args)

	elapsed_time = time.time() - start
	print("\n\nElapsed time : ", elapsed_time)
