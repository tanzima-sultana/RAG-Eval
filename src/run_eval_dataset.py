import time

from common import paths
from common.args import parse_eval_dataset_config
from common.args import make_eval_dataset_config_args
from common.args import make_ollama_args
from evaluation.eval_dataset import EvalDataset


def run_eval_dataset(args) -> None:
	"""
	Input: args, the parsed args of the eval dataset config.
	Output: nothing.
	What it does: loads the saved eval dataset, or, only if there is none, generates and saves one with its metadata.
	Generating calls a paid LLM, so it needs ALLOW_PAID_API=1.
	"""
	eval_dataset_args = make_eval_dataset_config_args(args)
	ollama_args = make_ollama_args(args)
	output_path = paths.get_output_path(eval_dataset_args.output_path, eval_dataset_args.dataset_name)

	print("\n\nEval dataset")
	s1 = time.time()
	eval_samples = EvalDataset(eval_dataset_args, ollama_args).compute_eval_dataset(output_path)
	e1 = time.time() - s1
	print(f"Eval samples : {len(eval_samples)}")
	print("\nEval dataset time : ", e1)


if __name__ == "__main__":
	print("\n ------------- Eval dataset -------- \n")
	start = time.time()

	"""
	Input: --config and any override flags from the command line.
	Output: nothing.
	What it does: parses the eval dataset config and runs the eval dataset step.
	"""
	args = parse_eval_dataset_config()
	run_eval_dataset(args)

	elapsed_time = time.time() - start
	print("\n\nElapsed time : ", elapsed_time)
