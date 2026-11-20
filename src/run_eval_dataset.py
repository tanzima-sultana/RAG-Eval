import time

from common import paths_temp
from common.args_temp import parse_args
from common.args_temp import make_ingestion_args
from common.args_temp import make_eval_dataset_args
from common.args_temp import make_ollama_args
from rag.build.ingestion import ingestion
from evaluation.eval_dataset import EvalDataset


def run_eval_dataset(args) -> None:
	"""
	Input: args, the parsed args.
	Output: nothing.
	What it does: loads the documents, then loads the saved eval dataset, or generates and saves one if there is none.
	Generating calls a paid LLM, so it needs ALLOW_PAID_API=1.
	"""
	# 1. Ingestion
	print("\n\nIngestion")
	ingestion_args = make_ingestion_args(args)

	s1 = time.time()
	documents = ingestion(ingestion_args)
	e1 = time.time() - s1
	print(f"Documents : {len(documents)}")
	print("\nIngestion time : ", e1)

	output_path = paths_temp.get_output_path(args.output_path, args.dataset_name)

	# 2. Eval dataset
	print("\n\nEval dataset")
	eval_dataset_args = make_eval_dataset_args(args)
	ollama_args = make_ollama_args(args)

	s2 = time.time()
	eval_samples = EvalDataset(eval_dataset_args, ollama_args).compute_eval_dataset(documents, output_path)
	e2 = time.time() - s2
	print(f"Eval samples : {len(eval_samples)}")
	print("\nEval dataset time : ", e2)


if __name__ == "__main__":
	print("\n ------------- Eval dataset -------- \n")
	start = time.time()

	"""
	Input: --config and any override flags from the command line.
	Output: nothing.
	What it does: parses the args and runs the eval dataset step.
	"""
	args = parse_args()
	run_eval_dataset(args)

	elapsed_time = time.time() - start
	print("\n\nElapsed time : ", elapsed_time)
