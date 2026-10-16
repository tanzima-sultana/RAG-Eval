import time

from build import run_build
from common.args import parse_eval_args
from eval import run_eval

if __name__ == "__main__":
	print("\n ------------- RAG Pipeline -------- \n")
	start = time.time()

	# The eval arguments hold every build and eval setting, so one parse covers the whole pipeline
	args = parse_eval_args()
	print(args)

	# 1. Build: l1 ingestion to l4 indexing. A cached layer is loaded, not built again
	print("\n\n========== Build ==========")
	s1 = time.time()
	run_build(args)
	print("\nBuild time : ", time.time() - s1)

	# 2. Eval: eval QnA, l5 query to l10 evaluation and the analysis
	print("\n\n========== Eval ==========")
	s2 = time.time()
	run_eval(args)
	print("\nEval time : ", time.time() - s2)

	# ---------
	end = time.time()
	elapsed_time = end - start
	print("\n\nElapsed time : ", elapsed_time)
