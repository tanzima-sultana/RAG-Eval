from common.args import EvalQnAArgs

def validate_eval_qna_args(eval_qna_args: EvalQnAArgs):
	"""
	Input: eval_qna_args, the eval QnA settings.
	Output: nothing. Raises ValueError for the first check that fails.
	What it does: checks eval_qna_model is not empty, and n_sample_documents and n_sample_evals are larger than 0.
	The QnA LLMs are checked by llm.validate.validate_llm.
	"""

	# 1. Check eval_qna_model
	if eval_qna_args.eval_qna_model.strip() == "":
		raise ValueError(
			"eval_qna_model is empty. Provide a SentenceTransformer model name, e.g. all-MiniLM-L6-v2."
		)

	# 2. Check n_sample_documents
	if eval_qna_args.n_sample_documents <= 0:
		raise ValueError(
			f"n_sample_documents ({eval_qna_args.n_sample_documents}) must be larger than 0"
		)

	# 3. Check n_sample_evals
	if eval_qna_args.n_sample_evals <= 0:
		raise ValueError(
			f"n_sample_evals ({eval_qna_args.n_sample_evals}) must be larger than 0"
		)
