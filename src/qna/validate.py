from common.args import EvalQnAArgs
from common.constants import CLAUDE_PRICES_PER_MILLION_TOKENS

def validate_eval_qna_args(eval_qna_args: EvalQnAArgs):
	"""
	Input: eval_qna_args, the eval QnA settings.
	Output: nothing. Raises ValueError for the first check that fails.
	What it does: checks eval_llm has a price, eval_model is not empty, and
	n_sample_documents and n_sample_evals are larger than 0.
	"""

	# 1. Check eval_llm
	if eval_qna_args.eval_llm not in CLAUDE_PRICES_PER_MILLION_TOKENS:
		supported_models = []
		for model_name in CLAUDE_PRICES_PER_MILLION_TOKENS:
			supported_models.append(model_name)
		raise ValueError(
			f"eval_llm ({eval_qna_args.eval_llm}) has no price in CLAUDE_PRICES_PER_MILLION_TOKENS. "
			f"Use one of {supported_models} or add its price to common/constants.py."
		)

	# 2. Check eval_model
	if eval_qna_args.eval_model.strip() == "":
		raise ValueError(
			"eval_model is empty. Provide a SentenceTransformer model name, e.g. all-MiniLM-L6-v2."
		)

	# 3. Check n_sample_documents
	if eval_qna_args.n_sample_documents <= 0:
		raise ValueError(
			f"n_sample_documents ({eval_qna_args.n_sample_documents}) must be larger than 0"
		)

	# 4. Check n_sample_evals
	if eval_qna_args.n_sample_evals <= 0:
		raise ValueError(
			f"n_sample_evals ({eval_qna_args.n_sample_evals}) must be larger than 0"
		)
