from common.args import EvalQnAArgs
from common.constants import CLAUDE_LLM_GENERATOR
from common.constants import CLAUDE_TRANSFORMS_MODEL
from common.constants import CLAUDE_QUESTION_MODEL
from common.constants import CLAUDE_PRICES_PER_MILLION_TOKENS

def validate_eval_qna_args(eval_qna_args: EvalQnAArgs):
	"""
	Input: eval_qna_args, the eval QnA settings.
	Output: nothing. Raises ValueError for the first check that fails.
	What it does: checks eval_qna_llm is supported and its models have prices, eval_qna_model is not empty, and
	n_sample_documents and n_sample_evals are larger than 0.
	"""

	# 1. Check eval_qna_llm
	if eval_qna_args.eval_qna_llm != CLAUDE_LLM_GENERATOR:
		raise ValueError(
			f"eval_qna_llm ({eval_qna_args.eval_qna_llm}) is not supported. Use \"{CLAUDE_LLM_GENERATOR}\"."
		)

	# 2. Check the Claude models have prices, so the cost can be computed after generation
	for model_name in (CLAUDE_TRANSFORMS_MODEL, CLAUDE_QUESTION_MODEL):
		if model_name not in CLAUDE_PRICES_PER_MILLION_TOKENS:
			raise ValueError(
				f"{model_name} has no price in CLAUDE_PRICES_PER_MILLION_TOKENS. Add its price to common/constants.py."
			)

	# 3. Check eval_qna_model
	if eval_qna_args.eval_qna_model.strip() == "":
		raise ValueError(
			"eval_qna_model is empty. Provide a SentenceTransformer model name, e.g. all-MiniLM-L6-v2."
		)

	# 4. Check n_sample_documents
	if eval_qna_args.n_sample_documents <= 0:
		raise ValueError(
			f"n_sample_documents ({eval_qna_args.n_sample_documents}) must be larger than 0"
		)

	# 5. Check n_sample_evals
	if eval_qna_args.n_sample_evals <= 0:
		raise ValueError(
			f"n_sample_evals ({eval_qna_args.n_sample_evals}) must be larger than 0"
		)
