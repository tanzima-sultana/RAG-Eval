from common.args import QueryArgs
from common.constants import CLAUDE_LLM_GENERATOR
from common.constants import OLLAMA_LLM

def validate_query_args(query_args: QueryArgs):
	"""
	Input: query_args, the query settings.
	Output: nothing. Raises ValueError for the first check that fails.
	What it does: checks embedding_model and query_llm_model are not empty, query_llm is supported, and
	n_multi_qus is at least 1. Whether the model can run is checked later by llm.get_llm().
	"""

	# 1. Check embedding_model
	if query_args.embedding_model.strip() == "":
		raise ValueError(
			"embedding_model is empty. Pass --model_name with the SentenceTransformer model the build used, e.g. all-MiniLM-L6-v2."
		)

	# 2. Check query_llm
	if query_args.query_llm != CLAUDE_LLM_GENERATOR and query_args.query_llm != OLLAMA_LLM:
		raise ValueError(
			f"query_llm ({query_args.query_llm}) is not supported. Use \"{OLLAMA_LLM}\" or \"{CLAUDE_LLM_GENERATOR}\"."
		)

	# 3. Check query_llm_model
	if query_args.query_llm_model.strip() == "":
		raise ValueError(
			"query_llm_model is empty. Provide a model name, e.g. llama3.1:8b."
		)

	# 4. Check n_multi_qus
	if query_args.n_multi_qus < 1:
		raise ValueError(
			f"n_multi_qus ({query_args.n_multi_qus}) must be at least 1"
		)
