from common import constants
from common.args import QueryArgs
from llm_provider.llm_provider import LLMProvider
from rag.serving.query.query_prompts import HYDE_PROMPT
from rag.serving.query.query_prompts import MULTI_QUERY_PROMPT
from rag.serving.query.query_prompts import REWRITE_PROMPT
from schema.query import Query


def multi_phrasings(question: str, n_multi_qus: int, query_llm_provider: LLMProvider) -> list[str]:
	"""
	Input: question, one question; n_multi_qus, the number of phrasings to ask for; query_llm_provider, the query LLM.
	Output: n_multi_qus phrasings of the question, or [] if the LLM did not return exactly n_multi_qus lines.
	What it does: asks the query LLM for the phrasings and splits its reply into lines.
	"""
	# The phrasings come one per line, sometimes with blank lines between them
	multi_reply = query_llm_provider.generate("", MULTI_QUERY_PROMPT.format(qus=question, n_multi_qus=n_multi_qus)).text
	phrasings = []
	for reply_line in multi_reply.split("\n"):
		stripped_line = reply_line.strip()
		if stripped_line != "":
			phrasings.append(stripped_line)

	# Any other number of lines means the reply is not just the phrasings, so no phrasings are kept
	if len(phrasings) != n_multi_qus:
		print(
			f"Multi-query I don't know for {question!r}: {len(phrasings)} lines instead of {n_multi_qus}. "
			f"Reply: {multi_reply!r}"
		)
		phrasings = []
	return phrasings


def hyde_answer(question: str, query_llm_provider: LLMProvider) -> str:
	"""
	Input: question, one question; query_llm_provider, the query LLM.
	Output: a hypothetical answer passage, or exactly I_DONT_KNOW if the LLM refused.
	What it does: asks the query LLM for the passage. A refusal is stored as I_DONT_KNOW, so retrieval can skip it.
	"""
	hyde_ans_str = query_llm_provider.generate("", HYDE_PROMPT.format(qus=question)).text.strip()
	hyde_ans_lower = hyde_ans_str.lower().replace("’", "'")
	if hyde_ans_lower.startswith("i don't know") or hyde_ans_lower.startswith("i do not know"):
		hyde_ans_str = constants.I_DONT_KNOW
	return hyde_ans_str


def make_original_query(query_args: QueryArgs, question_id: str, question: str) -> Query:
	"""
	Input: query_args, the query settings; question_id; question, one question.
	Output: the original Query of the question. Its transformed query is the question as it is.
	What it does: keeps the question as it is. It calls no LLM.
	"""
	query_llm, query_llm_model = query_args.query_llm_args.llm_model.split("/", 1)
	query = Query(
		query_type=constants.ORIGINAL_QUERY,
		llm=query_llm,
		llm_model=query_llm_model,
		question_id=question_id,
		original_question=question,
		transformed_query=question
	)
	return query


def make_rewritten_query(query_args: QueryArgs, question_id: str, question: str, query_llm_provider: LLMProvider) -> Query:
	"""
	Input: query_args, the query settings; question_id; question, one question; query_llm_provider, the query LLM.
	Output: the rewritten Query of the question. Its transformed query is the rewritten text.
	What it does: asks the query LLM to rewrite the question.
	"""
	rewritten_question = query_llm_provider.generate("", REWRITE_PROMPT.format(qus=question)).text.strip()

	query_llm, query_llm_model = query_args.query_llm_args.llm_model.split("/", 1)
	query = Query(
		query_type=constants.REWRITTEN_QUERY,
		llm=query_llm,
		llm_model=query_llm_model,
		question_id=question_id,
		original_question=question,
		transformed_query=rewritten_question
	)
	return query


def make_multi_queries(query_args: QueryArgs, question_id: str, question: str, query_llm_provider: LLMProvider) -> list[Query]:
	"""
	Input: query_args, the query settings; question_id; question, one question; query_llm_provider, the query LLM.
	Output: one multi Query per phrasing, each with one phrasing as its transformed query. The list is empty if
	the LLM did not return exactly n_multi_qus phrasings.
	What it does: asks the query LLM for n_multi_qus phrasings of the question.
	"""
	phrasings = multi_phrasings(question, query_args.n_multi_qus, query_llm_provider)

	query_llm, query_llm_model = query_args.query_llm_args.llm_model.split("/", 1)
	queries = []
	for phrasing in phrasings:
		query = Query(
			query_type=constants.MULTI_QUERY,
			llm=query_llm,
			llm_model=query_llm_model,
			question_id=question_id,
			original_question=question,
			transformed_query=phrasing
		)
		queries.append(query)
	return queries


def make_hyde_query(query_args: QueryArgs, question_id: str, question: str, query_llm_provider: LLMProvider) -> Query:
	"""
	Input: query_args, the query settings; question_id; question, one question; query_llm_provider, the query LLM.
	Output: the HyDE Query of the question. Its transformed query is the hypothetical answer passage, or I_DONT_KNOW
	if the LLM refused.
	What it does: asks the query LLM for a hypothetical answer passage.
	"""
	hypothetical_answer_text = hyde_answer(question, query_llm_provider)

	query_llm, query_llm_model = query_args.query_llm_args.llm_model.split("/", 1)
	query = Query(
		query_type=constants.HYDE_QUERY,
		llm=query_llm,
		llm_model=query_llm_model,
		question_id=question_id,
		original_question=question,
		transformed_query=hypothetical_answer_text
	)
	return query
