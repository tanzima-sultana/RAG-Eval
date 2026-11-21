import numpy as np

from langchain_huggingface import HuggingFaceEmbeddings

from common import constants
from common.args import QueryArgs
from llm_provider.llm_provider import LLMProvider
from rag.serving.query_transformation.query_prompts import HYDE_PROMPT
from rag.serving.query_transformation.query_prompts import MULTI_QUERY_PROMPT
from rag.serving.query_transformation.query_prompts import REWRITE_PROMPT
from schema.query import Ans
from schema.query import Query
from schema.query import Qus


def multi_phrasings(qus: str, n_multi_qus: int, query_llm_provider: LLMProvider) -> list[str]:
	"""
	Input: qus, one question; n_multi_qus, the number of phrasings to ask for; query_llm_provider, the query LLM.
	Output: n_multi_qus phrasings of qus, or [] if the LLM did not return exactly n_multi_qus lines.
	What it does: asks the query LLM for the phrasings and splits its reply into lines.
	"""
	# The phrasings come one per line, sometimes with blank lines between them
	multi_reply = query_llm_provider.generate("", MULTI_QUERY_PROMPT.format(qus=qus, n_multi_qus=n_multi_qus)).text
	phrasings = []
	for reply_line in multi_reply.split("\n"):
		stripped_line = reply_line.strip()
		if stripped_line != "":
			phrasings.append(stripped_line)

	# Any other number of lines means the reply is not just the phrasings, so no phrasings are kept
	if len(phrasings) != n_multi_qus:
		print(
			f"Multi-query I don't know for {qus!r}: {len(phrasings)} lines instead of {n_multi_qus}. "
			f"Reply: {multi_reply!r}"
		)
		phrasings = []
	return phrasings


def hyde_answer(qus: str, query_llm_provider: LLMProvider) -> str:
	"""
	Input: qus, one question; query_llm_provider, the query LLM.
	Output: a hypothetical answer passage, or exactly I_DONT_KNOW if the LLM refused.
	What it does: asks the query LLM for the passage. A refusal is stored as I_DONT_KNOW, so retrieval can skip it.
	"""
	hyde_ans_str = query_llm_provider.generate("", HYDE_PROMPT.format(qus=qus)).text.strip()
	hyde_ans_lower = hyde_ans_str.lower().replace("’", "'")
	if hyde_ans_lower.startswith("i don't know") or hyde_ans_lower.startswith("i do not know"):
		hyde_ans_str = constants.I_DONT_KNOW
	return hyde_ans_str


def transform_single_query(query_args: QueryArgs, query_id: str, qus: str, query_types: list[str], query_llm_provider: LLMProvider | None, embedding_model: HuggingFaceEmbeddings) -> Query:
	"""
	Input: query_args, the query settings; query_id; qus, one question; query_types, the query types to make;
	query_llm_provider, the query LLM, None if query_types is only original; embedding_model, normalized like the
	chunk embeddings.
	Output: the Query of qus, holding only query_types. The other query types are None.
	What it does: makes each query type of one question and embeds it. Tokens are split the same way as the BM25
	chunks. Nothing is saved.
	"""
	# 1. Original: the question as it is
	original_qus = None
	if constants.ORIGINAL_QUERY in query_types:
		original_vector = embedding_model.embed_documents([qus])[0]
		original_qus = Qus(qus=qus, tokens=qus.lower().split(), embeddings=np.array(original_vector, dtype=np.float32))

	# 2. Rewritten: the question rewritten by the query LLM
	rewritten_qus = None
	if constants.REWRITTEN_QUERY in query_types:
		rewritten_qus_str = query_llm_provider.generate("", REWRITE_PROMPT.format(qus=qus)).text.strip()
		rewritten_vector = embedding_model.embed_documents([rewritten_qus_str])[0]
		rewritten_qus = Qus(
			qus=rewritten_qus_str,
			tokens=rewritten_qus_str.lower().split(),
			embeddings=np.array(rewritten_vector, dtype=np.float32)
		)

	# 3. Multi: n_multi_qus phrasings, or none if the LLM did not return exactly n_multi_qus
	multi_qus = None
	if constants.MULTI_QUERY in query_types:
		phrasings = multi_phrasings(qus, query_args.n_multi_qus, query_llm_provider)
		multi_qus = []
		if len(phrasings) > 0:
			phrasing_vectors = embedding_model.embed_documents(phrasings)
			for phrasing_index in range(len(phrasings)):
				multi_qus.append(Qus(
					qus=phrasings[phrasing_index],
					tokens=phrasings[phrasing_index].lower().split(),
					embeddings=np.array(phrasing_vectors[phrasing_index], dtype=np.float32)
				))

	# 4. HyDE: a hypothetical answer passage, or I_DONT_KNOW
	hyde_ans = None
	if constants.HYDE_QUERY in query_types:
		hyde_ans_str = hyde_answer(qus, query_llm_provider)
		hyde_vector = embedding_model.embed_documents([hyde_ans_str])[0]
		hyde_ans = Ans(ans=hyde_ans_str, embeddings=np.array(hyde_vector, dtype=np.float32))

	query_llm, query_llm_model = query_args.query_llm_args.llm_model.split("/", 1)
	return Query(
		id=query_id,
		original_qus=original_qus,
		rewritten_qus=rewritten_qus,
		multi_qus=multi_qus,
		hyde_ans=hyde_ans,
		normalized=True,
		embedding_model=query_args.embedding_model,
		llm=query_llm,
		llm_model=query_llm_model
	)
