import torch
from sentence_transformers import SentenceTransformer

from common import paths
from common.args import EvalQnAArgs
from common.args import QueryArgs
from llm.schema import LLMArgs
from common.constants import I_DONT_KNOW
from stages.question.query.schema import Ans
from stages.question.query.schema import Qus
from stages.question.query.schema import Query
from stages.question.query.prompt import REWRITE_PROMPT
from stages.question.query.prompt import MULTI_QUERY_PROMPT
from stages.question.query.prompt import HYDE_PROMPT
from stages.question.query import cache
from stages.question.query import validate
from llm.llm import get_llm
from llm.validate import validate_llm
from stages.question.qna.schema import EvalQnA

def generate_queries(eval_qna: list[EvalQnA], query_args: QueryArgs, query_llm_args: LLMArgs) -> list[Query]:
	"""
	Input: eval_qna, the eval questions; query_args, the query settings; query_llm_args, the query LLM settings.
	Output: one Query per eval question, in the same order.
	What it does: asks the query LLM for a rewrite, n_multi_qus phrasings and a HyDE answer of each question,
	splits every question into BM25 tokens, embeds every qus and ans with the build's embedding model and prints the LLM cost.
	When the LLM does not return exactly n_multi_qus phrasings, multi_qus is empty; when it does not know the HyDE answer,
	hyde_ans is I_DONT_KNOW. Both are counted and printed, and the run goes on.
	"""
	# 1. Get the query LLM. It was checked by validate_llm in get_queries
	query_llm = get_llm(query_llm_args)

	# 2. Transform each question with the LLM
	original_qus_list = []
	rewritten_qus_list = []
	multi_qus_list = []
	hyde_ans_list = []
	multi_dont_know_count = 0
	hyde_dont_know_count = 0
	for qna_index in range(len(eval_qna)):
		qna_item = eval_qna[qna_index]

		# 1. Rewritten qus
		rewritten_qus_str = query_llm.generate("", REWRITE_PROMPT.format(qus=qna_item.qus)).text.strip()

		# Multi qus
		# The phrasings come one per line, sometimes with blank lines between them
		multi_reply = query_llm.generate("", MULTI_QUERY_PROMPT.format(qus=qna_item.qus, n_multi_qus=query_args.n_multi_qus)).text
		multi_qus_str_list = []
		for reply_line in multi_reply.split("\n"):
			stripped_line = reply_line.strip()
			if stripped_line != "":
				multi_qus_str_list.append(stripped_line)
		# Any other number of lines means the reply is not just the phrasings, e.g. it says it found nothing.
		# Treated as I don't know: no phrasings are kept for this question
		if len(multi_qus_str_list) != query_args.n_multi_qus:
			print(
				f"Multi-query I don't know for {qna_item.id}: {len(multi_qus_str_list)} lines instead of "
				f"{query_args.n_multi_qus}. Reply: {multi_reply!r}"
			)
			multi_qus_str_list = []
			multi_dont_know_count = multi_dont_know_count + 1

		# 3. Hypithetical ans
		hyde_ans_str = query_llm.generate("", HYDE_PROMPT.format(qus=qna_item.qus)).text.strip()
		# The prompt asks for a passage even when unsure, but if the LLM still refuses, the reply is stored as exactly
		# I_DONT_KNOW, so retrieval can skip it. The reply may add a full stop or a reason after it
		hyde_ans_lower = hyde_ans_str.lower().replace("\u2019", "'")
		if hyde_ans_lower.startswith("i don't know") or hyde_ans_lower.startswith("i do not know"):
			hyde_ans_str = I_DONT_KNOW
			hyde_dont_know_count = hyde_dont_know_count + 1

		original_qus_list.append(qna_item.qus)
		rewritten_qus_list.append(rewritten_qus_str)
		multi_qus_list.append(multi_qus_str_list)
		hyde_ans_list.append(hyde_ans_str)

		if (qna_index + 1) % 20 == 0 or qna_index + 1 == len(eval_qna):
			print(f"Questions transformed : {qna_index + 1} of {len(eval_qna)}")

	print(f"Multi-query I don't know : {multi_dont_know_count} of {len(eval_qna)}")
	print(f"HyDE I don't know : {hyde_dont_know_count} of {len(eval_qna)}")
	query_llm.compute_cost()

	# 3. Embed every qus and ans with the build's embedding model, normalized to length 1 like the chunk embeddings
	# Multi-query phrasings are embedded as one flat list: n_multi_qus rows per question, none for an I don't know
	flat_multi_qus_list = []
	for multi_qus_str_list in multi_qus_list:
		for multi_qus_str in multi_qus_str_list:
			flat_multi_qus_list.append(multi_qus_str)

	device = "cuda" if torch.cuda.is_available() else "cpu"
	embedding_model = SentenceTransformer(query_args.embedding_model, device=device)
	original_qus_vectors = embedding_model.encode(original_qus_list, batch_size=256, normalize_embeddings=True)
	rewritten_qus_vectors = embedding_model.encode(rewritten_qus_list, batch_size=256, normalize_embeddings=True)
	flat_multi_qus_vectors = embedding_model.encode(flat_multi_qus_list, batch_size=256, normalize_embeddings=True)
	hyde_ans_vectors = embedding_model.encode(hyde_ans_list, batch_size=256, normalize_embeddings=True)

	# 4. Build one Query per question. Tokens are split the same way as the BM25 chunks
	queries = []
	# Position of the question's first phrasing in flat_multi_qus_vectors
	flat_multi_position = 0
	for qna_index in range(len(eval_qna)):
		original_qus = Qus(
			qus=original_qus_list[qna_index],
			tokens=original_qus_list[qna_index].lower().split(),
			embeddings=original_qus_vectors[qna_index]
		)
		rewritten_qus = Qus(
			qus=rewritten_qus_list[qna_index],
			tokens=rewritten_qus_list[qna_index].lower().split(),
			embeddings=rewritten_qus_vectors[qna_index]
		)
		multi_qus = []
		for multi_qus_str in multi_qus_list[qna_index]:
			multi_qus.append(Qus(
				qus=multi_qus_str,
				tokens=multi_qus_str.lower().split(),
				embeddings=flat_multi_qus_vectors[flat_multi_position]
			))
			flat_multi_position = flat_multi_position + 1
		hyde_ans = Ans(
			ans=hyde_ans_list[qna_index],
			embeddings=hyde_ans_vectors[qna_index]
		)

		queries.append(Query(
			id=eval_qna[qna_index].id,
			original_qus=original_qus,
			rewritten_qus=rewritten_qus,
			multi_qus=multi_qus,
			hyde_ans=hyde_ans,
			normalized=True,
			embedding_model=query_args.embedding_model,
			llm=query_llm_args.llm,
			llm_model=query_llm_args.llm_model
		))

	return queries

def get_queries(output_path: str, eval_qna: list[EvalQnA], eval_qna_args: EvalQnAArgs, query_args: QueryArgs, query_llm_args: LLMArgs) -> list[Query]:
	"""
	Input: output_path, the dataset's output folder; eval_qna, the eval questions; eval_qna_args, the settings
	that identify the eval set; query_args, the embedding model of the build and the number of multi-query phrasings;
	query_llm_args, the query LLM settings.
	Output: one Query per eval question.
	What it does: validates query_args, then returns the cached queries if they exist,
	otherwise validates the query LLM, generates the queries, saves them and returns them.
	"""
	# 1. Validate the query settings
	validate.validate_query_args(query_args)

	# 2. Load cached queries
	# The eval QnA records hold the model that wrote them
	query_path = paths.get_query_path(output_path, eval_qna_args, eval_qna[0].llm_model, query_args, query_llm_args.llm_model)
	queries = cache.load_cached_queries(query_path)
	if queries is not None:
		print(f"Queries loaded from cache : {query_path}")
		return queries

	# 3. Validate the LLM only when it is needed, so a cached run needs no API keys, ALLOW_PAID_API or Ollama server
	validate_llm(query_llm_args)

	# 4. Generate and save the queries
	queries = generate_queries(eval_qna, query_args, query_llm_args)
	cache.save_queries(query_path, queries)
	print(f"Queries saved : {query_path}")
	return queries
