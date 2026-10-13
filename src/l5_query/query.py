import torch
from sentence_transformers import SentenceTransformer

from common import paths
from common.args import EvalQnAArgs
from common.args import QueryArgs
from l5_query.schema import Ans
from l5_query.schema import Qus
from l5_query.schema import Query
from l5_query.prompt import REWRITE_PROMPT
from l5_query.prompt import MULTI_QUERY_PROMPT
from l5_query.prompt import HYDE_PROMPT
from l5_query import cache
from l5_query import validate
from llm.llm import get_llm
from qna.schema import EvalQnA

def generate_queries(eval_qna: list[EvalQnA], query_args: QueryArgs) -> list[Query]:
	"""
	Input: eval_qna, the eval questions; query_args, the query settings.
	Output: one Query per eval question, in the same order.
	What it does: asks the query LLM for a rewrite, n_multi_qus phrasings and a HyDE answer of each question,
	splits every question into BM25 tokens, embeds every qus and ans with the build's embedding model and prints the LLM cost.
	Raises ValueError if the LLM does not return exactly n_multi_qus phrasings for a question.
	"""
	# 1. Get the query LLM. Checks the provider is usable, e.g. the Ollama server is running and the model is pulled
	query_llm = get_llm(query_args.query_llm, query_args.query_llm_model)

	# 2. Transform each question with the LLM
	original_qus_list = []
	rewritten_qus_list = []
	multi_qus_list = []
	hyde_ans_list = []
	for qna_index in range(len(eval_qna)):
		qna_item = eval_qna[qna_index]

		# 1. Rewritten qus
		rewritten_qus_str = query_llm.generate(REWRITE_PROMPT.format(qus=qna_item.qus)).strip()

		# Multi qus
		# The phrasings come one per line, sometimes with blank lines between them
		multi_reply = query_llm.generate(MULTI_QUERY_PROMPT.format(qus=qna_item.qus, n_multi_qus=query_args.n_multi_qus))
		multi_qus_str_list = []
		for reply_line in multi_reply.split("\n"):
			stripped_line = reply_line.strip()
			if stripped_line != "":
				multi_qus_str_list.append(stripped_line)
		if len(multi_qus_str_list) != query_args.n_multi_qus:
			raise ValueError(
				f"The query LLM returned {len(multi_qus_str_list)} phrasings instead of {query_args.n_multi_qus} "
				f"for {qna_item.id}. Its reply was:\n{multi_reply}"
			)

		# 3. Hypithetical ans
		hyde_ans_str = query_llm.generate(HYDE_PROMPT.format(qus=qna_item.qus)).strip()

		original_qus_list.append(qna_item.qus)
		rewritten_qus_list.append(rewritten_qus_str)
		multi_qus_list.append(multi_qus_str_list)
		hyde_ans_list.append(hyde_ans_str)

		if (qna_index + 1) % 20 == 0 or qna_index + 1 == len(eval_qna):
			print(f"Questions transformed : {qna_index + 1} of {len(eval_qna)}")

	query_llm.compute_cost()

	# 3. Embed every qus and ans with the build's embedding model, normalized to length 1 like the l3 chunk embeddings
	# Multi-query phrasings are embedded as one flat list, n_multi_qus rows per question
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

	# 4. Build one Query per question. Tokens are split the same way as the l4 BM25 chunks
	queries = []
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
		for multi_index in range(query_args.n_multi_qus):
			multi_qus_str = multi_qus_list[qna_index][multi_index]
			multi_qus.append(Qus(
				qus=multi_qus_str,
				tokens=multi_qus_str.lower().split(),
				embeddings=flat_multi_qus_vectors[qna_index * query_args.n_multi_qus + multi_index]
			))
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
			llm_model=query_args.query_llm_model
		))

	return queries

def get_queries(output_path: str, eval_qna: list[EvalQnA], eval_qna_args: EvalQnAArgs, query_args: QueryArgs) -> list[Query]:
	"""
	Input: output_path, the dataset's output folder; eval_qna, the eval questions; eval_qna_args, the settings
	that identify the eval set; query_args, the embedding model of the build, the query LLM provider,
	its model and the number of multi-query phrasings.
	Output: one Query per eval question.
	What it does: validates query_args, then returns the cached queries if they exist,
	otherwise generates them, saves them and returns them.
	"""
	# 1. Validate the query settings
	validate.validate_query_args(query_args)

	# 2. Load cached queries, or generate and save them
	query_path = paths.get_query_path(output_path, eval_qna_args, query_args)
	queries = cache.load_cached_queries(query_path)
	if queries is not None:
		print(f"Queries loaded from cache : {query_path}")
		return queries

	queries = generate_queries(eval_qna, query_args)
	cache.save_queries(query_path, queries)
	print(f"Queries saved : {query_path}")
	return queries
