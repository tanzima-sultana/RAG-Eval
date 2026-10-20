import time

from common import paths
from common.args import parse_eval_args
from common.args import make_ingest_args
from common.args import make_eval_qna_args
from common.args import make_query_args
from common.args import make_llm_args
from stages.build.ingestion.ingestion import ingestion
from stages.question.qna.qna import get_eval_qna
from stages.question.query.query import get_queries

def run_question(args):
	"""
	Input: args, the parsed eval arguments, holding the build settings and the eval QnA and query settings.
	Output: the ingested documents, the eval QnA and the queries.
	What it does: loads the documents, then loads or generates the eval QnA and the queries made from them.
	"""
	ingest_args = make_ingest_args(args)
	output_path = paths.get_output_path(ingest_args.input_path, ingest_args.output_path)
	eval_qna_args = make_eval_qna_args(args)
	qna_llm_args = make_llm_args(args.qna_llm, args.qna_llm_model, args.qna_input_price, args.qna_output_price, args.qna_cache_write_price, args.qna_cache_read_price, args.llm_max_tokens, args.llm_timeout_seconds, args.llm_cache_ttl)
	qna_kg_llm_args = make_llm_args(args.qna_kg_llm, args.qna_kg_llm_model, args.qna_kg_input_price, args.qna_kg_output_price, args.qna_kg_cache_write_price, args.qna_kg_cache_read_price, args.llm_max_tokens, args.llm_timeout_seconds, args.llm_cache_ttl)
	query_args = make_query_args(args)
	query_llm_args = make_llm_args(args.query_llm, args.query_llm_model, args.query_input_price, args.query_output_price, args.query_cache_write_price, args.query_cache_read_price, args.llm_max_tokens, args.llm_timeout_seconds, args.llm_cache_ttl)

	# 1. Load the ingested documents
	# The same ingestion call as build.py; it returns the documents from the ingestion cache
	print("\n\nLoading documents")
	s1 = time.time()
	documents = ingestion(ingest_args)
	e1 = time.time() - s1
	print("\nLoading documents time : ", e1)

	# 2. Load eval QnA
	# Reuses {output_path}/eval_qna_{n_sample_documents}_{n_sample_evals}_{qna_llm_model}_{eval_qna_model}.json if it exists
	print("\n\nLoading eval QnA")
	s2 = time.time()
	eval_qna = get_eval_qna(output_path, documents, eval_qna_args, qna_llm_args, qna_kg_llm_args)
	e2 = time.time() - s2
	print(f"Eval QnA loaded : {len(eval_qna)}")
	print("\nLoading eval QnA time : ", e2)

	# 3. Query
	print("\n\nLoading Query")
	s3 = time.time()
	queries = get_queries(output_path, eval_qna, eval_qna_args, query_args, query_llm_args)
	e3 = time.time() - s3
	print(f"Query loaded : {len(queries)}")
	print("\nLoading Query time : ", e3)

	return documents, eval_qna, queries

if __name__ == "__main__":
	print("\n ------------- RAG Question -------- \n")
	start = time.time()

	args = parse_eval_args()
	print(args)

	run_question(args)

	# ---------
	end = time.time()
	elapsed_time = end - start
	print("\n\nElapsed time : ", elapsed_time)
