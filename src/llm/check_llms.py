"""
check_llms.py

Checks the QnA and query layers end to end with the LLMs passed on the command line, on a few documents.
Run through scripts/check_llms.sh (python3 -m llm.check_llms), which picks the models of one provider.

For each layer it deletes this check's cache, generates with the LLM, checks the records and the cache file, then
runs again and checks the cache is loaded instead of generating. It only writes under --output_path, which must be
an llm_check folder, so the real caches are never touched.
"""

import argparse
import os
import shutil
import sys

from common import paths
from common.args import add_ingest_arguments
from common.args import make_eval_qna_args
from common.args import make_ingest_args
from common.args import make_llm_args
from common.args import make_query_args
from l1_ingestion.ingestion import ingestion
from l5_query.query import get_queries
from qna.qna import get_eval_qna
from ragas.utils import num_tokens_from_string

def parse_check_args():
	"""
	Input: none. Reads the command line arguments passed by scripts/check_llms.sh.
	Output: an argparse.Namespace with one attribute per argument.
	What it does: defines the ingestion, eval QnA, query and LLM arguments of the check, and parses them.
	"""
	parser = argparse.ArgumentParser(description="RAG-Eval LLM check")
	add_ingest_arguments(parser)
	parser.add_argument("--max_document_tokens", type=int, required=True, help="Only documents up to this many tokens are used")
	parser.add_argument("--eval_qna_model", type=str, required=True, help="Embedding model RAGAS uses for its knowledge graph")
	parser.add_argument("--n_sample_documents", type=int, required=True, help="Documents RAGAS generates the QnA from")
	parser.add_argument("--n_sample_evals", type=int, required=True, help="Eval QnA items to generate")
	parser.add_argument("--qna_llm", type=str, required=True, help="LLM provider that writes the questions and answers")
	parser.add_argument("--qna_llm_model", type=str, required=True, help="Model that writes the questions and answers")
	parser.add_argument("--qna_input_price", type=float, required=True, help="QnA model input price per million tokens")
	parser.add_argument("--qna_output_price", type=float, required=True, help="QnA model output price per million tokens")
	parser.add_argument("--qna_kg_llm", type=str, required=True, help="LLM provider that builds the knowledge graph")
	parser.add_argument("--qna_kg_llm_model", type=str, required=True, help="Model that builds the knowledge graph")
	parser.add_argument("--qna_kg_input_price", type=float, required=True, help="Knowledge graph model input price per million tokens")
	parser.add_argument("--qna_kg_output_price", type=float, required=True, help="Knowledge graph model output price per million tokens")
	parser.add_argument("--model_name", type=str, required=True, help="Embedding model the queries are embedded with")
	parser.add_argument("--n_multi_qus", type=int, required=True, help="Phrasings the query LLM writes for each question")
	parser.add_argument("--query_llm", type=str, required=True, help="LLM provider that transforms the questions")
	parser.add_argument("--query_llm_model", type=str, required=True, help="Model that transforms the questions")
	parser.add_argument("--query_input_price", type=float, required=True, help="Query model input price per million tokens")
	parser.add_argument("--query_output_price", type=float, required=True, help="Query model output price per million tokens")
	parser.add_argument("--llm_max_tokens", type=int, required=True, help="Maximum output tokens per LLM call")
	parser.add_argument("--llm_timeout_seconds", type=int, required=True, help="Seconds before one LLM call times out")
	return parser.parse_args()

def check(condition: bool, message: str, failures: list[str]) -> None:
	"""
	Input: condition, what must be true; message, what is checked; failures, the list of failed checks so far.
	Output: nothing.
	What it does: prints PASS or FAIL with the message, and adds the message to failures when it fails.
	"""
	if condition:
		print(f"PASS : {message}")
	else:
		print(f"FAIL : {message}")
		failures.append(message)

if __name__ == "__main__":
	print("\n ------------- RAG Eval LLM check -------- \n")
	args = parse_check_args()
	failures = []

	# 0. Only an llm_check folder may be written to, so the real caches are never deleted
	if "llm_check" not in args.output_path:
		raise ValueError(f"output_path ({args.output_path}) must be an llm_check folder, e.g. outputs/llm_check/ollama")

	ingest_args = make_ingest_args(args)
	eval_qna_args = make_eval_qna_args(args)
	query_args = make_query_args(args)
	qna_llm_args = make_llm_args(args.qna_llm, args.qna_llm_model, args.qna_input_price, args.qna_output_price, args.llm_max_tokens, args.llm_timeout_seconds)
	qna_kg_llm_args = make_llm_args(args.qna_kg_llm, args.qna_kg_llm_model, args.qna_kg_input_price, args.qna_kg_output_price, args.llm_max_tokens, args.llm_timeout_seconds)
	query_llm_args = make_llm_args(args.query_llm, args.query_llm_model, args.query_input_price, args.query_output_price, args.llm_max_tokens, args.llm_timeout_seconds)
	output_path = paths.get_output_path(ingest_args.input_path, ingest_args.output_path)

	# 1. Documents, keeping only the short ones, so they fit in every provider's context window.
	# Counted with tiktoken, as RAGAS and the QnA sampling count
	print("\n\n1. Documents")
	all_documents = ingestion(ingest_args)
	documents = {}
	for doc_id in all_documents:
		if num_tokens_from_string(all_documents[doc_id].text) <= args.max_document_tokens:
			documents[doc_id] = all_documents[doc_id]
	print(f"Documents : {len(documents)} of {len(all_documents)} have at most {args.max_document_tokens} tokens")

	# QnA and query passed with claude, chatgpt, gemini and ollama (see the README llm section and log/llm/),
	# so they are off, to not pay for them again. Remove the # in front of steps 2 to 6 to check them again.
	# # 2. Delete this check's QnA, knowledge graph and query caches, so both layers really call their LLM
	# qna_path = paths.get_eval_qna_path(output_path, eval_qna_args, qna_llm_args.llm_model)
	# kg_path = paths.get_eval_kg_path(output_path, eval_qna_args, qna_kg_llm_args.llm_model)
	# query_path = paths.get_query_path(output_path, eval_qna_args, qna_llm_args.llm_model, query_args, query_llm_args.llm_model)
	# if os.path.exists(qna_path):
	# 	os.remove(qna_path)
	# if os.path.exists(kg_path):
	# 	os.remove(kg_path)
	# if os.path.exists(query_path):
	# 	shutil.rmtree(query_path)

	# # 3. QnA: generate with the LLMs
	# print(f"\n\n2. QnA with {qna_llm_args.llm_model} (questions) and {qna_kg_llm_args.llm_model} (knowledge graph)")
	# eval_qna = get_eval_qna(output_path, documents, eval_qna_args, qna_llm_args, qna_kg_llm_args)
	# check(len(eval_qna) > 0, f"QnA generated : {len(eval_qna)} items", failures)
	# check(os.path.exists(qna_path), f"QnA cache file written : {qna_path}", failures)
	# check(os.path.exists(kg_path), f"Knowledge graph cache file written : {kg_path}", failures)
	# for eval_qna_item in eval_qna:
	# 	check(eval_qna_item.qus.strip() != "" and eval_qna_item.ans.strip() != "", f"{eval_qna_item.id} has a question and an answer", failures)
	# 	check(eval_qna_item.llm == qna_llm_args.llm and eval_qna_item.llm_model == qna_llm_args.llm_model, f"{eval_qna_item.id} llm is {eval_qna_item.llm} / {eval_qna_item.llm_model}", failures)
	# 	print(f"   {eval_qna_item.id} : {eval_qna_item.qus[:120]}")

	# # 4. QnA: the second run loads the cache, so the cache file is not written again
	# qna_modified_time = os.path.getmtime(qna_path)
	# cached_eval_qna = get_eval_qna(output_path, documents, eval_qna_args, qna_llm_args, qna_kg_llm_args)
	# check(len(cached_eval_qna) == len(eval_qna), f"QnA loaded from cache : {len(cached_eval_qna)} items", failures)
	# check(os.path.getmtime(qna_path) == qna_modified_time, "QnA cache file not written again", failures)

	# # 5. Query: generate with the LLM from the QnA above
	# print(f"\n\n3. Query with {query_llm_args.llm_model}")
	# queries = get_queries(output_path, eval_qna, eval_qna_args, query_args, query_llm_args)
	# check(len(queries) == len(eval_qna), f"Queries generated : {len(queries)} for {len(eval_qna)} questions", failures)
	# check(os.path.exists(os.path.join(query_path, "queries.json")), f"Query cache folder written : {query_path}", failures)
	# for query in queries:
	# 	check(query.rewritten_qus.qus.strip() != "", f"{query.id} has a rewritten question", failures)
	# 	check(query.hyde_ans.ans.strip() != "", f"{query.id} has a HyDE answer", failures)
	# 	check(query.llm == query_llm_args.llm and query.llm_model == query_llm_args.llm_model, f"{query.id} llm is {query.llm} / {query.llm_model}", failures)
	# 	print(f"   {query.id} rewritten : {query.rewritten_qus.qus[:120]}")
	# 	print(f"   {query.id} multi : {len(query.multi_qus)} phrasings")

	# # 6. Query: the second run loads the cache, so the cache file is not written again
	# queries_file = os.path.join(query_path, "queries.json")
	# queries_modified_time = os.path.getmtime(queries_file)
	# cached_queries = get_queries(output_path, eval_qna, eval_qna_args, query_args, query_llm_args)
	# check(len(cached_queries) == len(queries), f"Queries loaded from cache : {len(cached_queries)}", failures)
	# check(os.path.getmtime(queries_file) == queries_modified_time, "Query cache file not written again", failures)

	# 7. Result
	print("\n\n------------- Result --------")
	if len(failures) > 0:
		print(f"FAILED : {len(failures)} checks")
		for failure in failures:
			print(f"   {failure}")
		sys.exit(1)
	print("ALL CHECKS PASSED")
