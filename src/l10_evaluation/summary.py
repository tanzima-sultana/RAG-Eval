"""
summary.py

Prints the retrieval evaluation and the answer evaluation of every retrieval setup as one table row, averaged over its
eval questions.
"""

from l6_retrieval.schema import Retrieved
from l10_evaluation.schema import AnswerEvaluation
from l10_evaluation.schema import RetrievalEvaluation

def print_retrieval_summary(retrieval_evaluations: list[tuple[Retrieved, RetrievalEvaluation]]) -> None:
	"""
	Input: retrieval_evaluations, one (Retrieved, RetrievalEvaluation) per eval question and retrieval setup.
	Output: nothing. Prints the table.
	What it does: groups the evaluations by setup (chunk type, retrieval type, index type, query type, reranking)
	and prints the mean of each metric per setup.
	"""
	# 1. Group the evaluations by setup
	evaluations_by_setup = {}
	for retrieved, retrieval_evaluation in retrieval_evaluations:
		setup = (retrieved.chunk_type, retrieved.retrieval_type, retrieved.index_type, retrieved.query_type, retrieved.reranking)
		if setup not in evaluations_by_setup:
			evaluations_by_setup[setup] = []
		evaluations_by_setup[setup].append(retrieval_evaluation)

	# 2. Print the header
	print("\n\nRetrieval evaluation per setup (mean over the eval questions)")
	header = (
		f"{'chunk':<9}{'retrieval':<10}{'index':<19}{'query':<10}{'rerank':<7}{'k':>3}{'n':>5}  "
		f"{'c_hit':>6}{'c_prec':>7}{'c_rec':>6}{'c_mrr':>6}{'c_ndcg':>7}  "
		f"{'d_hit':>6}{'d_prec':>7}{'d_rec':>6}{'d_mrr':>6}{'d_ndcg':>7}"
	)
	print(header)

	# 3. Print one row per setup, in setup order
	for setup in sorted(evaluations_by_setup):
		chunk_type, retrieval_type, index_type, query_type, reranking = setup
		retrieval_evaluation_list = evaluations_by_setup[setup]
		question_count = len(retrieval_evaluation_list)

		chunk_hit_sum = 0.0
		chunk_precision_sum = 0.0
		chunk_recall_sum = 0.0
		chunk_mrr_sum = 0.0
		chunk_ndcg_sum = 0.0
		doc_hit_sum = 0.0
		doc_precision_sum = 0.0
		doc_recall_sum = 0.0
		doc_mrr_sum = 0.0
		doc_ndcg_sum = 0.0
		for retrieval_evaluation in retrieval_evaluation_list:
			chunk_hit_sum = chunk_hit_sum + retrieval_evaluation.chunk_evaluation.hit
			chunk_precision_sum = chunk_precision_sum + retrieval_evaluation.chunk_evaluation.precision
			chunk_recall_sum = chunk_recall_sum + retrieval_evaluation.chunk_evaluation.recall
			chunk_mrr_sum = chunk_mrr_sum + retrieval_evaluation.chunk_evaluation.mrr
			chunk_ndcg_sum = chunk_ndcg_sum + retrieval_evaluation.chunk_evaluation.ndcg
			doc_hit_sum = doc_hit_sum + retrieval_evaluation.doc_evaluation.hit
			doc_precision_sum = doc_precision_sum + retrieval_evaluation.doc_evaluation.precision
			doc_recall_sum = doc_recall_sum + retrieval_evaluation.doc_evaluation.recall
			doc_mrr_sum = doc_mrr_sum + retrieval_evaluation.doc_evaluation.mrr
			doc_ndcg_sum = doc_ndcg_sum + retrieval_evaluation.doc_evaluation.ndcg

		# Every retrieval evaluation of one setup has the same k
		k = retrieval_evaluation_list[0].k
		if reranking:
			rerank_text = "yes"
		else:
			rerank_text = "no"

		row = (
			f"{chunk_type:<9}{retrieval_type:<10}{index_type:<19}{query_type:<10}{rerank_text:<7}{k:>3}{question_count:>5}  "
			f"{chunk_hit_sum / question_count:>6.3f}{chunk_precision_sum / question_count:>7.3f}"
			f"{chunk_recall_sum / question_count:>6.3f}{chunk_mrr_sum / question_count:>6.3f}"
			f"{chunk_ndcg_sum / question_count:>7.3f}  "
			f"{doc_hit_sum / question_count:>6.3f}{doc_precision_sum / question_count:>7.3f}"
			f"{doc_recall_sum / question_count:>6.3f}{doc_mrr_sum / question_count:>6.3f}"
			f"{doc_ndcg_sum / question_count:>7.3f}"
		)
		print(row)

def print_answer_summary(retrieved_list: list[Retrieved], answer_evaluations: list[AnswerEvaluation]) -> None:
	"""
	Input: retrieved_list and answer_evaluations, in the same order, so answer_evaluations[i] belongs to retrieved_list[i].
	Output: nothing. Prints the table.
	What it does: groups the answer evaluations by the setup of their Retrieved and prints, per setup, the share of
	I_DONT_KNOW answers and the mean correctness, faithfulness and relevance. I_DONT_KNOW answers count as 0.
	"""
	# 1. Group the answer evaluations by setup, and keep each setup's k: rerank_top_k for a reranked setup, else top_k
	evaluations_by_setup = {}
	k_by_setup = {}
	for index in range(len(retrieved_list)):
		retrieved = retrieved_list[index]
		setup = (retrieved.chunk_type, retrieved.retrieval_type, retrieved.index_type, retrieved.query_type, retrieved.reranking)
		if setup not in evaluations_by_setup:
			evaluations_by_setup[setup] = []
			if retrieved.reranking:
				k_by_setup[setup] = retrieved.rerank_top_k
			else:
				k_by_setup[setup] = retrieved.top_k
		evaluations_by_setup[setup].append(answer_evaluations[index])

	# 2. Print the header, with the judge of the answers
	first_answer_evaluation = answer_evaluations[0]
	print(f"\n\nAnswer evaluation per setup (mean over the eval questions), judged by {first_answer_evaluation.framework} with {first_answer_evaluation.llm_model}")
	header = (
		f"{'chunk':<9}{'retrieval':<10}{'index':<19}{'query':<10}{'rerank':<7}{'k':>3}{'n':>5}  "
		f"{'idk':>6}{'correct':>9}{'faithful':>10}{'relevant':>10}"
	)
	print(header)

	# 3. Print one row per setup, in setup order
	for setup in sorted(evaluations_by_setup):
		chunk_type, retrieval_type, index_type, query_type, reranking = setup
		answer_evaluation_list = evaluations_by_setup[setup]
		question_count = len(answer_evaluation_list)

		i_dont_know_sum = 0.0
		correctness_sum = 0.0
		faithfulness_sum = 0.0
		relevance_sum = 0.0
		for answer_evaluation in answer_evaluation_list:
			i_dont_know_sum = i_dont_know_sum + answer_evaluation.i_dont_know
			correctness_sum = correctness_sum + answer_evaluation.correctness
			faithfulness_sum = faithfulness_sum + answer_evaluation.faithfulness
			relevance_sum = relevance_sum + answer_evaluation.relevance

		k = k_by_setup[setup]
		if reranking:
			rerank_text = "yes"
		else:
			rerank_text = "no"

		row = (
			f"{chunk_type:<9}{retrieval_type:<10}{index_type:<19}{query_type:<10}{rerank_text:<7}{k:>3}{question_count:>5}  "
			f"{i_dont_know_sum / question_count:>6.3f}{correctness_sum / question_count:>9.3f}"
			f"{faithfulness_sum / question_count:>10.3f}{relevance_sum / question_count:>10.3f}"
		)
		print(row)
