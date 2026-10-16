"""
summary.py

Averages the l10 evaluations of every retrieval setup over its eval questions: one RetrievalSummary per retrieval
setup and one AnswerSummary per kept setup.
"""

from analysis.schema import AnswerSummary
from analysis.schema import RetrievalSummary
from analysis.schema import Setup
from l6_retrieval.schema import Retrieved
from l10_evaluation.schema import AnswerEvaluation
from l10_evaluation.schema import RetrievalEvaluation

def get_setup_key(retrieved: Retrieved) -> tuple:
	"""
	Input: retrieved, one Retrieved.
	Output: its setup as a tuple (chunk type, retrieval type, index type, query type, reranking, k), usable as a dict
	key and sortable, unlike a Setup.
	What it does: takes k as rerank_top_k for a reranked Retrieved, else top_k.
	"""
	if retrieved.reranking:
		k = retrieved.rerank_top_k
	else:
		k = retrieved.top_k
	setup_key = (retrieved.chunk_type, retrieved.retrieval_type, retrieved.index_type, retrieved.query_type, retrieved.reranking, k)
	return setup_key

def get_retrieval_summaries(retrieval_evaluations: list[tuple[Retrieved, RetrievalEvaluation]]) -> list[RetrievalSummary]:
	"""
	Input: retrieval_evaluations, one (Retrieved, RetrievalEvaluation) per eval question and retrieval setup.
	Output: one RetrievalSummary per setup, in setup order.
	What it does: groups the evaluations by setup and averages each chunk-level and doc-level metric.
	"""
	# 1. Group the evaluations by setup. The key is a tuple, since a Setup cannot be a dict key.
	# k is part of the setup: every evaluation of one setup has the same k
	evaluations_by_setup = {}
	for retrieved, retrieval_evaluation in retrieval_evaluations:
		setup_key = (retrieved.chunk_type, retrieved.retrieval_type, retrieved.index_type, retrieved.query_type, retrieved.reranking, retrieval_evaluation.k)
		if setup_key not in evaluations_by_setup:
			evaluations_by_setup[setup_key] = []
		evaluations_by_setup[setup_key].append(retrieval_evaluation)

	# 2. Average each metric per setup, in setup order
	retrieval_summaries = []
	for setup_key in sorted(evaluations_by_setup):
		chunk_type, retrieval_type, index_type, query_type, reranking, k = setup_key
		setup_retrieval_evaluations = evaluations_by_setup[setup_key]
		question_count = len(setup_retrieval_evaluations)

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
		for retrieval_evaluation in setup_retrieval_evaluations:
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

		setup = Setup(
			chunk_type=chunk_type,
			retrieval_type=retrieval_type,
			index_type=index_type,
			query_type=query_type,
			reranking=reranking,
			k=k
		)
		retrieval_summary = RetrievalSummary(
			setup=setup,
			n=question_count,
			chunk_hit=chunk_hit_sum / question_count,
			chunk_precision=chunk_precision_sum / question_count,
			chunk_recall=chunk_recall_sum / question_count,
			chunk_mrr=chunk_mrr_sum / question_count,
			chunk_ndcg=chunk_ndcg_sum / question_count,
			doc_hit=doc_hit_sum / question_count,
			doc_precision=doc_precision_sum / question_count,
			doc_recall=doc_recall_sum / question_count,
			doc_mrr=doc_mrr_sum / question_count,
			doc_ndcg=doc_ndcg_sum / question_count
		)
		retrieval_summaries.append(retrieval_summary)

	return retrieval_summaries

def get_answer_summaries(retrieved_list: list[Retrieved], answer_evaluations: list[AnswerEvaluation]) -> list[AnswerSummary]:
	"""
	Input: retrieved_list and answer_evaluations, in the same order, so answer_evaluations[i] belongs to retrieved_list[i].
	Output: one AnswerSummary per kept setup, in setup order.
	What it does: groups the answer evaluations by the setup of their Retrieved and averages the I_DONT_KNOW share,
	correctness, faithfulness and relevance.
	"""
	# 1. Group the answer evaluations by the setup of their Retrieved
	evaluations_by_setup = {}
	for index in range(len(retrieved_list)):
		setup_key = get_setup_key(retrieved_list[index])
		if setup_key not in evaluations_by_setup:
			evaluations_by_setup[setup_key] = []
		evaluations_by_setup[setup_key].append(answer_evaluations[index])

	# 2. Average each field per setup, in setup order. I_DONT_KNOW answers already hold 0 on the three scores.
	# A score the judge could not give (None) is left out of that score's mean
	answer_summaries = []
	for setup_key in sorted(evaluations_by_setup):
		chunk_type, retrieval_type, index_type, query_type, reranking, k = setup_key
		setup_answer_evaluations = evaluations_by_setup[setup_key]
		question_count = len(setup_answer_evaluations)

		unscored_count = 0
		i_dont_know_sum = 0.0
		correctness_sum = 0.0
		correctness_count = 0
		faithfulness_sum = 0.0
		faithfulness_count = 0
		relevance_sum = 0.0
		relevance_count = 0
		for answer_evaluation in setup_answer_evaluations:
			i_dont_know_sum = i_dont_know_sum + answer_evaluation.i_dont_know
			if answer_evaluation.correctness is None or answer_evaluation.faithfulness is None or answer_evaluation.relevance is None:
				unscored_count = unscored_count + 1
			if answer_evaluation.correctness is not None:
				correctness_sum = correctness_sum + answer_evaluation.correctness
				correctness_count = correctness_count + 1
			if answer_evaluation.faithfulness is not None:
				faithfulness_sum = faithfulness_sum + answer_evaluation.faithfulness
				faithfulness_count = faithfulness_count + 1
			if answer_evaluation.relevance is not None:
				relevance_sum = relevance_sum + answer_evaluation.relevance
				relevance_count = relevance_count + 1

		# A setup whose every answer lacks a score gets 0 for it; its unscored count shows why
		correctness_mean = 0.0
		if correctness_count > 0:
			correctness_mean = correctness_sum / correctness_count
		faithfulness_mean = 0.0
		if faithfulness_count > 0:
			faithfulness_mean = faithfulness_sum / faithfulness_count
		relevance_mean = 0.0
		if relevance_count > 0:
			relevance_mean = relevance_sum / relevance_count

		setup = Setup(
			chunk_type=chunk_type,
			retrieval_type=retrieval_type,
			index_type=index_type,
			query_type=query_type,
			reranking=reranking,
			k=k
		)
		answer_summary = AnswerSummary(
			setup=setup,
			n=question_count,
			unscored=unscored_count,
			i_dont_know=i_dont_know_sum / question_count,
			correctness=correctness_mean,
			faithfulness=faithfulness_mean,
			relevance=relevance_mean
		)
		answer_summaries.append(answer_summary)

	return answer_summaries
