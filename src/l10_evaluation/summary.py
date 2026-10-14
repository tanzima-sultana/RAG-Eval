"""
summary.py

Prints the retrieval quality of every retrieval setup as one table row, averaged over its eval questions.
"""

from l6_retrieval.schema import Retrieved
from l10_evaluation.schema import RetrievalQuality

def print_retrieval_summary(retrieval_qualities: list[tuple[Retrieved, RetrievalQuality]]) -> None:
	"""
	Input: retrieval_qualities, one (Retrieved, RetrievalQuality) per eval question and retrieval setup.
	Output: nothing. Prints the table.
	What it does: groups the qualities by setup (chunk type, retrieval type, index type, query type, reranking)
	and prints the mean of each metric per setup.
	"""
	# 1. Group the qualities by setup
	qualities_by_setup = {}
	for retrieved, retrieval_quality in retrieval_qualities:
		setup = (retrieved.chunk_type, retrieved.retrieval_type, retrieved.index_type, retrieved.query_type, retrieved.reranking)
		if setup not in qualities_by_setup:
			qualities_by_setup[setup] = []
		qualities_by_setup[setup].append(retrieval_quality)

	# 2. Print the header
	print("\n\nRetrieval quality per setup (mean over the eval questions)")
	header = (
		f"{'chunk':<9}{'retrieval':<10}{'index':<19}{'query':<10}{'rerank':<7}{'k':>3}{'n':>5}  "
		f"{'c_hit':>6}{'c_prec':>7}{'c_rec':>6}{'c_mrr':>6}{'c_ndcg':>7}  "
		f"{'d_hit':>6}{'d_prec':>7}{'d_rec':>6}{'d_mrr':>6}{'d_ndcg':>7}"
	)
	print(header)

	# 3. Print one row per setup, in setup order
	for setup in sorted(qualities_by_setup):
		chunk_type, retrieval_type, index_type, query_type, reranking = setup
		retrieval_quality_list = qualities_by_setup[setup]
		question_count = len(retrieval_quality_list)

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
		for retrieval_quality in retrieval_quality_list:
			chunk_hit_sum = chunk_hit_sum + retrieval_quality.chunk_quality.hit
			chunk_precision_sum = chunk_precision_sum + retrieval_quality.chunk_quality.precision
			chunk_recall_sum = chunk_recall_sum + retrieval_quality.chunk_quality.recall
			chunk_mrr_sum = chunk_mrr_sum + retrieval_quality.chunk_quality.mrr
			chunk_ndcg_sum = chunk_ndcg_sum + retrieval_quality.chunk_quality.ndcg
			doc_hit_sum = doc_hit_sum + retrieval_quality.doc_quality.hit
			doc_precision_sum = doc_precision_sum + retrieval_quality.doc_quality.precision
			doc_recall_sum = doc_recall_sum + retrieval_quality.doc_quality.recall
			doc_mrr_sum = doc_mrr_sum + retrieval_quality.doc_quality.mrr
			doc_ndcg_sum = doc_ndcg_sum + retrieval_quality.doc_quality.ndcg

		# Every quality of one setup has the same k
		k = retrieval_quality_list[0].k
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
