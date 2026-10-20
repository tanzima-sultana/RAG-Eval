"""
retrieval_evaluation.py

Measures how well each Retrieved found its eval question's source, at chunk level and at doc level.
"""

import math

from common.constants import CHUNK_CONTEXT_OVERLAP_THRESHOLD
from stages.build.chunking.schema import Chunk
from stages.evaluation.retrieval.schema import Retrieved
from stages.evaluation.evaluation.schema import Evaluation
from stages.evaluation.evaluation.schema import RetrievalEvaluation
from stages.evaluation.evaluation.validate import validate_n_retrieval_evaluation
from stages.question.qna.schema import EvalQnA

class RetrievalEvaluator:
	def __init__(self, eval_qna: list[EvalQnA]):
		"""
		Input: eval_qna, the eval questions.
		Output: none.
		What it does: stores each eval question by its id, so each Retrieved finds its question by retrieved.id.
		"""
		self.eval_qna_by_id = {}
		for eval_qna_item in eval_qna:
			self.eval_qna_by_id[eval_qna_item.id] = eval_qna_item

	def compute_retrieval_evaluation(self, retrieved: Retrieved, chunks: dict[str, Chunk]) -> RetrievalEvaluation:
		"""
		Input: retrieved, one Retrieved; chunks, the chunks of its chunk type.
		Output: its RetrievalEvaluation, at chunk level and at doc level.
		What it does: finds the eval question of retrieved and scores its chunks at both levels.
		"""
		eval_qna_item = self.eval_qna_by_id[retrieved.id]

		# The number of chunks the retriever was asked for. Missing chunks count as not relevant
		if retrieved.reranking:
			k = retrieved.rerank_top_k
		else:
			k = retrieved.top_k

		chunk_evaluation = self.compute_chunk_evaluation(retrieved, chunks, eval_qna_item, k)
		doc_evaluation = self.compute_doc_evaluation(retrieved, eval_qna_item, k)

		return RetrievalEvaluation(
			k=k,
			chunk_evaluation=chunk_evaluation,
			doc_evaluation=doc_evaluation
		)

	def get_filtered_retrieval_evaluations(self, n_retrieval_evaluation: int, retrieval_evaluations: list[tuple[Retrieved, RetrievalEvaluation]]) -> list[tuple[Retrieved, RetrievalEvaluation]]:
		"""
		Input: n_retrieval_evaluation, the number of setups to keep of each kind; retrieval_evaluations, one
		(Retrieved, RetrievalEvaluation) per eval question and setup.
		Output: the (Retrieved, RetrievalEvaluation) of the kept setups, at most n_retrieval_evaluation x the number of
		questions of each kind.
		What it does: ranks the setups by mean chunk-level recall, then mean chunk-level MRR, and keeps the best
		n_retrieval_evaluation plain setups and the best n_retrieval_evaluation reranked setups. Plain and reranked setups
		are ranked apart, because they are scored at a different k. A Retrieved with the same question and the same
		chunks as one already kept is skipped, since it would give the LLM the same prompt.
		"""
		validate_n_retrieval_evaluation(n_retrieval_evaluation)

		# 1. Group the evaluations by setup
		retrieval_evaluations_by_setup = {}
		for retrieved, retrieval_evaluation in retrieval_evaluations:
			setup = (retrieved.chunk_type, retrieved.retrieval_type, retrieved.index_type, retrieved.query_type, retrieved.reranking)
			if setup not in retrieval_evaluations_by_setup:
				retrieval_evaluations_by_setup[setup] = []
			retrieval_evaluations_by_setup[setup].append((retrieved, retrieval_evaluation))

		# 2. Score each setup by its mean chunk-level recall and MRR, plain and reranked apart
		plain_setup_scores = []
		reranked_setup_scores = []
		for setup in retrieval_evaluations_by_setup:
			setup_retrieval_evaluations = retrieval_evaluations_by_setup[setup]
			recall_sum = 0.0
			mrr_sum = 0.0
			for retrieved, retrieval_evaluation in setup_retrieval_evaluations:
				recall_sum = recall_sum + retrieval_evaluation.chunk_evaluation.recall
				mrr_sum = mrr_sum + retrieval_evaluation.chunk_evaluation.mrr
			mean_recall = recall_sum / len(setup_retrieval_evaluations)
			mean_mrr = mrr_sum / len(setup_retrieval_evaluations)

			reranking = setup[4]
			if reranking:
				reranked_setup_scores.append((mean_recall, mean_mrr, setup))
			else:
				plain_setup_scores.append((mean_recall, mean_mrr, setup))

		# 3. Keep the best n_retrieval_evaluation of each. Equal recall and MRR fall back to the setup order
		best_plain_setup_scores = sorted(plain_setup_scores, reverse=True)[:n_retrieval_evaluation]
		best_reranked_setup_scores = sorted(reranked_setup_scores, reverse=True)[:n_retrieval_evaluation]

		# 4. Collect the evaluations of the kept setups, best first
		kept_retrieval_evaluations = []
		for mean_recall, mean_mrr, setup in best_plain_setup_scores:
			kept_retrieval_evaluations.extend(retrieval_evaluations_by_setup[setup])
		for mean_recall, mean_mrr, setup in best_reranked_setup_scores:
			kept_retrieval_evaluations.extend(retrieval_evaluations_by_setup[setup])

		# 5. Skip duplicates: the same question with the same chunks, in the same order, as one already kept
		filtered_retrieval_evaluations = []
		kept_question_chunks = set()
		duplicate_count = 0
		for retrieved, retrieval_evaluation in kept_retrieval_evaluations:
			question_chunks = (retrieved.id, tuple(retrieved.chunk_ids))
			if question_chunks in kept_question_chunks:
				duplicate_count = duplicate_count + 1
				continue
			kept_question_chunks.add(question_chunks)
			filtered_retrieval_evaluations.append((retrieved, retrieval_evaluation))

		print(f"\n\nFiltered retrieved outputs : {len(filtered_retrieval_evaluations)} ({duplicate_count} duplicates skipped)")
		return filtered_retrieval_evaluations

	def compute_chunk_evaluation(self, retrieved: Retrieved, chunks: dict[str, Chunk], eval_qna_item: EvalQnA, k: int) -> Evaluation:
		"""
		Input: one Retrieved, the chunks, its eval question and k.
		Output: the Evaluation at chunk level.
		What it does: a chunk is relevant if it comes from a gold doc and at least CHUNK_CONTEXT_OVERLAP_THRESHOLD of
		its words are in one of the context passages. Recall is the share of context passages found.
		"""
		# 1. Split each context passage into its lowercase words
		# A passage is about three chunks long, so a relevant chunk sits inside a passage
		passage_word_sets = []
		for passage in eval_qna_item.context:
			passage_word_sets.append(set(passage.lower().split()))

		# 2. Go over the chunks best first, counting the relevant ones and the passages they come from
		relevant_count = 0
		first_relevant_rank = 0
		dcg = 0.0
		found_passage_positions = set()
		for position in range(len(retrieved.chunk_ids)):
			chunk_id = retrieved.chunk_ids[position]
			doc_id = retrieved.doc_ids[position]
			if doc_id not in eval_qna_item.doc_ids:
				continue

			chunk_word_set = set(chunks[chunk_id].text.lower().split())
			if len(chunk_word_set) == 0:
				continue

			is_relevant = False
			for passage_position in range(len(passage_word_sets)):
				shared_words = chunk_word_set & passage_word_sets[passage_position]
				overlap = len(shared_words) / len(chunk_word_set)
				if overlap >= CHUNK_CONTEXT_OVERLAP_THRESHOLD:
					is_relevant = True
					found_passage_positions.add(passage_position)
			if not is_relevant:
				continue

			rank = position + 1
			relevant_count = relevant_count + 1
			if first_relevant_rank == 0:
				first_relevant_rank = rank
			dcg = dcg + 1.0 / math.log2(rank + 1)

		# 3. Work out the metrics
		# The best order puts every relevant chunk found at the top
		ideal_dcg = 0.0
		for rank in range(1, relevant_count + 1):
			ideal_dcg = ideal_dcg + 1.0 / math.log2(rank + 1)

		hit = 0
		mrr = 0.0
		ndcg = 0.0
		if relevant_count > 0:
			hit = 1
			mrr = 1.0 / first_relevant_rank
			ndcg = dcg / ideal_dcg

		recall = 0.0
		if len(passage_word_sets) > 0:
			recall = len(found_passage_positions) / len(passage_word_sets)

		return Evaluation(
			hit=hit,
			precision=relevant_count / k,
			recall=recall,
			mrr=mrr,
			ndcg=ndcg,
			first_relevant_rank=first_relevant_rank
		)

	def compute_doc_evaluation(self, retrieved: Retrieved, eval_qna_item: EvalQnA, k: int) -> Evaluation:
		"""
		Input: one Retrieved, its eval question and k.
		Output: the Evaluation at doc level.
		What it does: a chunk is relevant if it comes from a gold doc. Recall is the share of gold docs found.
		"""
		# 1. Go over the chunks best first, counting the relevant ones and the gold docs they come from
		relevant_count = 0
		first_relevant_rank = 0
		dcg = 0.0
		found_doc_ids = set()
		for position in range(len(retrieved.doc_ids)):
			doc_id = retrieved.doc_ids[position]
			if doc_id not in eval_qna_item.doc_ids:
				continue

			rank = position + 1
			relevant_count = relevant_count + 1
			found_doc_ids.add(doc_id)
			if first_relevant_rank == 0:
				first_relevant_rank = rank
			dcg = dcg + 1.0 / math.log2(rank + 1)

		# 2. Work out the metrics
		# The best order puts every relevant chunk found at the top
		ideal_dcg = 0.0
		for rank in range(1, relevant_count + 1):
			ideal_dcg = ideal_dcg + 1.0 / math.log2(rank + 1)

		hit = 0
		mrr = 0.0
		ndcg = 0.0
		if relevant_count > 0:
			hit = 1
			mrr = 1.0 / first_relevant_rank
			ndcg = dcg / ideal_dcg

		recall = 0.0
		if len(eval_qna_item.doc_ids) > 0:
			recall = len(found_doc_ids) / len(set(eval_qna_item.doc_ids))

		return Evaluation(
			hit=hit,
			precision=relevant_count / k,
			recall=recall,
			mrr=mrr,
			ndcg=ndcg,
			first_relevant_rank=first_relevant_rank
		)
