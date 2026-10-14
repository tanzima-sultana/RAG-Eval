"""
retrieval_evaluation.py

Measures how well each Retrieved found its eval question's source, at chunk level and at doc level.
"""

import math

from common.constants import CHUNK_CONTEXT_OVERLAP_THRESHOLD
from l2_chunking.schema import Chunk
from l6_retrieval.schema import Retrieved
from l10_evaluation.schema import Quality
from l10_evaluation.schema import RetrievalQuality
from qna.schema import EvalQnA

class RetrievalEvaluation:
	def __init__(self, eval_qna: list[EvalQnA]):
		"""
		Input: eval_qna, the eval questions.
		Output: none.
		What it does: stores each eval question by its id, so each Retrieved finds its question by retrieved.id.
		"""
		self.eval_qna_by_id = {}
		for eval_qna_item in eval_qna:
			self.eval_qna_by_id[eval_qna_item.id] = eval_qna_item

	def compute_retrieval_quality(self, retrieved: Retrieved, chunks: dict[str, Chunk]) -> RetrievalQuality:
		"""
		Input: retrieved, one Retrieved; chunks, the l2 chunks of its chunk type.
		Output: its RetrievalQuality, at chunk level and at doc level.
		What it does: finds the eval question of retrieved and scores its chunks at both levels.
		"""
		eval_qna_item = self.eval_qna_by_id[retrieved.id]

		# The number of chunks the retriever was asked for. Missing chunks count as not relevant
		if retrieved.reranking:
			k = retrieved.rerank_top_k
		else:
			k = retrieved.top_k

		chunk_quality = self.compute_chunk_quality(retrieved, chunks, eval_qna_item, k)
		doc_quality = self.compute_doc_quality(retrieved, eval_qna_item, k)

		return RetrievalQuality(
			k=k,
			chunk_quality=chunk_quality,
			doc_quality=doc_quality
		)

	def compute_chunk_quality(self, retrieved: Retrieved, chunks: dict[str, Chunk], eval_qna_item: EvalQnA, k: int) -> Quality:
		"""
		Input: one Retrieved, the l2 chunks, its eval question and k.
		Output: the Quality at chunk level.
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

		return Quality(
			hit=hit,
			precision=relevant_count / k,
			recall=recall,
			mrr=mrr,
			ndcg=ndcg,
			first_relevant_rank=first_relevant_rank
		)

	def compute_doc_quality(self, retrieved: Retrieved, eval_qna_item: EvalQnA, k: int) -> Quality:
		"""
		Input: one Retrieved, its eval question and k.
		Output: the Quality at doc level.
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

		return Quality(
			hit=hit,
			precision=relevant_count / k,
			recall=recall,
			mrr=mrr,
			ndcg=ndcg,
			first_relevant_rank=first_relevant_rank
		)
