import math

from common import constants
from schema.chunk import Chunk
from schema.eval_sample import EvalSample
from schema.retrieved import Retrieved
from schema.retrieved_eval import Eval
from schema.retrieved_eval import RetrievedEval


def compute_eval(relevant_count: int, first_relevant_rank: int, dcg: float, recall: float, k: int) -> Eval:
	"""
	Input: relevant_count, the relevant chunks found; first_relevant_rank, 0 for a miss; dcg, their summed
	discounted gain; recall; k, the number of chunks asked for.
	Output: the Eval of one level.
	What it does: works out hit, precision, MRR and nDCG. The best order puts every relevant chunk found at the top.
	"""
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

	return Eval(
		hit=hit,
		precision=relevant_count / k,
		recall=recall,
		mrr=mrr,
		ndcg=ndcg,
		first_relevant_rank=first_relevant_rank
	)


class RetrievalEvaluation:
	def __init__(self, eval_samples: list[EvalSample]):
		"""
		Input: eval_samples, the eval questions.
		Output: nothing.
		What it does: stores each eval question by its id, so each Retrieved finds its question by retrieved.id.
		"""
		self.eval_samples_by_id = {}
		for eval_sample in eval_samples:
			self.eval_samples_by_id[eval_sample.id] = eval_sample

	def compute_retrieved_evals(self, retrieved_items: list[Retrieved], chunks: dict[str, Chunk]) -> list[RetrievedEval]:
		"""
		Input: retrieved_items, the results to score; chunks, a dict of chunk_id to Chunk of their chunk type.
		Output: one RetrievedEval per Retrieved, in the same order.
		What it does: scores each Retrieved against its eval question at chunk level and at doc level.
		"""
		retrieved_evals = []
		for retrieved in retrieved_items:
			eval_sample = self.eval_samples_by_id[retrieved.id]

			# The number of chunks the retriever was asked for. Missing chunks count as not relevant
			if retrieved.setup.reranking:
				k = retrieved.rerank_top_k
			else:
				k = retrieved.top_k

			retrieved_evals.append(RetrievedEval(
				retrieved=retrieved,
				chunk_eval=self.chunk_eval(retrieved, chunks, eval_sample, k),
				doc_eval=self.doc_eval(retrieved, eval_sample, k)
			))

		print(f"Retrieved scored : {len(retrieved_evals)}")
		return retrieved_evals

	def chunk_eval(self, retrieved: Retrieved, chunks: dict[str, Chunk], eval_sample: EvalSample, k: int) -> Eval:
		"""
		Input: retrieved; chunks; eval_sample, its eval question; k.
		Output: the Eval at chunk level.
		What it does: a chunk is relevant if it comes from an EvalSample doc and at least CHUNK_CONTEXT_OVERLAP_THRESHOLD of
		its words are in one of the context passages. Recall is the share of context passages found.
		"""
		# 1. Split each context passage into its lowercase words
		passage_word_sets = []
		for passage in eval_sample.context:
			passage_word_sets.append(set(passage.lower().split()))

		# 2. Go over the chunks best first, counting the relevant ones and the passages they come from
		relevant_count = 0
		first_relevant_rank = 0
		dcg = 0.0
		found_passage_positions = set()
		for position in range(len(retrieved.chunk_ids)):
			if retrieved.doc_ids[position] not in eval_sample.doc_ids:
				continue

			chunk_word_set = set(chunks[retrieved.chunk_ids[position]].text.lower().split())
			if len(chunk_word_set) == 0:
				continue

			is_relevant = False
			for passage_position in range(len(passage_word_sets)):
				shared_words = chunk_word_set & passage_word_sets[passage_position]
				overlap = len(shared_words) / len(chunk_word_set)
				if overlap >= constants.CHUNK_CONTEXT_OVERLAP_THRESHOLD:
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
		recall = 0.0
		if len(passage_word_sets) > 0:
			recall = len(found_passage_positions) / len(passage_word_sets)
		return compute_eval(relevant_count, first_relevant_rank, dcg, recall, k)

	def doc_eval(self, retrieved: Retrieved, eval_sample: EvalSample, k: int) -> Eval:
		"""
		Input: retrieved; eval_sample, its eval question; k.
		Output: the Eval at doc level.
		What it does: a chunk is relevant if it comes from an EvalSample doc. Recall is the share of EvalSample docs found.
		"""
		# 1. Go over the chunks best first, counting the relevant ones and the EvalSample docs they come from
		relevant_count = 0
		first_relevant_rank = 0
		dcg = 0.0
		found_doc_ids = set()
		for position in range(len(retrieved.doc_ids)):
			doc_id = retrieved.doc_ids[position]
			if doc_id not in eval_sample.doc_ids:
				continue

			rank = position + 1
			relevant_count = relevant_count + 1
			found_doc_ids.add(doc_id)
			if first_relevant_rank == 0:
				first_relevant_rank = rank
			dcg = dcg + 1.0 / math.log2(rank + 1)

		# 2. Work out the metrics
		recall = 0.0
		if len(eval_sample.doc_ids) > 0:
			recall = len(found_doc_ids) / len(set(eval_sample.doc_ids))
		return compute_eval(relevant_count, first_relevant_rank, dcg, recall, k)
