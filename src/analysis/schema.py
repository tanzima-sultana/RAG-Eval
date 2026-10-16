import dataclasses

@dataclasses.dataclass
class Setup:
	"""
	One retrieval setup: what was searched and how. Shared by the retrieval and answer summaries and the comparison,
	so all of them name a setup the same way.

	Fields:
		chunk_type:     fixed, sentence or semantic.
		retrieval_type: dense, bm25 or hybrid.
		index_type:     The index or "+" joined index pair searched, e.g. faiss_flatip+bm25.
		query_type:     original, rewritten, multi or hyde.
		reranking:      True if the cross-encoder reranked the chunks.
		k:              Number of chunks scored: rerank_top_k for a reranked setup, else top_k.
	"""
	chunk_type: str
	retrieval_type: str
	index_type: str
	query_type: str
	reranking: bool
	k: int

@dataclasses.dataclass
class RetrievalSummary:
	"""
	The retrieval evaluation of one setup, averaged over its eval questions. One row of the retrieval summary table.

	Fields:
		setup:           The retrieval setup.
		n:               Number of eval questions averaged.
		chunk_hit:       Mean chunk-level hit.
		chunk_precision: Mean chunk-level precision.
		chunk_recall:    Mean chunk-level recall.
		chunk_mrr:       Mean chunk-level MRR.
		chunk_ndcg:      Mean chunk-level nDCG.
		doc_hit:         Mean doc-level hit.
		doc_precision:   Mean doc-level precision.
		doc_recall:      Mean doc-level recall.
		doc_mrr:         Mean doc-level MRR.
		doc_ndcg:        Mean doc-level nDCG.
	"""
	setup: Setup
	n: int
	chunk_hit: float
	chunk_precision: float
	chunk_recall: float
	chunk_mrr: float
	chunk_ndcg: float
	doc_hit: float
	doc_precision: float
	doc_recall: float
	doc_mrr: float
	doc_ndcg: float

@dataclasses.dataclass
class AnswerSummary:
	"""
	The answer evaluation of one setup, averaged over its eval questions. One row of the answer summary table.
	I_DONT_KNOW answers count as 0 on the three scores. A score the judge could not give (None) is left out of that
	score's mean.

	Fields:
		setup:        The retrieval setup the answers were generated from.
		n:            Number of eval questions averaged.
		unscored:     Number of answers with at least one score the judge could not give.
		i_dont_know:  Share of I_DONT_KNOW answers.
		correctness:  Mean correctness.
		faithfulness: Mean faithfulness.
		relevance:    Mean relevance.
	"""
	setup: Setup
	n: int
	unscored: int
	i_dont_know: float
	correctness: float
	faithfulness: float
	relevance: float

@dataclasses.dataclass
class AnalysisRun:
	"""
	The saved record of one eval run, read back by later runs to compare against.
	A run with the same settings_hash as a saved one replaces it, so the history holds only different experiments.

	Fields:
		run_id:              Unique id of the run, from its start time.
		created_at:          When the run was analysed, as an ISO date and time.
		settings:            Every setting of the run, as {setting name: value}.
		settings_hash:       The sha256 of settings, the same for two runs with the same settings.
		judge:               The answer evaluation framework and judge model, e.g. "ragas gpt-4.1-mini".
		retrieval_summaries: One RetrievalSummary per retrieval setup.
		answer_summaries:    One AnswerSummary per kept setup, the setups answers were generated for.
	"""
	run_id: str
	created_at: str
	settings: dict
	settings_hash: str
	judge: str
	retrieval_summaries: list[RetrievalSummary]
	answer_summaries: list[AnswerSummary]
