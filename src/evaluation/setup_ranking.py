from schema.retrieved_eval import RetrievedEval
from schema.retrieved_eval import SetupScore
from schema.setup import Setup


def rank_setups(retrieved_evals: list[RetrievedEval], n_top_setups: int) -> list[Setup]:
	"""
	Input: retrieved_evals, one RetrievedEval per val query per setup; n_top_setups, the number of setups to keep.
	Output: the best n_top_setups setups, best first.
	What it does: scores each setup by the mean of its chunk level scores over its val queries, ranks all setups
	together, plain and reranked, by mean recall and then mean MRR, prints the ranking and keeps the top ones.
	"""
	if n_top_setups < 1:
		raise ValueError(f"n_top_setups ({n_top_setups}) must be at least 1.")

	# 1. Group the evals by their setup, keeping the order the setups first appear in
	setups = []
	evals_by_setup = {}
	for retrieved_eval in retrieved_evals:
		setup = retrieved_eval.retrieved.setup
		if setup not in evals_by_setup:
			setups.append(setup)
			evals_by_setup[setup] = []
		evals_by_setup[setup].append(retrieved_eval)

	# 2. Score each setup by the mean of each chunk level score over its val queries
	setup_scores = []
	for setup in setups:
		setup_evals = evals_by_setup[setup]
		hit_sum = 0.0
		recall_sum = 0.0
		mrr_sum = 0.0
		ndcg_sum = 0.0
		for retrieved_eval in setup_evals:
			hit_sum = hit_sum + retrieved_eval.chunk_eval.hit
			recall_sum = recall_sum + retrieved_eval.chunk_eval.recall
			mrr_sum = mrr_sum + retrieved_eval.chunk_eval.mrr
			ndcg_sum = ndcg_sum + retrieved_eval.chunk_eval.ndcg
		queries_count = len(setup_evals)
		setup_scores.append(SetupScore(
			setup=setup,
			queries_count=queries_count,
			mean_hit=hit_sum / queries_count,
			mean_recall=recall_sum / queries_count,
			mean_mrr=mrr_sum / queries_count,
			mean_ndcg=ndcg_sum / queries_count
		))

	# 3. Rank by mean recall, then mean MRR. The sort is stable, so equal scores keep the setup order
	ranked_setup_scores = sorted(
		setup_scores,
		key=lambda setup_score: (setup_score.mean_recall, setup_score.mean_mrr),
		reverse=True
	)

	# 4. Print the ranking and keep the top n_top_setups
	print(f"Setups ranked : {len(ranked_setup_scores)}, kept : {min(n_top_setups, len(ranked_setup_scores))}")
	for rank in range(len(ranked_setup_scores)):
		if rank == n_top_setups:
			break
		setup_score = ranked_setup_scores[rank]
		setup = setup_score.setup
		print(
			f"{rank + 1}. {setup.chunk_type} {setup.index_type} {setup.query_type} {setup.retrieval_type} "
			f"reranking={setup.reranking} : recall {setup_score.mean_recall:.4f}, mrr {setup_score.mean_mrr:.4f}, "
			f"ndcg {setup_score.mean_ndcg:.4f}, hit {setup_score.mean_hit:.4f} over {setup_score.queries_count} queries"
		)

	best_setups = []
	for setup_score in ranked_setup_scores[:n_top_setups]:
		best_setups.append(setup_score.setup)
	return best_setups
