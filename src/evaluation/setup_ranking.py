import dataclasses
import json
import os

from common import paths
from schema.manifest import BuildManifest
from schema.retrieved_eval import RetrievedEval
from schema.retrieved_eval import SetupScore
from schema.setup import Setup
from schema.setup import SetupRecord


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


def save_best_setups(best_setups: list[Setup], manifest: BuildManifest, output_path: str, eval_dataset_hash: str, benchmark_hash: str) -> list[SetupRecord]:
	"""
	Input: best_setups, best first; manifest, the build manifest, which says where each index and chunk folder is;
	output_path, the dataset's output folder; eval_dataset_hash, from the eval dataset metadata; benchmark_hash, from
	get_benchmark_hash.
	Output: one SetupRecord per setup, in the same order.
	What it does: takes each setup's index location(s) and chunk folder from the manifest, and saves the records as
	JSON through a temporary file to best_setups_{build_hash}_{eval_dataset_hash}_{benchmark_hash}.json.
	"""
	setup_records = []
	for setup in best_setups:
		# The chunk folder of the setup's chunk type
		chunk_path = None
		for chunk_entry in manifest.chunk_entries:
			if chunk_entry.chunk_type == setup.chunk_type:
				chunk_path = chunk_entry.chunk_path
		if chunk_path is None:
			raise ValueError(f"The build manifest has no {setup.chunk_type} chunks.")

		# Hybrid's index_type is the dense and BM25 index types joined with "+", so it gets two locations in that order
		index_paths = []
		for index_type in setup.index_type.split("+"):
			index_location = None
			for index_entry in manifest.index_entries:
				if index_entry.chunk_type == setup.chunk_type and index_entry.index_type == index_type:
					index_location = index_entry.location
			if index_location is None:
				raise ValueError(f"The build manifest has no {setup.chunk_type} {index_type} index.")
			index_paths.append(index_location)

		setup_records.append(SetupRecord(
			setup=setup,
			index_paths=tuple(index_paths),
			chunk_path=chunk_path
		))

	best_setups_file = paths.get_best_setups_path(output_path, manifest.build_hash, eval_dataset_hash, benchmark_hash)
	os.makedirs(os.path.dirname(best_setups_file), exist_ok=True)
	setup_record_dicts = []
	for setup_record in setup_records:
		setup_record_dicts.append(dataclasses.asdict(setup_record))
	with open(best_setups_file + ".tmp", "w", encoding="utf-8") as f:
		json.dump(setup_record_dicts, f, indent=2)
	os.replace(best_setups_file + ".tmp", best_setups_file)

	print(f"Best setups saved : {best_setups_file}")
	return setup_records


def load_best_setups(output_path: str, build_hash: str, eval_dataset_hash: str, benchmark_hash: str) -> list[SetupRecord]:
	"""
	Input: output_path, the dataset's output folder; build_hash, eval_dataset_hash and benchmark_hash, the three
	hashes the file is named by.
	Output: the saved SetupRecord list, best first.
	What it does: reads the best setups JSON, and raises FileNotFoundError if the benchmark has not saved it.
	"""
	best_setups_file = paths.get_best_setups_path(output_path, build_hash, eval_dataset_hash, benchmark_hash)
	if not os.path.exists(best_setups_file):
		raise FileNotFoundError(f"No saved best setups at {best_setups_file}. Run scripts/benchmark.sh first.")

	with open(best_setups_file, encoding="utf-8") as f:
		setup_record_dicts = json.load(f)

	setup_records = []
	for setup_record_dict in setup_record_dicts:
		setup_records.append(SetupRecord(
			setup=Setup(**setup_record_dict["setup"]),
			index_paths=tuple(setup_record_dict["index_paths"]),
			chunk_path=setup_record_dict["chunk_path"]
		))
	return setup_records
