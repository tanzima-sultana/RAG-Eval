import dataclasses
import json
import os

from common import paths
from common.args import QueryArgs
from common.args import RetrievalArgs
from schema.manifest import BuildManifest
from schema.retrieved_eval import RetrievedEval
from schema.retrieved_eval import SetupScore
from schema.setup import Setup
from schema.setup import SetupArtifacts
from schema.setup import SetupLineage
from schema.setup import SetupParams
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


def save_best_setups(best_setups: list[Setup], manifest: BuildManifest, manifest_path: str, retrieval_args: RetrievalArgs, query_args: QueryArgs, output_path: str, eval_dataset_hash: str, benchmark_hash: str) -> list[SetupRecord]:
	"""
	Input: best_setups, best first; manifest, the build manifest, which says where each index is; manifest_path, its
	file; retrieval_args, the retrieval settings the setups were scored with; query_args, the query settings that
	wrote the query texts; output_path, the dataset's output folder; eval_dataset_hash, from the eval dataset metadata;
	benchmark_hash, from get_benchmark_hash.
	Output: one SetupRecord per setup, in the same order.
	What it does: builds each record grouped by kind: the artifacts to load (the index paths from the manifest and the
	embedding model), the params that decide its results, flat, and its lineage. Saves the records as JSON through a
	temporary file to best_setups_{build_hash}_{eval_dataset_hash}_{benchmark_hash}.json.
	"""
	setup_records = []
	for setup in best_setups:
		# 1. Each index of the setup. Hybrid's index_type is the dense and BM25 index types joined with "+", so it gets
		# two paths, in that order
		index_paths = []
		for index_type in setup.index_type.split("+"):
			index_location = None
			for index_entry in manifest.index_entries:
				if index_entry.chunk_type == setup.chunk_type and index_entry.index_type == index_type:
					index_location = index_entry.location
			if index_location is None:
				raise ValueError(f"The build manifest has no {setup.chunk_type} {index_type} index.")
			index_paths.append(index_location)

		# 2. The record, grouped by kind
		setup_records.append(SetupRecord(
			artifacts=SetupArtifacts(
				index_paths=tuple(index_paths),
				embedding_model=manifest.embedding_model
			),
			params=SetupParams(
				chunk_type=setup.chunk_type,
				index_type=setup.index_type,
				query_type=setup.query_type,
				retrieval_type=setup.retrieval_type,
				reranking=setup.reranking,
				top_k=retrieval_args.top_k,
				rerank_top_k=retrieval_args.rerank_top_k,
				cross_encoder_model=retrieval_args.cross_encoder_model,
				query_llm_model=query_args.query_llm_args.llm_model,
				query_llm_max_tokens=query_args.query_llm_args.max_tokens,
				n_multi_qus=query_args.n_multi_qus,
				ollama_context_length=query_args.ollama_context_length
			),
			lineage=SetupLineage(
				manifest_path=manifest_path,
				build_hash=manifest.build_hash,
				eval_dataset_hash=eval_dataset_hash,
				benchmark_hash=benchmark_hash
			)
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


def load_best_setups(best_setups_file: str) -> list[SetupRecord]:
	"""
	Input: best_setups_file, the best setups JSON the benchmark saved.
	Output: the saved SetupRecord list, best first.
	What it does: reads the best setups JSON, one object per section, and raises FileNotFoundError if the benchmark has
	not saved it.
	"""
	if not os.path.exists(best_setups_file):
		raise FileNotFoundError(f"No saved best setups at {best_setups_file}. Run scripts/benchmark.sh first.")

	with open(best_setups_file, encoding="utf-8") as f:
		setup_record_dicts = json.load(f)

	setup_records = []
	for setup_record_dict in setup_record_dicts:
		artifacts_dict = setup_record_dict["artifacts"]
		setup_records.append(SetupRecord(
			artifacts=SetupArtifacts(
				index_paths=tuple(artifacts_dict["index_paths"]),
				embedding_model=artifacts_dict["embedding_model"]
			),
			params=SetupParams(**setup_record_dict["params"]),
			lineage=SetupLineage(**setup_record_dict["lineage"])
		))
	return setup_records
