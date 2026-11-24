import dataclasses
import json
import os
import time

from common import paths
from common.constants import BM25_INDEX
from common.constants import BM25_RETRIEVAL
from common.constants import DENSE_RETRIEVAL
from common.constants import HYBRID_RETRIEVAL
from common.constants import HYDE_QUERY
from common.constants import QDRANT_INDEX
from common.args import parse_benchmark_config
from common.args import make_benchmark_args
from common.args import make_manifest_chunking_args
from common.args import make_manifest_index_args
from common.args import make_ollama_args
from common.args import make_query_args
from common.args import make_pipeline_args
from common.args import make_retrieval_args
from common.args import make_vectordb_args
from evaluation.eval_dataset.cache import load_eval_dataset
from evaluation.retrieval_evaluation import RetrievalEvaluation
from evaluation.setup_ranking import rank_setups
from evaluation.setup_ranking import save_best_setups
from rag.build.chunking import Chunking
from rag.build.indexing import Indexing
from rag.build.indexing_vectordb import IndexingVectorDb
from rag.build.ingestion import load_documents
from rag.build.manifest import load_build_manifest_from_path
from rag.serving.pipeline import Pipeline
from rag.serving.query.query import QueryTransformation
from rag.serving.retrieval.retrieval import Retrieval
from schema.eval_dataset_metadata import EvalDatasetMetadata


def run_benchmark(args) -> None:
	"""
	Input: args, the parsed benchmark args.
	Output: nothing.
	What it does: runs the benchmark on one build and one eval dataset: reads the build manifest and the eval dataset
	metadata, checks they are of the same dataset, loads the documents and the eval set, transforms the queries, then
	for every chunk type and index the manifest lists loads the saved chunks and index, retrieves the val queries and
	scores them, and saves the best setups. It never builds and never generates the eval dataset.
	"""
	# 1. The build manifest and the eval dataset metadata. The only check is that they are of the same dataset
	print("\n\nBuild manifest and eval dataset metadata")
	manifest = load_build_manifest_from_path(args.manifest_path)
	print(f"Build manifest : {args.manifest_path}, built at {manifest.built_at}")

	with open(args.eval_dataset_metadata_path, encoding="utf-8") as f:
		eval_dataset_metadata = EvalDatasetMetadata(**json.load(f))
	print(f"Eval dataset metadata : {args.eval_dataset_metadata_path}")

	if manifest.dataset_name != eval_dataset_metadata.dataset_name:
		raise ValueError(
			f"The build manifest is of dataset {manifest.dataset_name}, but the eval dataset metadata is of dataset "
			f"{eval_dataset_metadata.dataset_name}."
		)

	output_path = paths.get_output_path(args.output_path, manifest.dataset_name)

	# 2. Documents, the ones the build saved for its input files
	print("\n\nDocuments")
	s1 = time.time()
	documents = load_documents(manifest.documents_path, manifest.input_hash)
	if documents is None:
		raise FileNotFoundError(
			f"No saved document list doc_ids_{manifest.input_hash}.json in {manifest.documents_path}. Run scripts/build.sh first."
		)
	e1 = time.time() - s1
	print(f"Documents : {len(documents)}")
	print("\nDocuments time : ", e1)

	# 3. Eval dataset, the one the metadata names
	print("\n\nEval dataset")
	s2 = time.time()
	eval_samples = load_eval_dataset(eval_dataset_metadata.eval_dataset_path)
	if eval_samples is None:
		raise FileNotFoundError(
			f"No saved eval dataset at {eval_dataset_metadata.eval_dataset_path}. Run scripts/eval_dataset.sh first."
		)
	e2 = time.time() - s2
	print(f"Eval dataset : {len(eval_samples)} from {eval_dataset_metadata.eval_dataset_path}")
	print("\nEval dataset time : ", e2)

	# 4. Query transformation of the eval questions, with the build's embedding model
	print("\n\nQuery transformation")
	query_args = make_query_args(args)
	query_path = os.path.join(args.output_path, "queries")

	s3 = time.time()
	query_transformation = QueryTransformation(query_args, query_path)
	queries = []
	for eval_sample in eval_samples:
		query = query_transformation.query(eval_sample.id, eval_sample.qus)
		queries.append(query)
	e3 = time.time() - s3
	print("\nQuery transformation time : ", e3)

	# 5. Query split: val for retrieval and retrieval evaluation, test for answer generation and answer evaluation
	print("\n\nQuery split")
	val_queries, test_queries = query_transformation.split_queries(queries)

	# Scores of every result, of every chunk type
	retrieval_evaluation = RetrievalEvaluation(eval_samples)
	retrieved_evals = []

	# The query settings of each query type searched, one Retrieval each: no extra query type for original, then one
	# for each extra query type the query layer made. The retrieved files of every setup are saved in one folder
	search_query_args_list = [dataclasses.replace(query_args, query_types=[])]
	for query_type in query_args.query_types:
		search_query_args_list.append(dataclasses.replace(query_args, query_types=[query_type]))
	retrieved_path = output_path + "/retrieved/"

	# 6. Every chunk type the build made, with the settings it made them with
	for chunk_entry in manifest.chunk_entries:
		chunk_type = chunk_entry.chunk_type
		print(f"\n\n========== Chunk type : {chunk_type} ==========")
		print("\n\nChunking")
		chunking_args = make_manifest_chunking_args(chunk_entry, manifest.embedding_model)

		s5 = time.time()
		chunks = Chunking(chunking_args).load_saved_chunks(documents, chunk_entry.chunk_path)
		e5 = time.time() - s5
		print("\nChunking time : ", e5)

		# The index settings of this chunk type, from every index the build made of it
		chunk_type_index_settings = {}
		for index_entry in manifest.index_entries:
			if index_entry.chunk_type != chunk_type:
				continue
			for setting_name in index_entry.index_settings:
				chunk_type_index_settings[setting_name] = index_entry.index_settings[setting_name]

		# The retrieval settings of this chunk type. Each setup gets a copy with its one retrieval type. Every result of
		# this chunk type is collected in retrieved_items
		retrieval_args = make_retrieval_args(args, chunk_type)
		dense_retrieval_args = dataclasses.replace(retrieval_args, retrieval_types=[DENSE_RETRIEVAL])
		bm25_retrieval_args = dataclasses.replace(retrieval_args, retrieval_types=[BM25_RETRIEVAL])
		hybrid_retrieval_args = dataclasses.replace(retrieval_args, retrieval_types=[HYBRID_RETRIEVAL])
		retrieved_items = []

		# 7. The BM25 index of this chunk type, loaded once and kept for BM25 and hybrid retrieval
		bm25_index = None
		for index_entry in manifest.index_entries:
			if index_entry.chunk_type != chunk_type or index_entry.index_type != BM25_INDEX:
				continue
			print(f"\n\nIndexing : {BM25_INDEX}")
			index_args = make_manifest_index_args(chunking_args, BM25_INDEX, chunk_type_index_settings, manifest.input_hash)

			s7 = time.time()
			bm25_index = Indexing(index_args).load_saved_index(chunks, index_entry.location)
			e7 = time.time() - s7
			print("\nIndexing time : ", e7)

		# 8. BM25 retrieval: one Retrieval per query type, each query retrieved one at a time. HyDE is searched with
		# dense only
		if bm25_index is not None and BM25_RETRIEVAL in retrieval_args.retrieval_types:
			print(f"\n\nRetrieval : {BM25_RETRIEVAL} {BM25_INDEX}")
			s8 = time.time()
			for search_query_args in search_query_args_list:
				if search_query_args.query_types == [HYDE_QUERY]:
					continue
				retrieval = Retrieval(bm25_retrieval_args, search_query_args, None, None, bm25_index, retrieved_path)
				for query in val_queries:
					retrieved_items.append(retrieval.retrieve(query))
			e8 = time.time() - s8
			print("\nRetrieval time : ", e8)

		# 9. Every dense index the build made of this chunk type, FAISS and Qdrant, one at a time: dense retrieval on
		# it, and hybrid retrieval on it with the BM25 index
		for index_entry in manifest.index_entries:
			if index_entry.chunk_type != chunk_type or index_entry.index_type == BM25_INDEX:
				continue
			index_type = index_entry.index_type

			s9 = time.time()
			faiss_index = None
			vectordb_index = None
			if index_type == QDRANT_INDEX:
				print(f"\n\nVector DB indexing : {index_type}")
				vectordb_args = make_vectordb_args(args, chunking_args, index_type, manifest.input_hash)
				vectordb_index = IndexingVectorDb(vectordb_args).load_saved_vectordb(chunks, index_entry.location)
			else:
				print(f"\n\nIndexing : {index_type}")
				index_args = make_manifest_index_args(chunking_args, index_type, chunk_type_index_settings, manifest.input_hash)
				faiss_index = Indexing(index_args).load_saved_index(chunks, index_entry.location)
			e9 = time.time() - s9
			print("\nIndexing time : ", e9)

			# Dense retrieval on this index, every query type
			if DENSE_RETRIEVAL in retrieval_args.retrieval_types:
				print(f"\n\nRetrieval : {DENSE_RETRIEVAL} {index_type}")
				s10 = time.time()
				for search_query_args in search_query_args_list:
					retrieval = Retrieval(dense_retrieval_args, search_query_args, faiss_index, vectordb_index, None, retrieved_path)
					for query in val_queries:
						retrieved_items.append(retrieval.retrieve(query))
				e10 = time.time() - s10
				print("\nRetrieval time : ", e10)

			# Hybrid retrieval on this index and the BM25 index. HyDE is searched with dense only
			if bm25_index is not None and HYBRID_RETRIEVAL in retrieval_args.retrieval_types:
				print(f"\n\nRetrieval : {HYBRID_RETRIEVAL} {index_type}+{BM25_INDEX}")
				s11 = time.time()
				for search_query_args in search_query_args_list:
					if search_query_args.query_types == [HYDE_QUERY]:
						continue
					retrieval = Retrieval(hybrid_retrieval_args, search_query_args, faiss_index, vectordb_index, bm25_index, retrieved_path)
					for query in val_queries:
						retrieved_items.append(retrieval.retrieve(query))
				e11 = time.time() - s11
				print("\nRetrieval time : ", e11)

		print(f"\nRetrieved : {len(retrieved_items)}")

		# 11. Retrieval evaluation of this chunk type's results, while its chunks are loaded
		print("\n\nRetrieval evaluation")
		s13 = time.time()
		chunk_type_retrieved_evals = retrieval_evaluation.compute_retrieved_evaluation(retrieved_items, chunks)
		retrieved_evals.extend(chunk_type_retrieved_evals)
		e13 = time.time() - s13
		print("\nRetrieval evaluation time : ", e13)

	print(f"\n\nRetrieved evals : {len(retrieved_evals)}")

	# 12. Rank every setup, keep the best n_top_setups and save them with every setting that made their results, named
	# by the build, eval dataset and benchmark hashes
	print("\n\nSetup ranking")
	best_setups = rank_setups(retrieved_evals, args.n_top_setups)
	print(f"Best setups : {len(best_setups)}")
	ollama_args = make_ollama_args(args)
	benchmark_hash = paths.get_benchmark_hash(make_benchmark_args(args, query_args), ollama_args)
	setup_records = save_best_setups(
		best_setups, manifest, args.manifest_path, make_retrieval_args(args, ""), query_args, output_path,
		eval_dataset_metadata.eval_dataset_hash, benchmark_hash
	)

	# 13. The pipeline of the best setup the config names, run on every test query. It loads the setup's indexes by
	# path and reuses the query and retrieved caches
	print("\n\nPipeline")
	s14 = time.time()
	if args.setup_rank < 1 or args.setup_rank > len(setup_records):
		raise ValueError(f"setup_rank ({args.setup_rank}) must be between 1 and the number of best setups ({len(setup_records)}).")
	setup_record = setup_records[args.setup_rank - 1]
	pipeline_args = make_pipeline_args(
		setup_record, query_path, retrieved_path, args.query_llm_timeout_seconds, args.ollama_api_base, args.qdrant_url
	)
	pipeline = Pipeline(pipeline_args)

	eval_samples_by_id = {}
	for eval_sample in eval_samples:
		eval_samples_by_id[eval_sample.id] = eval_sample
	pipeline_results = []
	for test_query in test_queries:
		eval_sample = eval_samples_by_id[test_query.id]
		pipeline_results.append(pipeline.invoke(eval_sample.id, eval_sample.qus))
	e14 = time.time() - s14
	print(f"Pipeline results : {len(pipeline_results)}")
	print("\nPipeline time : ", e14)


if __name__ == "__main__":
	print("\n ------------- Benchmark -------- \n")
	start = time.time()

	"""
	Input: --config and any override flags from the command line.
	Output: nothing.
	What it does: parses the benchmark config and runs the benchmark.
	"""
	args = parse_benchmark_config()
	run_benchmark(args)

	elapsed_time = time.time() - start
	print("\n\nElapsed time : ", elapsed_time)
