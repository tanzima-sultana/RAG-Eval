import argparse
import dataclasses
import hashlib
import json
import os

from common.args import ChunkingArgs
from common.args import EmbedderArgs
from common.args import Bm25IndexingArgs
from common.args import FaissHNSWArgs
from common.args import FaissIVFArgs
from common.args import EvalDatasetArgs
from common.args import OllamaArgs
from common.args import QueryArgs
from common.args import BenchmarkArgs
from common.args import RetrievalArgs
from common.constants import FAISS_FLATIP_INDEX
from common.constants import FAISS_IVF_INDEX
from common.constants import FAISS_HNSW_INDEX


# ------ Dataset
def get_output_path(output_path: str, dataset_name: str) -> str:
	"""
	Input: output_path, the root output folder; dataset_name.
	Output: {output_path}/{dataset_name}
	What it does: builds the dataset's output folder.
	"""
	return os.path.join(output_path, dataset_name)


# ------ Ingestion
def get_documents_path(output_path: str) -> str:
	"""
	Input: output_path, the dataset's output folder.
	Output: {output_path}/documents
	What it does: builds the folder that holds the ingested documents.
	"""
	return os.path.join(output_path, "documents")


# ------ Chunking
def get_chunking_hash(embedder_args: EmbedderArgs, chunking_args: ChunkingArgs, ingestion_hash: str) -> str:
	"""
	Input: embedder_args; chunking_args; ingestion_hash, the hash of the input files the chunks are made from.
	Output: the first 12 hex characters of the SHA-256 of the ingestion hash, the embedding provider, the model name
	and the chunk args, as JSON with sorted keys.
	What it does: names one chunking run. It includes the ingestion hash, so a change upstream gives a new chunking hash.
	"""
	hashed_args = {}
	hashed_args["ingestion_hash"] = ingestion_hash
	hashed_args["embedding_provider"] = embedder_args.embedding_provider
	hashed_args["model_name"] = embedder_args.model_name
	hashed_args["chunk_type"] = chunking_args.chunk_type
	hashed_args["min_chunk_size"] = chunking_args.min_chunk_size
	hashed_args["max_chunk_size"] = chunking_args.max_chunk_size
	hashed_args["fixed_chunk_overlap"] = chunking_args.fixed_chunk_overlap

	hashed_args_json = json.dumps(hashed_args, sort_keys=True)
	return hashlib.sha256(hashed_args_json.encode("utf-8")).hexdigest()[:12]


def get_chunk_path(output_path: str, embedder_args: EmbedderArgs, chunking_args: ChunkingArgs) -> str:
	"""
	Input: output_path, the dataset's output folder; embedder_args; chunking_args.
	Output: {output_path}/chunks/chunks_{hash}
	What it does: builds the folder that holds the chunks made with these settings, one file per document. The folder is
	shared by every run with these settings, whatever input files they read. The hash is the first 12 hex characters of
	the SHA-256 of the embedding provider, the model name and the chunk args, as JSON with sorted keys.
	"""
	hashed_args = {}
	hashed_args["embedding_provider"] = embedder_args.embedding_provider
	hashed_args["model_name"] = embedder_args.model_name
	hashed_args["chunk_type"] = chunking_args.chunk_type
	hashed_args["min_chunk_size"] = chunking_args.min_chunk_size
	hashed_args["max_chunk_size"] = chunking_args.max_chunk_size
	hashed_args["fixed_chunk_overlap"] = chunking_args.fixed_chunk_overlap

	hashed_args_json = json.dumps(hashed_args, sort_keys=True)
	chunk_hash = hashlib.sha256(hashed_args_json.encode("utf-8")).hexdigest()[:12]
	return os.path.join(output_path, "chunks", f"chunks_{chunk_hash}")


# ------ Embedding
def get_embedding_hash(embedder_args: EmbedderArgs, chunking_hash: str) -> str:
	"""
	Input: embedder_args; chunking_hash, the hash of the chunking run the chunks are from.
	Output: the first 12 hex characters of the SHA-256 of the chunking hash, the embedding provider and the model name,
	as JSON with sorted keys.
	What it does: names one embedding run. It includes the chunking hash, so a change upstream gives a new embedding hash.
	"""
	hashed_args = {}
	hashed_args["chunking_hash"] = chunking_hash
	hashed_args["embedding_provider"] = embedder_args.embedding_provider
	hashed_args["model_name"] = embedder_args.model_name

	hashed_args_json = json.dumps(hashed_args, sort_keys=True)
	return hashlib.sha256(hashed_args_json.encode("utf-8")).hexdigest()[:12]


def get_embedding_path(output_path: str, embedder_args: EmbedderArgs, chunking_args: ChunkingArgs) -> str:
	"""
	Input: output_path, the dataset's output folder; embedder_args; chunking_args.
	Output: {output_path}/embeddings/embeddings_{hash}
	What it does: builds the folder that holds the embeddings made with these settings, one file per document. The
	folder is shared by every run with these settings. The hash is the first 12 hex characters of the SHA-256 of the
	embedding provider, the model name and the chunk args, as JSON with sorted keys.
	"""
	hashed_args = {}
	hashed_args["embedding_provider"] = embedder_args.embedding_provider
	hashed_args["model_name"] = embedder_args.model_name
	hashed_args["chunk_type"] = chunking_args.chunk_type
	hashed_args["min_chunk_size"] = chunking_args.min_chunk_size
	hashed_args["max_chunk_size"] = chunking_args.max_chunk_size
	hashed_args["fixed_chunk_overlap"] = chunking_args.fixed_chunk_overlap

	hashed_args_json = json.dumps(hashed_args, sort_keys=True)
	embedding_hash = hashlib.sha256(hashed_args_json.encode("utf-8")).hexdigest()[:12]
	return os.path.join(output_path, "embeddings", f"embeddings_{embedding_hash}")


# ------ Indexing
def get_faiss_hash(faiss_type: str, faiss_args: FaissIVFArgs | FaissHNSWArgs | None, embedding_hash: str) -> str:
	"""
	Input: faiss_type, one of faiss_types; faiss_args, the FaissIVFArgs of faiss_ivf, the FaissHNSWArgs of faiss_hnsw, or
	None for faiss_flatip; embedding_hash, from embedding.
	Output: the first 12 hex characters of the SHA-256 of the embedding hash, the FAISS type and that type's settings
	(ivf_nlist_multiplier, or hnsw_m and hnsw_ef_construction), as JSON with sorted keys.
	What it does: names one FAISS index. It changes when the embeddings or that index's settings change.
	"""
	hashed_args = {}
	hashed_args["embedding_hash"] = embedding_hash
	hashed_args["faiss_type"] = faiss_type

	if faiss_type == FAISS_FLATIP_INDEX:
		pass
	elif faiss_type == FAISS_IVF_INDEX:
		hashed_args["ivf_nlist_multiplier"] = faiss_args.ivf_nlist_multiplier
	elif faiss_type == FAISS_HNSW_INDEX:
		hashed_args["hnsw_m"] = faiss_args.hnsw_m
		hashed_args["hnsw_ef_construction"] = faiss_args.hnsw_ef_construction
	else:
		raise ValueError(f"faiss_type ({faiss_type}) is not a FAISS index type.")

	hashed_args_json = json.dumps(hashed_args, sort_keys=True)
	return hashlib.sha256(hashed_args_json.encode("utf-8")).hexdigest()[:12]


def get_faiss_index_path(output_path: str, faiss_type: str, faiss_hash: str) -> str:
	"""
	Input: output_path, the dataset's output folder; faiss_type; faiss_hash, from get_faiss_hash.
	Output: {output_path}/indexes/{faiss_type}_{faiss_hash}
	What it does: builds the folder that holds one FAISS index.
	"""
	return os.path.join(output_path, "indexes", f"{faiss_type}_{faiss_hash}")


def get_bm25_hash(bm25_args: Bm25IndexingArgs, chunking_hash: str) -> str:
	"""
	Input: bm25_args, the BM25 indexing settings; chunking_hash, from chunking.
	Output: the first 12 hex characters of the SHA-256 of the chunking hash, bm25_k1, bm25_b and bm25_tokenizer, as JSON
	with sorted keys.
	What it does: names the BM25 index. It changes when the chunks or a BM25 setting change. BM25 reads only the chunk
	texts, so the embedding hash is not part of it.
	"""
	hashed_args = {}
	hashed_args["chunking_hash"] = chunking_hash
	hashed_args["bm25_k1"] = bm25_args.bm25_k1
	hashed_args["bm25_b"] = bm25_args.bm25_b
	hashed_args["bm25_tokenizer"] = bm25_args.bm25_tokenizer

	hashed_args_json = json.dumps(hashed_args, sort_keys=True)
	return hashlib.sha256(hashed_args_json.encode("utf-8")).hexdigest()[:12]


def get_bm25_index_path(output_path: str, bm25_hash: str) -> str:
	"""
	Input: output_path, the dataset's output folder; bm25_hash, from get_bm25_hash.
	Output: {output_path}/indexes/bm25_{bm25_hash}
	What it does: builds the folder that holds the BM25 index.
	"""
	return os.path.join(output_path, "indexes", f"bm25_{bm25_hash}")


# ------ Vector DB
def get_vectordb_hash(output_path: str, vectordb_type: str, embedding_hash: str) -> str:
	"""
	Input: output_path, the dataset's output folder ({root}/{dataset_name}); vectordb_type, e.g. qdrant; embedding_hash,
	from embedding.
	Output: the first 12 hex characters of the SHA-256 of the root folder name, the dataset name, the vector DB type and
	the embedding hash, as JSON with sorted keys.
	What it does: names one vector DB collection. The root folder name keeps a temp/ run and an outputs/ run of the same
	dataset in separate collections.
	"""
	normalized_output_path = os.path.normpath(output_path)
	dataset_name = os.path.basename(normalized_output_path)
	root_folder_name = os.path.basename(os.path.dirname(normalized_output_path))

	hashed_args = {}
	hashed_args["root_folder_name"] = root_folder_name
	hashed_args["dataset_name"] = dataset_name
	hashed_args["vectordb_type"] = vectordb_type
	hashed_args["embedding_hash"] = embedding_hash

	hashed_args_json = json.dumps(hashed_args, sort_keys=True)
	return hashlib.sha256(hashed_args_json.encode("utf-8")).hexdigest()[:12]


def get_qdrant_collection_name(vectordb_type: str, vectordb_hash: str) -> str:
	"""
	Input: vectordb_type, e.g. qdrant; vectordb_hash, from get_vectordb_hash.
	Output: {vectordb_type}_{vectordb_hash}
	What it does: builds the vector DB collection name.
	"""
	return f"{vectordb_type}_{vectordb_hash}"


# ------ Build manifest
def get_build_hash(ingestion_hash: str, chunking_hash: str, embedding_hash: str, bm25_hash: str, faiss_hash: str, vectordb_hash: str) -> str:
	"""
	Input: ingestion_hash, chunking_hash, embedding_hash and bm25_hash, from their layers; faiss_hash and vectordb_hash,
	from indexing, with "" for the one that this setup does not have.
	Output: the first 12 hex characters of the SHA-256 of the six hashes, as JSON with sorted keys.
	What it does: names one setup. It changes when any layer's hash of that setup changes.
	"""
	hashed_args = {}
	hashed_args["ingestion_hash"] = ingestion_hash
	hashed_args["chunking_hash"] = chunking_hash
	hashed_args["embedding_hash"] = embedding_hash
	hashed_args["bm25_hash"] = bm25_hash
	hashed_args["faiss_hash"] = faiss_hash
	hashed_args["vectordb_hash"] = vectordb_hash

	hashed_args_json = json.dumps(hashed_args, sort_keys=True)
	return hashlib.sha256(hashed_args_json.encode("utf-8")).hexdigest()[:12]


def get_build_manifest_path(output_path: str, dataset_name: str, dataset_size: int, build_hashes: list[str]) -> str:
	"""
	Input: output_path, the dataset's output folder; dataset_name; dataset_size; build_hashes, the build hash of every
	setup of the run, in the order they were built.
	Output: {output_path}/build/{dataset_name}_{dataset_size}_{hash}.json
	What it does: builds the file the build saves its manifests to. The hash is the first 12 hex characters of the
	SHA-256 of the list of build hashes, so the file name tells the dataset and its size and changes when the set of
	setups changes.
	"""
	build_hashes_json = json.dumps(build_hashes)
	manifests_hash = hashlib.sha256(build_hashes_json.encode("utf-8")).hexdigest()[:12]
	manifest_file_name = f"{dataset_name}_{dataset_size}_{manifests_hash}.json"
	return os.path.join(output_path, "build", manifest_file_name)


# ------ Setup ranking
def get_best_setups_path(output_path: str, build_hash: str, eval_dataset_hash: str, benchmark_hash: str) -> str:
	"""
	Input: output_path, the dataset's output folder; build_hash, from the build manifest; eval_dataset_hash, from the
	eval dataset metadata; benchmark_hash, from get_benchmark_hash.
	Output: {output_path}/best_setups/best_setups_{build_hash}_{eval_dataset_hash}_{benchmark_hash}.json
	What it does: builds the file the benchmark saves its best setups to, named by the three hashes.
	"""
	file_name = f"best_setups_{build_hash}_{eval_dataset_hash}_{benchmark_hash}.json"
	return os.path.join(output_path, "best_setups", file_name)


def get_benchmark_hash(benchmark_args: BenchmarkArgs, ollama_args: OllamaArgs) -> str:
	"""
	Input: benchmark_args, the benchmark settings that change its results; ollama_args.
	Output: the first 12 hex characters of the SHA-256 of those settings.
	What it does: hashes the query settings, the retrieval and ranking settings, as JSON with sorted keys. The Ollama
	context length is included only when the query LLM is an Ollama model.
	"""
	hashed_args = make_query_hashed_args(benchmark_args.query_args)
	hashed_args["retrieval_types"] = benchmark_args.retrieval_types
	hashed_args["top_k"] = benchmark_args.top_k
	hashed_args["reranking"] = benchmark_args.reranking
	hashed_args["rerank_top_k"] = benchmark_args.rerank_top_k
	hashed_args["cross_encoder_model"] = benchmark_args.cross_encoder_model
	hashed_args["n_top_setups"] = benchmark_args.n_top_setups

	ollama_providers = ["ollama", "ollama_chat"]
	query_llm_provider = benchmark_args.query_args.query_llm_args.llm_model.split("/", 1)[0]
	if query_llm_provider in ollama_providers:
		hashed_args["ollama_context_length"] = ollama_args.ollama_context_length

	hashed_args_json = json.dumps(hashed_args, sort_keys=True)
	return hashlib.sha256(hashed_args_json.encode("utf-8")).hexdigest()[:12]


# ------ Eval dataset
def get_eval_dataset_hash(eval_dataset_args: EvalDatasetArgs, ollama_args: OllamaArgs) -> str:
	"""
	Input: eval_dataset_args; ollama_args.
	Output: the first 12 hex characters of the SHA-256 of the eval dataset args.
	What it does: hashes every arg that shapes the eval dataset, as JSON with sorted keys. The Ollama context length
	is included only when one of the two LLMs is an Ollama model.
	"""
	hashed_args = {}
	hashed_args["dataset_name"] = eval_dataset_args.dataset_name
	hashed_args["dataset_size"] = eval_dataset_args.dataset_size
	hashed_args["documents_path"] = eval_dataset_args.documents_path
	hashed_args["eval_dataset_llm_model"] = eval_dataset_args.eval_dataset_llm_args.llm_model
	hashed_args["eval_dataset_llm_max_tokens"] = eval_dataset_args.eval_dataset_llm_args.max_tokens
	hashed_args["eval_dataset_kg_llm_model"] = eval_dataset_args.eval_dataset_kg_llm_args.llm_model
	hashed_args["eval_dataset_kg_llm_max_tokens"] = eval_dataset_args.eval_dataset_kg_llm_args.max_tokens
	hashed_args["embedding_model"] = eval_dataset_args.eval_dataset_embedding_model
	hashed_args["n_sample_documents"] = eval_dataset_args.n_sample_documents
	hashed_args["n_evals_requested"] = eval_dataset_args.n_evals_requested
	hashed_args["sample_documents_seed"] = eval_dataset_args.sample_documents_seed
	hashed_args["min_document_tokens"] = eval_dataset_args.min_document_tokens
	hashed_args["passage_match_chars"] = eval_dataset_args.passage_match_chars

	ollama_providers = ["ollama", "ollama_chat"]
	eval_dataset_llm_provider = eval_dataset_args.eval_dataset_llm_args.llm_model.split("/", 1)[0]
	eval_dataset_kg_llm_provider = eval_dataset_args.eval_dataset_kg_llm_args.llm_model.split("/", 1)[0]
	if eval_dataset_llm_provider in ollama_providers or eval_dataset_kg_llm_provider in ollama_providers:
		hashed_args["ollama_context_length"] = ollama_args.ollama_context_length

	hashed_args_json = json.dumps(hashed_args, sort_keys=True)
	return hashlib.sha256(hashed_args_json.encode("utf-8")).hexdigest()[:12]


def get_eval_dataset_path(output_path: str, eval_dataset_args: EvalDatasetArgs, ollama_args: OllamaArgs) -> str:
	"""
	Input: output_path, the dataset's output folder; eval_dataset_args; ollama_args.
	Output: {output_path}/eval_dataset/eval_dataset_{hash}.json
	What it does: builds the eval dataset file path from the hash of its args.
	"""
	eval_dataset_hash = get_eval_dataset_hash(eval_dataset_args, ollama_args)
	return os.path.join(output_path, "eval_dataset", f"eval_dataset_{eval_dataset_hash}.json")


def get_eval_dataset_metadata_path(output_path: str, eval_dataset_args: EvalDatasetArgs, ollama_args: OllamaArgs) -> str:
	"""
	Input: output_path, the dataset's output folder; eval_dataset_args; ollama_args.
	Output: {output_path}/eval_dataset/eval_dataset_{hash}_metadata.json
	What it does: builds the eval dataset metadata file path from the hash of its args.
	"""
	eval_dataset_hash = get_eval_dataset_hash(eval_dataset_args, ollama_args)
	return os.path.join(output_path, "eval_dataset", f"eval_dataset_{eval_dataset_hash}_metadata.json")


def get_eval_dataset_kg_path(output_path: str, eval_dataset_args: EvalDatasetArgs, ollama_args: OllamaArgs) -> str:
	"""
	Input: output_path, the dataset's output folder; eval_dataset_args; ollama_args.
	Output: {output_path}/eval_dataset/kg_{hash}.json
	What it does: builds the RAGAS knowledge graph file path from the same hash as its eval dataset.
	"""
	eval_dataset_hash = get_eval_dataset_hash(eval_dataset_args, ollama_args)
	return os.path.join(output_path, "eval_dataset", f"kg_{eval_dataset_hash}.json")


# ------ Query transformation
def make_query_hashed_args(query_args: QueryArgs) -> dict:
	"""
	Input: query_args.
	Output: a dict of every query setting that shapes the queries, keyed by name.
	What it does: collects each QueryArgs field and the query LLM's model and max_tokens. The LLM timeout is left out,
	since it does not change the queries.
	"""
	hashed_args = {}
	hashed_args["query_llm_model"] = query_args.query_llm_args.llm_model
	hashed_args["query_llm_max_tokens"] = query_args.query_llm_args.max_tokens
	hashed_args["n_multi_qus"] = query_args.n_multi_qus
	hashed_args["query_types"] = query_args.query_types
	hashed_args["query_split"] = query_args.query_split
	hashed_args["val_queries_size"] = query_args.val_queries_size
	hashed_args["test_queries_size"] = query_args.test_queries_size
	return hashed_args


def get_query_hash(query_args: QueryArgs, query_text: str) -> str:
	"""
	Input: query_args, the settings of one query type; query_text, the text the query is looked up by: the question
	for original, rewritten and hyde, one phrasing for multi.
	Output: the first 12 hex characters of the SHA-256 of the text and the settings that shape the query.
	What it does: hashes the text, the query type, the query LLM's model and max_tokens, n_multi_qus and the Ollama
	context length, as JSON with sorted keys. The LLM timeout and the split settings are left out, since they do not
	change the query.
	"""
	hashed_args = {}
	hashed_args["query_text"] = query_text
	hashed_args["query_type"] = query_args.query_type
	hashed_args["query_llm_model"] = query_args.query_llm_args.llm_model
	hashed_args["query_llm_max_tokens"] = query_args.query_llm_args.max_tokens
	hashed_args["n_multi_qus"] = query_args.n_multi_qus
	hashed_args["ollama_context_length"] = query_args.ollama_context_length

	hashed_args_json = json.dumps(hashed_args, sort_keys=True)
	return hashlib.sha256(hashed_args_json.encode("utf-8")).hexdigest()[:12]


def get_query_file_path(query_path: str, query_hash: str) -> str:
	"""
	Input: query_path, the folder of the saved queries; query_hash, from get_query_hash.
	Output: {query_path}/{query_hash}.json
	What it does: builds the file one query is saved to.
	"""
	return os.path.join(query_path, f"{query_hash}.json")


# ------ Retrieval
def get_retrieved_file_path(retrieved_path: str, retrieval_args: RetrievalArgs, query_type: str, index_type: str, index_paths: list[str], query_texts: list[str], original_qus: str) -> str:
	"""
	Input: retrieved_path, the folder of the retrieved files; retrieval_args, the setup's retrieval settings; query_type;
	index_type; index_paths, the paths of the indexes searched; query_texts, the texts searched; original_qus, the
	original question.
	Output: {retrieved_path}/{hash}.json
	What it does: names the file by a hash of what the result is made from: the texts, the setup and the indexes it
	searched, plus the original question and rerank settings when reranking. An index path is named by a hash of the
	chunk and index settings it was built with, so it stands for them.
	"""
	hashed_args = {}
	hashed_args["query_texts"] = query_texts
	hashed_args["query_type"] = query_type
	hashed_args["retrieval_type"] = retrieval_args.retrieval_types[0]
	hashed_args["top_k"] = retrieval_args.top_k
	hashed_args["index_type"] = index_type
	hashed_args["index_paths"] = index_paths
	hashed_args["reranking"] = retrieval_args.reranking

	if retrieval_args.reranking:
		hashed_args["original_qus"] = original_qus
		hashed_args["rerank_top_k"] = retrieval_args.rerank_top_k
		hashed_args["cross_encoder_model"] = retrieval_args.cross_encoder_model

	hashed_args_json = json.dumps(hashed_args, sort_keys=True)
	retrieved_hash = hashlib.sha256(hashed_args_json.encode("utf-8")).hexdigest()[:12]
	return os.path.join(retrieved_path, f"{retrieved_hash}.json")
