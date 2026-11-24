import argparse
import hashlib
import json
import os

from common.args import ChunkingArgs
from common.args import EmbedderArgs
from common.args import IndexArgs
from common.args import VectorDbArgs
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
def get_chunk_path(output_path: str, embedder_args: EmbedderArgs, chunking_args: ChunkingArgs) -> str:
	"""
	Input: output_path, the dataset's output folder; embedder_args; chunking_args.
	Output: {output_path}/chunks/chunks_{hash}
	What it does: builds the folder that holds one chunking run's chunks. The hash is the first 12 hex characters of
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
def get_embedding_path(output_path: str, embedder_args: EmbedderArgs, chunking_args: ChunkingArgs) -> str:
	"""
	Input: output_path, the dataset's output folder; embedder_args; chunking_args.
	Output: {output_path}/embeddings/embeddings_{hash}
	What it does: builds the folder that holds one chunking run's embeddings. The hash is the first 12 hex characters
	of the SHA-256 of the embedding provider, the model name and the chunk args, as JSON with sorted keys.
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
def get_faiss_index_path(output_path: str, embedder_args: EmbedderArgs, chunking_args: ChunkingArgs, index_args: IndexArgs) -> str:
	"""
	Input: output_path, the dataset's output folder; embedder_args; chunking_args; index_args.
	Output: {output_path}/indexes/{index_type}_{hash}
	What it does: builds the folder that holds one FAISS index. The hash is the first 12 hex characters of the SHA-256
	of the index type, the embedding provider, the model name, the chunk args, the index type's settings (ivf_nlist_multiplier,
	or hnsw_m and hnsw_ef_construction) and the input hash, as JSON with sorted keys.
	"""
	hashed_args = {}
	hashed_args["index_type"] = index_args.index_type
	hashed_args["input_hash"] = index_args.input_hash
	hashed_args["embedding_provider"] = embedder_args.embedding_provider
	hashed_args["model_name"] = embedder_args.model_name
	hashed_args["chunk_type"] = chunking_args.chunk_type
	hashed_args["min_chunk_size"] = chunking_args.min_chunk_size
	hashed_args["max_chunk_size"] = chunking_args.max_chunk_size
	hashed_args["fixed_chunk_overlap"] = chunking_args.fixed_chunk_overlap

	if index_args.index_type == FAISS_FLATIP_INDEX:
		pass
	elif index_args.index_type == FAISS_IVF_INDEX:
		hashed_args["ivf_nlist_multiplier"] = index_args.ivf_nlist_multiplier
	elif index_args.index_type == FAISS_HNSW_INDEX:
		hashed_args["hnsw_m"] = index_args.hnsw_m
		hashed_args["hnsw_ef_construction"] = index_args.hnsw_ef_construction
	else:
		raise ValueError(f"index_type ({index_args.index_type}) is not a FAISS index type.")

	hashed_args_json = json.dumps(hashed_args, sort_keys=True)
	index_hash = hashlib.sha256(hashed_args_json.encode("utf-8")).hexdigest()[:12]
	return os.path.join(output_path, "indexes", f"{index_args.index_type}_{index_hash}")


def get_bm25_index_path(output_path: str, embedder_args: EmbedderArgs, chunking_args: ChunkingArgs, index_args: IndexArgs) -> str:
	"""
	Input: output_path, the dataset's output folder; embedder_args; chunking_args; index_args.
	Output: {output_path}/indexes/bm25_{hash}
	What it does: builds the folder that holds one BM25 index. The hash is the first 12 hex characters of the SHA-256
	of the embedding provider, the model name, the chunk args, bm25_k1, bm25_b, bm25_tokenizer and the input hash, as
	JSON with sorted keys.
	"""
	hashed_args = {}
	hashed_args["input_hash"] = index_args.input_hash
	hashed_args["embedding_provider"] = embedder_args.embedding_provider
	hashed_args["model_name"] = embedder_args.model_name
	hashed_args["chunk_type"] = chunking_args.chunk_type
	hashed_args["min_chunk_size"] = chunking_args.min_chunk_size
	hashed_args["max_chunk_size"] = chunking_args.max_chunk_size
	hashed_args["fixed_chunk_overlap"] = chunking_args.fixed_chunk_overlap
	hashed_args["bm25_k1"] = index_args.bm25_k1
	hashed_args["bm25_b"] = index_args.bm25_b
	hashed_args["bm25_tokenizer"] = index_args.bm25_tokenizer

	hashed_args_json = json.dumps(hashed_args, sort_keys=True)
	index_hash = hashlib.sha256(hashed_args_json.encode("utf-8")).hexdigest()[:12]
	return os.path.join(output_path, "indexes", f"bm25_{index_hash}")


# ------ Vector DB
def get_qdrant_collection_name(output_path: str, embedder_args: EmbedderArgs, chunking_args: ChunkingArgs, vectordb_args: VectorDbArgs) -> str:
	"""
	Input: output_path, the dataset's output folder ({root}/{dataset_name}); embedder_args; chunking_args;
	vectordb_args.
	Output: qdrant_{hash}
	What it does: builds the Qdrant collection name. The hash is the first 12 hex characters of the SHA-256 of the
	root folder name, the dataset name, the embedding provider, the model name, the chunk args and the input hash, as
	JSON with sorted keys. The root folder name keeps a temp/ run and an outputs/ run of the same dataset in separate
	collections.
	"""
	normalized_output_path = os.path.normpath(output_path)
	dataset_name = os.path.basename(normalized_output_path)
	root_folder_name = os.path.basename(os.path.dirname(normalized_output_path))

	hashed_args = {}
	hashed_args["root_folder_name"] = root_folder_name
	hashed_args["dataset_name"] = dataset_name
	hashed_args["input_hash"] = vectordb_args.input_hash
	hashed_args["embedding_provider"] = embedder_args.embedding_provider
	hashed_args["model_name"] = embedder_args.model_name
	hashed_args["chunk_type"] = chunking_args.chunk_type
	hashed_args["min_chunk_size"] = chunking_args.min_chunk_size
	hashed_args["max_chunk_size"] = chunking_args.max_chunk_size
	hashed_args["fixed_chunk_overlap"] = chunking_args.fixed_chunk_overlap

	hashed_args_json = json.dumps(hashed_args, sort_keys=True)
	collection_hash = hashlib.sha256(hashed_args_json.encode("utf-8")).hexdigest()[:12]
	return f"qdrant_{collection_hash}"


# ------ Build manifest
def get_build_hash(parsed_args: argparse.Namespace, input_hash: str) -> str:
	"""
	Input: parsed_args, the parsed build args: every setting of the ingestion, embedder, chunking, indexing and
	vectordb sections, with any command line overrides; input_hash, the hash of the input files.
	Output: the first 12 hex characters of the SHA-256 of all the build args and the input hash, as JSON with sorted keys.
	What it does: hashes every build arg and the input hash, so a change to any of them gives a new build hash.
	"""
	hashed_args = {}
	for arg_name in vars(parsed_args):
		hashed_args[arg_name] = getattr(parsed_args, arg_name)
	hashed_args["input_hash"] = input_hash

	hashed_args_json = json.dumps(hashed_args, sort_keys=True)
	return hashlib.sha256(hashed_args_json.encode("utf-8")).hexdigest()[:12]


def get_build_manifest_path(output_path: str, parsed_args: argparse.Namespace, input_hash: str) -> str:
	"""
	Input: output_path, the dataset's output folder; parsed_args, the parsed build args; input_hash, the hash of the
	input files.
	Output: {output_path}/build/manifest_{dataset_name}_{dataset_size}_{hash}.json
	What it does: builds the file the build saves its manifest to, named by the dataset name and size, so it is
	readable without opening it, and the hash of all the build args and the input hash.
	"""
	build_hash = get_build_hash(parsed_args, input_hash)
	manifest_file_name = f"manifest_{parsed_args.dataset_name}_{parsed_args.dataset_size}_{build_hash}.json"
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
