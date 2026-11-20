import argparse
import hashlib
import json
import os

from common.args import ChunkingArgs
from common.args import EmbeddingArgs
from common.args import IndexArgs
from common.args import VectorDbArgs
from common.args import EvalDatasetArgs
from common.args import OllamaArgs
from common.args import QueryArgs
from common.args import RetrievalArgs
from common.constants import FAISS_FLATIP_INDEX
from common.constants import FAISS_IVF_INDEX
from common.constants import FAISS_HNSW_INDEX
from common.constants import BM25_INDEX
from common.constants import QDRANT_INDEX


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
def get_chunk_path(output_path: str, chunking_args: ChunkingArgs) -> str:
	"""
	Input: output_path, the dataset's output folder; chunking_args.
	Output: {output_path}/chunks/chunks_{hash}
	What it does: builds the folder that holds one chunking run's chunks. The hash is the first 12 hex characters of
	the SHA-256 of the chunk args, as JSON with sorted keys.
	"""
	hashed_args = {}
	hashed_args["chunk_type"] = chunking_args.chunk_type
	hashed_args["model_name"] = chunking_args.model_name
	hashed_args["min_chunk_size"] = chunking_args.min_chunk_size
	hashed_args["max_chunk_size"] = chunking_args.max_chunk_size
	hashed_args["fixed_chunk_overlap"] = chunking_args.fixed_chunk_overlap
	hashed_args["semantic_threshold"] = chunking_args.semantic_threshold

	hashed_args_json = json.dumps(hashed_args, sort_keys=True)
	chunk_hash = hashlib.sha256(hashed_args_json.encode("utf-8")).hexdigest()[:12]
	return os.path.join(output_path, "chunks", f"chunks_{chunk_hash}")


# ------ Embedding
def get_embedding_path(output_path: str, embedding_args: EmbeddingArgs) -> str:
	"""
	Input: output_path, the dataset's output folder; embedding_args.
	Output: {output_path}/embeddings/embeddings_{hash}
	What it does: builds the folder that holds one chunking run's embeddings. The hash is the first 12 hex characters
	of the SHA-256 of the chunk args, as JSON with sorted keys.
	"""
	chunking_args = embedding_args.chunking_args
	hashed_args = {}
	hashed_args["chunk_type"] = chunking_args.chunk_type
	hashed_args["model_name"] = chunking_args.model_name
	hashed_args["min_chunk_size"] = chunking_args.min_chunk_size
	hashed_args["max_chunk_size"] = chunking_args.max_chunk_size
	hashed_args["fixed_chunk_overlap"] = chunking_args.fixed_chunk_overlap
	hashed_args["semantic_threshold"] = chunking_args.semantic_threshold

	hashed_args_json = json.dumps(hashed_args, sort_keys=True)
	embedding_hash = hashlib.sha256(hashed_args_json.encode("utf-8")).hexdigest()[:12]
	return os.path.join(output_path, "embeddings", f"embeddings_{embedding_hash}")


# ------ Indexing
def get_faiss_index_path(output_path: str, index_args: IndexArgs) -> str:
	"""
	Input: output_path, the dataset's output folder; index_args.
	Output: {output_path}/indexes/{index_type}_{hash}
	What it does: builds the folder that holds one FAISS index. The hash is the first 12 hex characters of the SHA-256
	of the index type, the chunk args, the index type's settings (ivf_nlist, or hnsw_m and hnsw_ef_construction) and
	the input hash, as JSON with sorted keys.
	"""
	chunking_args = index_args.chunking_args
	hashed_args = {}
	hashed_args["index_type"] = index_args.index_type
	hashed_args["input_hash"] = index_args.input_hash
	hashed_args["chunk_type"] = chunking_args.chunk_type
	hashed_args["model_name"] = chunking_args.model_name
	hashed_args["min_chunk_size"] = chunking_args.min_chunk_size
	hashed_args["max_chunk_size"] = chunking_args.max_chunk_size
	hashed_args["fixed_chunk_overlap"] = chunking_args.fixed_chunk_overlap
	hashed_args["semantic_threshold"] = chunking_args.semantic_threshold

	if index_args.index_type == FAISS_FLATIP_INDEX:
		pass
	elif index_args.index_type == FAISS_IVF_INDEX:
		hashed_args["ivf_nlist"] = index_args.ivf_nlist
	elif index_args.index_type == FAISS_HNSW_INDEX:
		hashed_args["hnsw_m"] = index_args.hnsw_m
		hashed_args["hnsw_ef_construction"] = index_args.hnsw_ef_construction
	else:
		raise ValueError(f"index_type ({index_args.index_type}) is not a FAISS index type.")

	hashed_args_json = json.dumps(hashed_args, sort_keys=True)
	index_hash = hashlib.sha256(hashed_args_json.encode("utf-8")).hexdigest()[:12]
	return os.path.join(output_path, "indexes", f"{index_args.index_type}_{index_hash}")


def get_bm25_index_path(output_path: str, index_args: IndexArgs) -> str:
	"""
	Input: output_path, the dataset's output folder; index_args.
	Output: {output_path}/indexes/bm25_{hash}
	What it does: builds the folder that holds one BM25 index. The hash is the first 12 hex characters of the SHA-256
	of the chunk args, bm25_k1, bm25_b, bm25_tokenizer and the input hash, as JSON with sorted keys.
	"""
	chunking_args = index_args.chunking_args
	hashed_args = {}
	hashed_args["input_hash"] = index_args.input_hash
	hashed_args["chunk_type"] = chunking_args.chunk_type
	hashed_args["model_name"] = chunking_args.model_name
	hashed_args["min_chunk_size"] = chunking_args.min_chunk_size
	hashed_args["max_chunk_size"] = chunking_args.max_chunk_size
	hashed_args["fixed_chunk_overlap"] = chunking_args.fixed_chunk_overlap
	hashed_args["semantic_threshold"] = chunking_args.semantic_threshold
	hashed_args["bm25_k1"] = index_args.bm25_k1
	hashed_args["bm25_b"] = index_args.bm25_b
	hashed_args["bm25_tokenizer"] = index_args.bm25_tokenizer

	hashed_args_json = json.dumps(hashed_args, sort_keys=True)
	index_hash = hashlib.sha256(hashed_args_json.encode("utf-8")).hexdigest()[:12]
	return os.path.join(output_path, "indexes", f"bm25_{index_hash}")


# ------ Vector DB
def get_qdrant_collection_name(output_path: str, vectordb_args: VectorDbArgs) -> str:
	"""
	Input: output_path, the dataset's output folder ({root}/{dataset_name}); vectordb_args.
	Output: qdrant_{hash}
	What it does: builds the Qdrant collection name. The hash is the first 12 hex characters of the SHA-256 of the
	root folder name, the dataset name, the chunk args and the input hash, as JSON with sorted keys. The root folder
	name keeps a temp/ run and an outputs/ run of the same dataset in separate collections.
	"""
	chunking_args = vectordb_args.chunking_args
	normalized_output_path = os.path.normpath(output_path)
	dataset_name = os.path.basename(normalized_output_path)
	root_folder_name = os.path.basename(os.path.dirname(normalized_output_path))

	hashed_args = {}
	hashed_args["root_folder_name"] = root_folder_name
	hashed_args["dataset_name"] = dataset_name
	hashed_args["input_hash"] = vectordb_args.input_hash
	hashed_args["chunk_type"] = chunking_args.chunk_type
	hashed_args["model_name"] = chunking_args.model_name
	hashed_args["min_chunk_size"] = chunking_args.min_chunk_size
	hashed_args["max_chunk_size"] = chunking_args.max_chunk_size
	hashed_args["fixed_chunk_overlap"] = chunking_args.fixed_chunk_overlap
	hashed_args["semantic_threshold"] = chunking_args.semantic_threshold

	hashed_args_json = json.dumps(hashed_args, sort_keys=True)
	collection_hash = hashlib.sha256(hashed_args_json.encode("utf-8")).hexdigest()[:12]
	return f"qdrant_{collection_hash}"


# ------ Build manifest
def get_build_hash(parsed_args: argparse.Namespace, input_hash: str) -> str:
	"""
	Input: parsed_args, the parsed build args: every setting of the ingestion, chunking, indexing and vectordb
	sections, with any command line overrides; input_hash, the hash of the input files.
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
	Output: {output_path}/build/manifest_{hash}.json
	What it does: builds the file the build saves its manifest to, named by the hash of all the build args and the
	input hash.
	"""
	build_hash = get_build_hash(parsed_args, input_hash)
	return os.path.join(output_path, "build", f"manifest_{build_hash}.json")


# ------ Setup ranking
def get_best_setups_path(output_path: str) -> str:
	"""
	Input: output_path, the dataset's output folder.
	Output: {output_path}/best_setups/best_setups.json
	What it does: builds the file the benchmark saves its best setups to.
	"""
	return os.path.join(output_path, "best_setups", "best_setups.json")


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
def get_query_path(output_path: str, eval_dataset_args: EvalDatasetArgs, query_args: QueryArgs) -> str:
	"""
	Input: output_path, the dataset's output folder; eval_dataset_args, the settings of the eval dataset the queries are made from;
	query_args.
	Output: {output_path}/queries/{n_sample_documents}_{n_evals_requested}_{eval_dataset_llm_model}_{eval_dataset_embedding_model}_
	{embedding_model}_{query_llm_model}_{n_multi_qus}
	What it does: builds the folder that holds one query run. Any "/" or ":" in a model name is replaced with "-".
	"""
	eval_dataset_llm_model = eval_dataset_args.eval_dataset_llm_args.llm_model.replace("/", "-").replace(":", "-")
	eval_dataset_embedding_model = eval_dataset_args.eval_dataset_embedding_model.replace("/", "-").replace(":", "-")
	embedding_model = query_args.embedding_model.replace("/", "-").replace(":", "-")
	query_llm_model = query_args.query_llm_args.llm_model.replace("/", "-").replace(":", "-")
	folder_name = (
		f"{eval_dataset_args.n_sample_documents}_{eval_dataset_args.n_evals_requested}_{eval_dataset_llm_model}_{eval_dataset_embedding_model}_"
		f"{embedding_model}_{query_llm_model}_{query_args.n_multi_qus}"
	)
	return os.path.join(output_path, "queries", folder_name)


# ------ Retrieval
def get_retrieved_path(output_path: str, retrieval_args: RetrievalArgs, query_args: QueryArgs, val_queries_count: int, index_type: str) -> str:
	"""
	Input: output_path, the dataset's output folder; retrieval_args; query_args, for the query types;
	val_queries_count, the number of val queries searched; index_type, one index type, or a dense and the
	BM25 index type joined with "+" for hybrid, e.g. faiss_flatip+bm25.
	Output: {output_path}/retrieved/{chunk_type}/{model_name}/{min}_{max}_{overlap}_{threshold}/{index_folder}/
	val_{val_queries_count}/{query_types joined by "_"}
	where index_folder is flatip, ivf_{nlist}, hnsw_{m}_{ef_construction}, bm25_{k1}_{b}_{tokenizer} or qdrant,
	joined with "+" for hybrid.
	What it does: builds the folder that holds the retrieved results of one index or hybrid pair.
	"""
	index_args = retrieval_args.index_args
	chunking_args = index_args.chunking_args
	settings_folder = (
		f"{chunking_args.min_chunk_size}_{chunking_args.max_chunk_size}_"
		f"{chunking_args.fixed_chunk_overlap}_{chunking_args.semantic_threshold}"
	)

	index_folders = []
	for one_index_type in index_type.split("+"):
		if one_index_type == FAISS_FLATIP_INDEX:
			index_folders.append("flatip")
		elif one_index_type == FAISS_IVF_INDEX:
			index_folders.append(f"ivf_{index_args.ivf_nlist}")
		elif one_index_type == FAISS_HNSW_INDEX:
			index_folders.append(f"hnsw_{index_args.hnsw_m}_{index_args.hnsw_ef_construction}")
		elif one_index_type == BM25_INDEX:
			index_folders.append(f"bm25_{index_args.bm25_k1}_{index_args.bm25_b}_{index_args.bm25_tokenizer}")
		elif one_index_type == QDRANT_INDEX:
			index_folders.append("qdrant")
		else:
			raise ValueError(f"index_type ({one_index_type}) is not a supported index type.")
	index_folder = "+".join(index_folders)

	split_folder = f"val_{val_queries_count}"
	query_types_folder = "_".join(query_args.query_types)

	return os.path.join(
		output_path, "retrieved", chunking_args.chunk_type, chunking_args.model_name, settings_folder,
		index_folder, split_folder, query_types_folder
	)
