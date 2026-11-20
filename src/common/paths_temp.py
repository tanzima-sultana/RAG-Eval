import os

from common.args_temp import ChunkingArgs
from common.args_temp import EmbeddingArgs
from common.args_temp import IndexArgs
from common.args_temp import VectorDbArgs
from common.args_temp import EvalDatasetArgs
from common.args_temp import QueryArgs
from common.args_temp import RetrievalArgs
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
	Output: {output_path}/chunks/{chunk_type}/{model_name}/{min}_{max}_{overlap}_{threshold}
	What it does: builds the folder that holds one chunking run's chunks.
	"""
	settings_folder = (
		f"{chunking_args.min_chunk_size}_{chunking_args.max_chunk_size}_"
		f"{chunking_args.fixed_chunk_overlap}_{chunking_args.semantic_threshold}"
	)
	return os.path.join(output_path, "chunks", chunking_args.chunk_type, chunking_args.model_name, settings_folder)


# ------ Embedding
def get_embedding_path(output_path: str, embedding_args: EmbeddingArgs) -> str:
	"""
	Input: output_path, the dataset's output folder; embedding_args.
	Output: {output_path}/embeddings/{chunk_type}/{model_name}/{min}_{max}_{overlap}_{threshold}
	What it does: builds the folder that holds one chunking run's embeddings.
	"""
	chunking_args = embedding_args.chunking_args
	settings_folder = (
		f"{chunking_args.min_chunk_size}_{chunking_args.max_chunk_size}_"
		f"{chunking_args.fixed_chunk_overlap}_{chunking_args.semantic_threshold}"
	)
	return os.path.join(output_path, "embeddings", chunking_args.chunk_type, chunking_args.model_name, settings_folder)


# ------ Indexing
def get_faiss_index_path(output_path: str, index_args: IndexArgs) -> str:
	"""
	Input: output_path, the dataset's output folder; index_args.
	Output: {output_path}/indexes/{index_type}/{chunk_type}/{model_name}/{min}_{max}_{overlap}_{threshold}/{faiss_folder}
	where faiss_folder is flatip, ivf_{nlist} or hnsw_{m}_{ef_construction}.
	What it does: builds the folder that holds one FAISS index.
	"""
	chunking_args = index_args.chunking_args
	settings_folder = (
		f"{chunking_args.min_chunk_size}_{chunking_args.max_chunk_size}_"
		f"{chunking_args.fixed_chunk_overlap}_{chunking_args.semantic_threshold}"
	)

	if index_args.index_type == FAISS_FLATIP_INDEX:
		faiss_folder = "flatip"
	elif index_args.index_type == FAISS_IVF_INDEX:
		faiss_folder = f"ivf_{index_args.ivf_nlist}"
	elif index_args.index_type == FAISS_HNSW_INDEX:
		faiss_folder = f"hnsw_{index_args.hnsw_m}_{index_args.hnsw_ef_construction}"
	else:
		raise ValueError(f"index_type ({index_args.index_type}) is not a FAISS index type.")

	return os.path.join(
		output_path, "indexes", index_args.index_type, chunking_args.chunk_type, chunking_args.model_name,
		settings_folder, faiss_folder
	)


def get_bm25_index_path(output_path: str, index_args: IndexArgs) -> str:
	"""
	Input: output_path, the dataset's output folder; index_args.
	Output: {output_path}/indexes/bm25/{chunk_type}/{model_name}/{min}_{max}_{overlap}_{threshold}/{k1}_{b}_{tokenizer}
	What it does: builds the folder that holds one BM25 index.
	"""
	chunking_args = index_args.chunking_args
	settings_folder = (
		f"{chunking_args.min_chunk_size}_{chunking_args.max_chunk_size}_"
		f"{chunking_args.fixed_chunk_overlap}_{chunking_args.semantic_threshold}"
	)
	bm25_folder = f"{index_args.bm25_k1}_{index_args.bm25_b}_{index_args.bm25_tokenizer}"

	return os.path.join(
		output_path, "indexes", "bm25", chunking_args.chunk_type, chunking_args.model_name,
		settings_folder, bm25_folder
	)


# ------ Vector DB
def get_qdrant_collection_name(output_path: str, vectordb_args: VectorDbArgs) -> str:
	"""
	Input: output_path, the dataset's output folder ({root}/{dataset_name}); vectordb_args.
	Output: {root folder name}_{dataset_name}_{chunk_type}_{model_name}_{min}_{max}_{overlap}_{threshold}
	What it does: builds the Qdrant collection name. The root folder name keeps a temp/ run and an outputs/ run of the
	same dataset in separate collections. Any "/" is replaced with "-", as Qdrant does not allow it.
	"""
	chunking_args = vectordb_args.chunking_args
	normalized_output_path = os.path.normpath(output_path)
	dataset_name = os.path.basename(normalized_output_path)
	root_folder_name = os.path.basename(os.path.dirname(normalized_output_path))

	collection_name = (
		f"{root_folder_name}_{dataset_name}_{chunking_args.chunk_type}_{chunking_args.model_name}_"
		f"{chunking_args.min_chunk_size}_{chunking_args.max_chunk_size}_"
		f"{chunking_args.fixed_chunk_overlap}_{chunking_args.semantic_threshold}"
	)
	return collection_name.replace("/", "-")


# ------ Eval dataset
def get_eval_dataset_path(output_path: str, eval_dataset_args: EvalDatasetArgs) -> str:
	"""
	Input: output_path, the dataset's output folder; eval_dataset_args.
	Output: {output_path}/eval_dataset/eval_dataset_{n_sample_documents}_{n_sample_evals}_{eval_dataset_llm_model}_{eval_dataset_embedding_model}.json
	What it does: builds the eval dataset file path. Any "/" or ":" in a model name is replaced with "-".
	"""
	eval_dataset_llm_model = eval_dataset_args.eval_dataset_llm_args.llm_model.replace("/", "-").replace(":", "-")
	eval_dataset_embedding_model = eval_dataset_args.eval_dataset_embedding_model.replace("/", "-").replace(":", "-")
	file_name = (
		f"eval_dataset_{eval_dataset_args.n_sample_documents}_{eval_dataset_args.n_sample_evals}_"
		f"{eval_dataset_llm_model}_{eval_dataset_embedding_model}.json"
	)
	return os.path.join(output_path, "eval_dataset", file_name)


def get_eval_dataset_kg_path(output_path: str, eval_dataset_args: EvalDatasetArgs) -> str:
	"""
	Input: output_path, the dataset's output folder; eval_dataset_args.
	Output: {output_path}/eval_dataset/kg_{n_sample_documents}_{eval_dataset_kg_llm_model}_{eval_dataset_embedding_model}.json
	What it does: builds the RAGAS knowledge graph file path. Any "/" or ":" in a model name is replaced with "-".
	"""
	eval_dataset_kg_llm_model = eval_dataset_args.eval_dataset_kg_llm_args.llm_model.replace("/", "-").replace(":", "-")
	eval_dataset_embedding_model = eval_dataset_args.eval_dataset_embedding_model.replace("/", "-").replace(":", "-")
	file_name = f"kg_{eval_dataset_args.n_sample_documents}_{eval_dataset_kg_llm_model}_{eval_dataset_embedding_model}.json"
	return os.path.join(output_path, "eval_dataset", file_name)


# ------ Query transformation
def get_query_path(output_path: str, eval_dataset_args: EvalDatasetArgs, query_args: QueryArgs) -> str:
	"""
	Input: output_path, the dataset's output folder; eval_dataset_args, the settings of the eval dataset the queries are made from;
	query_args.
	Output: {output_path}/queries/{n_sample_documents}_{n_sample_evals}_{eval_dataset_llm_model}_{eval_dataset_embedding_model}_
	{embedding_model}_{query_llm_model}_{n_multi_qus}
	What it does: builds the folder that holds one query run. Any "/" or ":" in a model name is replaced with "-".
	"""
	eval_dataset_llm_model = eval_dataset_args.eval_dataset_llm_args.llm_model.replace("/", "-").replace(":", "-")
	eval_dataset_embedding_model = eval_dataset_args.eval_dataset_embedding_model.replace("/", "-").replace(":", "-")
	embedding_model = query_args.embedding_model.replace("/", "-").replace(":", "-")
	query_llm_model = query_args.query_llm_args.llm_model.replace("/", "-").replace(":", "-")
	folder_name = (
		f"{eval_dataset_args.n_sample_documents}_{eval_dataset_args.n_sample_evals}_{eval_dataset_llm_model}_{eval_dataset_embedding_model}_"
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
