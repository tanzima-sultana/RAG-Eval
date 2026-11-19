import os

from common.args_temp import ChunkingArgs
from common.args_temp import EmbeddingArgs
from common.args_temp import IndexArgs
from common.args_temp import VectorDbArgs
from common.args_temp import QnaArgs
from common.constants import FAISS_FLATIP_INDEX
from common.constants import FAISS_IVF_INDEX
from common.constants import FAISS_HNSW_INDEX


# ------ Dataset
def get_output_path(output_path: str, framework: str, dataset_name: str) -> str:
	"""
	Input: output_path, the root output folder; framework; dataset_name.
	Output: {output_path}/{framework}/{dataset_name}
	What it does: builds the dataset's output folder for one framework.
	"""
	return os.path.join(output_path, framework, dataset_name)


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
	Input: output_path, the dataset's output folder ({root}/{framework}/{dataset_name}); vectordb_args.
	Output: {framework}_{dataset_name}_{chunk_type}_{model_name}_{min}_{max}_{overlap}_{threshold}
	What it does: builds the Qdrant collection name. Any "/" is replaced with "-", as Qdrant does not allow it.
	"""
	chunking_args = vectordb_args.chunking_args
	normalized_output_path = os.path.normpath(output_path)
	dataset_name = os.path.basename(normalized_output_path)
	framework = os.path.basename(os.path.dirname(normalized_output_path))

	collection_name = (
		f"{framework}_{dataset_name}_{chunking_args.chunk_type}_{chunking_args.model_name}_"
		f"{chunking_args.min_chunk_size}_{chunking_args.max_chunk_size}_"
		f"{chunking_args.fixed_chunk_overlap}_{chunking_args.semantic_threshold}"
	)
	return collection_name.replace("/", "-")


# ------ QnA
def get_qna_path(output_path: str, qna_args: QnaArgs) -> str:
	"""
	Input: output_path, the dataset's output folder; qna_args.
	Output: {output_path}/qna/qna_{n_sample_documents}_{n_sample_evals}_{qna_llm_model}_{qna_embedding_model}.json
	What it does: builds the QnA file path. Any "/" or ":" in a model name is replaced with "-".
	"""
	qna_llm_model = qna_args.qna_llm_args.llm_model.replace("/", "-").replace(":", "-")
	qna_embedding_model = qna_args.qna_embedding_model.replace("/", "-").replace(":", "-")
	file_name = (
		f"qna_{qna_args.n_sample_documents}_{qna_args.n_sample_evals}_"
		f"{qna_llm_model}_{qna_embedding_model}.json"
	)
	return os.path.join(output_path, "qna", file_name)


def get_qna_kg_path(output_path: str, qna_args: QnaArgs) -> str:
	"""
	Input: output_path, the dataset's output folder; qna_args.
	Output: {output_path}/qna/kg_{n_sample_documents}_{qna_kg_llm_model}_{qna_embedding_model}.json
	What it does: builds the RAGAS knowledge graph file path. Any "/" or ":" in a model name is replaced with "-".
	"""
	qna_kg_llm_model = qna_args.qna_kg_llm_args.llm_model.replace("/", "-").replace(":", "-")
	qna_embedding_model = qna_args.qna_embedding_model.replace("/", "-").replace(":", "-")
	file_name = f"kg_{qna_args.n_sample_documents}_{qna_kg_llm_model}_{qna_embedding_model}.json"
	return os.path.join(output_path, "qna", file_name)
