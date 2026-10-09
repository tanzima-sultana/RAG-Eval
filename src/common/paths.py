
import os

from common.args import ChunkingArgs
from common.args import IndexArgs
from common.constants import NO_FRAMEWORK
from common.constants import FAISS_FLATIP_INDEX, FAISS_IVF_INDEX, FAISS_HNSW_INDEX

# ------ Chunking
def get_chunk_path(output_path:str, chunking_args: ChunkingArgs) -> str:
	"""
	Input: output_path, the dataset's output folder; chunking_args, the
	chunking parameters.
	Output: the chunk folder path as a string, ending with "/".
	What it does: builds the folder that holds one chunking run's chunks:
	{output_path}/chunks/{chunk_type}/{model_name}/{min}_{max}_{overlap}_{threshold}/
	for the built-in chunkers, and the same with a {framework}/ folder after
	chunks/ when a framework such as langchain is used, so the two never
	share a cache.
	"""

	if chunking_args.framework == NO_FRAMEWORK:
		framework_folder = ""
	else:
		framework_folder = f"{chunking_args.framework}/"

	return (
		f"{output_path}/chunks/{framework_folder}{chunking_args.chunk_type}/{chunking_args.model_name}/"
		f"{chunking_args.min_chunk_size}_{chunking_args.max_chunk_size}_"
		f"{chunking_args.fixed_chunk_overlap}_{chunking_args.semantic_threshold}/"
	)

# ------ Embedding
def get_embedding_path(output_path:str, chunking_args: ChunkingArgs) -> str:
	"""
	Input: output_path, the dataset's output folder; chunking_args, the
	chunking parameters.
	Output: the chunk folder path as a string, ending with "/".
	What it does: builds the folder that holds one chunking run's chunks:
	{output_path}/chunks/{chunk_type}/{model_name}/{min}_{max}_{overlap}_{threshold}/
	for the built-in chunkers, and the same with a {framework}/ folder after
	chunks/ when a framework such as langchain is used, so the two never
	share a cache.
	"""

	if chunking_args.framework == NO_FRAMEWORK:
		framework_folder = ""
	else:
		framework_folder = f"{chunking_args.framework}/"

	return (
		f"{output_path}/embeddings/{framework_folder}{chunking_args.chunk_type}/{chunking_args.model_name}/"
		f"{chunking_args.min_chunk_size}_{chunking_args.max_chunk_size}_"
		f"{chunking_args.fixed_chunk_overlap}_{chunking_args.semantic_threshold}/"
	)

# ------ Indexing
def get_faiss_index_path(output_path:str, chunking_args: ChunkingArgs, index_args: IndexArgs) -> str:
	"""
	Input: output_path, the dataset's output folder; chunking_args, the
	chunking parameters; index_args, the indexing parameters.
	Output: the FAISS index folder path as a string, ending with "/".
	What it does: builds the folder that holds one FAISS index build:
	{output_path}/indexes/faiss/{chunk_type}/{model_name}/{min}_{max}_{overlap}_{threshold}/{faiss_folder}/
	where faiss_folder is "flatip", "ivf_{nlist}" or "hnsw_{m}_{ef_construction}",
	so changing a FAISS parameter never reuses an old index. A {framework}/
	folder is added after faiss/ when a framework such as langchain is used.
	"""

	if chunking_args.framework == NO_FRAMEWORK:
		framework_folder = ""
	else:
		framework_folder = f"{chunking_args.framework}/"

	if index_args.faiss_index_type == FAISS_FLATIP_INDEX:
		faiss_folder = "flatip"
	elif index_args.faiss_index_type == FAISS_IVF_INDEX:
		faiss_folder = f"ivf_{index_args.ivf_nlist}"
	elif index_args.faiss_index_type == FAISS_HNSW_INDEX:
		faiss_folder = f"hnsw_{index_args.hnsw_m}_{index_args.hnsw_ef_construction}"
	else:
		raise ValueError(f"faiss_index_type ({index_args.faiss_index_type}) is not supported")

	return (
		f"{output_path}/indexes/faiss/{framework_folder}{chunking_args.chunk_type}/{chunking_args.model_name}/"
		f"{chunking_args.min_chunk_size}_{chunking_args.max_chunk_size}_"
		f"{chunking_args.fixed_chunk_overlap}_{chunking_args.semantic_threshold}/"
		f"{faiss_folder}/"
	)

def get_bm25_index_path(output_path:str, chunking_args: ChunkingArgs, index_args: IndexArgs) -> str:
	"""
	Input: output_path, the dataset's output folder; chunking_args, the
	chunking parameters; index_args, the indexing parameters.
	Output: the BM25 index folder path as a string, ending with "/".
	What it does: builds the folder that holds one BM25 index build:
	{output_path}/indexes/bm25/{chunk_type}/{model_name}/{min}_{max}_{overlap}_{threshold}/{k1}_{b}_{tokenizer}/
	so changing a BM25 parameter never reuses an old index. A {framework}/
	folder is added after bm25/ when a framework such as langchain is used.
	"""

	if chunking_args.framework == NO_FRAMEWORK:
		framework_folder = ""
	else:
		framework_folder = f"{chunking_args.framework}/"

	return (
		f"{output_path}/indexes/bm25/{framework_folder}{chunking_args.chunk_type}/{chunking_args.model_name}/"
		f"{chunking_args.min_chunk_size}_{chunking_args.max_chunk_size}_"
		f"{chunking_args.fixed_chunk_overlap}_{chunking_args.semantic_threshold}/"
		f"{index_args.bm25_k1}_{index_args.bm25_b}_{index_args.bm25_tokenizer}/"
	)

def get_qdrant_index_path(output_path:str, chunking_args: ChunkingArgs) -> str:
	"""
	Input: output_path, the dataset's output folder; chunking_args, the
	chunking parameters.
	Output: the Qdrant index folder path as a string, ending with "/".
	What it does: builds the folder that holds one Qdrant index build:
	{output_path}/indexes/qdrant/{chunk_type}/{model_name}/{min}_{max}_{overlap}_{threshold}/
	A {framework}/ folder is added after qdrant/ when a framework such as
	langchain is used.
	"""

	if chunking_args.framework == NO_FRAMEWORK:
		framework_folder = ""
	else:
		framework_folder = f"{chunking_args.framework}/"

	return (
		f"{output_path}/indexes/qdrant/{framework_folder}{chunking_args.chunk_type}/{chunking_args.model_name}/"
		f"{chunking_args.min_chunk_size}_{chunking_args.max_chunk_size}_"
		f"{chunking_args.fixed_chunk_overlap}_{chunking_args.semantic_threshold}/"
	)
