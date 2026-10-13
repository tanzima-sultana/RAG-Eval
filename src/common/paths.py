
import os

from common.args import ChunkingArgs
from common.args import IndexArgs
from common.args import EvalQnAArgs
from common.args import QueryArgs
from common.constants import NO_FRAMEWORK
from common.constants import FAISS_FLATIP_INDEX, FAISS_IVF_INDEX, FAISS_HNSW_INDEX

# ------ Dataset
def get_output_path(input_path:str, output_path:str) -> str:
	"""
	Input: input_path, a Hugging Face dataset id or a local file/folder path;
	output_path, the base output folder.
	Output: the folder "{output_path}/{dataset_name}" as a string.
	What it does: takes the last part of input_path as the dataset name,
	without trailing slashes or a file extension. For example
	"pixparse/pdfa-eng-wds" gives "pdfa-eng-wds", "data/docs/" gives "docs"
	and "data/report.pdf" gives "report".
	"""
	path_without_trailing_slash = input_path.rstrip("/")
	last_path_part = os.path.basename(path_without_trailing_slash)
	name_and_extension = os.path.splitext(last_path_part)
	dataset_name = name_and_extension[0]
	return f"{output_path}/{dataset_name}"

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

def get_qdrant_collection_name(output_path:str, chunking_args: ChunkingArgs) -> str:
	"""
	Input: output_path, the dataset's output folder; chunking_args, the
	chunking parameters.
	Output: the Qdrant collection name as a string.
	What it does: builds the name of the collection on the Qdrant server
	that holds one Qdrant index build:
	{dataset_name}_{chunk_type}_{model_name}_{min}_{max}_{overlap}_{threshold}
	where dataset_name is the last folder of output_path. A {framework}_
	part is added after the dataset name when a framework such as
	langchain is used. Any "/" in model_name is replaced with "-", because
	Qdrant does not allow "/" in collection names.
	"""

	if chunking_args.framework == NO_FRAMEWORK:
		framework_part = ""
	else:
		framework_part = f"{chunking_args.framework}_"

	dataset_name = os.path.basename(output_path.rstrip("/"))
	model_name = chunking_args.model_name.replace("/", "-")

	return (
		f"{dataset_name}_{framework_part}{chunking_args.chunk_type}_{model_name}_"
		f"{chunking_args.min_chunk_size}_{chunking_args.max_chunk_size}_"
		f"{chunking_args.fixed_chunk_overlap}_{chunking_args.semantic_threshold}"
	)

# ------ Eval QnA
def get_eval_qna_path(output_path:str, eval_qna_args: EvalQnAArgs) -> str:
	"""
	Input: output_path, the dataset's output folder; eval_qna_args, the eval QnA settings.
	Output: the eval QnA cache file path as a string.
	What it does: builds {output_path}/eval_qna_{n_sample_documents}_{n_sample_evals}_{eval_qna_llm}_{eval_qna_model}.json,
	with any "/" in the model name replaced by "-".
	"""
	eval_qna_model = eval_qna_args.eval_qna_model.replace("/", "-")

	return (
		f"{output_path}/eval_qna_{eval_qna_args.n_sample_documents}_"
		f"{eval_qna_args.n_sample_evals}_{eval_qna_args.eval_qna_llm}_{eval_qna_model}.json"
	)

def get_eval_kg_path(output_path:str, eval_qna_args: EvalQnAArgs) -> str:
	"""
	Input: output_path, the dataset's output folder; eval_qna_args, the eval QnA settings.
	Output: the RAGAS knowledge graph cache file path as a string.
	What it does: builds {output_path}/eval_kg_{n_sample_documents}_{eval_qna_llm}_{eval_qna_model}.json,
	with any "/" in the model name replaced by "-".
	"""
	eval_qna_model = eval_qna_args.eval_qna_model.replace("/", "-")

	return (
		f"{output_path}/eval_kg_{eval_qna_args.n_sample_documents}_"
		f"{eval_qna_args.eval_qna_llm}_{eval_qna_model}.json"
	)

# ------ Query
def get_query_path(output_path:str, eval_qna_args: EvalQnAArgs, query_args: QueryArgs) -> str:
	"""
	Input: output_path, the dataset's output folder; eval_qna_args, the settings of the eval set the queries are made from;
	query_args, the query settings.
	Output: the query cache folder path as a string.
	What it does: builds {output_path}/queries/{n_sample_documents}_{n_sample_evals}_{eval_qna_llm}_{eval_qna_model}_
	{embedding_model}_{query_llm}_{query_llm_model}_{n_multi_qus}, with any "/" or ":" in the model names replaced by "-".
	"""
	eval_qna_model = eval_qna_args.eval_qna_model.replace("/", "-")
	embedding_model = query_args.embedding_model.replace("/", "-")
	query_llm_model = query_args.query_llm_model.replace("/", "-").replace(":", "-")

	return (
		f"{output_path}/queries/{eval_qna_args.n_sample_documents}_{eval_qna_args.n_sample_evals}_"
		f"{eval_qna_args.eval_qna_llm}_{eval_qna_model}_{embedding_model}_"
		f"{query_args.query_llm}_{query_llm_model}_{query_args.n_multi_qus}"
	)
