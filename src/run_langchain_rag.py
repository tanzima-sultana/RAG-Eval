import time

from common import paths_temp
from common.constants import LANGCHAIN_FRAMEWORK
from common.args_temp import make_ingestion_args
from common.args_temp import make_chunking_args
from common.args_temp import make_embedding_args
from common.args_temp import make_index_args
from langchain_rag.ingestion import ingestion
from langchain_rag.chunking import Chunking
from langchain_rag.embedding import Embedding
from langchain_rag.indexing import Indexing


def run_langchain_rag(args) -> None:
	"""
	Input: args, the parsed args.
	Output: nothing.
	What it does: runs the LangChain RAG. For now ingestion, chunking, embedding and indexing.
	"""
	# 1. Ingestion
	print("\n\nIngestion")
	ingestion_args = make_ingestion_args(args)

	s1 = time.time()
	documents = ingestion(ingestion_args)
	e1 = time.time() - s1
	print(f"Documents : {len(documents)}")
	print("\nIngestion time : ", e1)

	output_path = paths_temp.get_output_path(args.output_path, LANGCHAIN_FRAMEWORK, args.dataset_name)

	# 2. Chunking, once per chunk type
	for chunk_type in args.chunk_types:
		print(f"\n\n========== Chunk type : {chunk_type} ==========")
		print("\n\nChunking")
		chunking_args = make_chunking_args(args, chunk_type)

		s2 = time.time()
		chunks = Chunking(chunking_args).compute_chunks(documents, output_path)
		e2 = time.time() - s2
		print("\nChunking time : ", e2)

		# 3. Embedding
		print("\n\nEmbedding")
		embedding_args = make_embedding_args(chunking_args)

		s3 = time.time()
		embeddings = Embedding(embedding_args).compute_embeddings(chunks, output_path)
		e3 = time.time() - s3
		print("\nEmbedding time : ", e3)

		# 4. Indexing, once per index type
		for index_type in args.index_types:
			print(f"\n\nIndexing : {index_type}")
			index_args = make_index_args(args, chunking_args, index_type)

			s4 = time.time()
			index = Indexing(index_args).compute_index(chunks, embeddings, output_path)
			e4 = time.time() - s4
			print("\nIndexing time : ", e4)
