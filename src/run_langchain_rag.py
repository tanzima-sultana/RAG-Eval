import time

from common import paths_temp
from common.constants import LANGCHAIN_FRAMEWORK
from common.args_temp import make_ingestion_args
from common.args_temp import make_chunking_args
from common.args_temp import make_embedding_args
from common.args_temp import make_index_args
from common.args_temp import make_vectordb_args
from common.args_temp import make_qna_args
from common.args_temp import make_ollama_args
from common.args_temp import make_query_args
from langchain_rag.ingestion import ingestion
from langchain_rag.chunking import Chunking
from langchain_rag.embedding import Embedding
from langchain_rag.indexing import Indexing
from langchain_rag.indexing_vectordb import IndexingVectorDb
from langchain_rag.qna_set import QnaSet
from langchain_rag.query_transformation import QueryTransformation


def run_langchain_rag(args) -> None:
	"""
	Input: args, the parsed args.
	Output: nothing.
	What it does: runs the LangChain RAG. For now the build layers, QnA and query transformation.
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

		# 5. Vector DB indexing, once per vector DB type
		for vectordb_type in args.vectordb_types:
			print(f"\n\nVector DB indexing : {vectordb_type}")
			vectordb_args = make_vectordb_args(args, chunking_args, vectordb_type)

			s5 = time.time()
			vectordb = IndexingVectorDb(vectordb_args).compute_vectordb(chunks, embeddings, output_path)
			e5 = time.time() - s5
			print("\nVector DB indexing time : ", e5)

	# 6. QnA, once for the dataset
	print("\n\nQnA")
	qna_args = make_qna_args(args)
	ollama_args = make_ollama_args(args)

	s6 = time.time()
	qna_items = QnaSet(qna_args, ollama_args).compute_qna(documents, output_path)
	e6 = time.time() - s6
	print("\nQnA time : ", e6)

	# 7. Query transformation, once for the QnA set
	print("\n\nQuery transformation")
	query_args = make_query_args(args)

	s7 = time.time()
	queries = QueryTransformation(query_args, ollama_args).compute_queries(qna_items, qna_args, output_path)
	e7 = time.time() - s7
	print("\nQuery transformation time : ", e7)
