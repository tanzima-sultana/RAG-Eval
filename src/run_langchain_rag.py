import time

from common import paths_temp
from common.constants import LANGCHAIN_FRAMEWORK
from common.constants import HYBRID_RETRIEVAL
from common.args_temp import make_ingestion_args
from common.args_temp import make_chunking_args
from common.args_temp import make_embedding_args
from common.args_temp import make_index_args
from common.args_temp import make_vectordb_args
from common.args_temp import make_qna_args
from common.args_temp import make_ollama_args
from common.args_temp import make_query_args
from common.args_temp import make_retrieval_args
from langchain_rag.ingestion import ingestion
from langchain_rag.chunking import Chunking
from langchain_rag.embedding import Embedding
from langchain_rag.indexing import Indexing
from langchain_rag.indexing_vectordb import IndexingVectorDb
from langchain_rag.qna_set import QnaSet
from langchain_rag.query_transformation import QueryTransformation
from langchain_rag.retrieval import Retrieval


def run_langchain_rag(args) -> None:
	"""
	Input: args, the parsed args.
	Output: nothing.
	What it does: runs the LangChain RAG. For now ingestion, QnA, query transformation, then per chunk type the build
	layers and retrieval.
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

	# 2. QnA, once for the dataset
	print("\n\nQnA")
	qna_args = make_qna_args(args)
	ollama_args = make_ollama_args(args)

	s2 = time.time()
	qna_items = QnaSet(qna_args, ollama_args).compute_qna(documents, output_path)
	e2 = time.time() - s2
	print("\nQnA time : ", e2)

	# 3. Query transformation, once for the QnA set
	print("\n\nQuery transformation")
	query_args = make_query_args(args)

	s3 = time.time()
	query_transformation = QueryTransformation(query_args, ollama_args)
	queries = query_transformation.compute_queries(qna_items, qna_args, output_path)
	e3 = time.time() - s3
	print("\nQuery transformation time : ", e3)

	# 4. Query split: val for retrieval and retrieval evaluation, test for answer generation and answer evaluation
	print("\n\nQuery split")
	val_queries, test_queries = query_transformation.split_queries(queries)

	# 5. Chunking, once per chunk type
	for chunk_type in args.chunk_types:
		print(f"\n\n========== Chunk type : {chunk_type} ==========")
		print("\n\nChunking")
		chunking_args = make_chunking_args(args, chunk_type)

		s5 = time.time()
		chunks = Chunking(chunking_args).compute_chunks(documents, output_path)
		e5 = time.time() - s5
		print("\nChunking time : ", e5)

		# 6. Embedding
		print("\n\nEmbedding")
		embedding_args = make_embedding_args(chunking_args)

		s6 = time.time()
		embeddings = Embedding(embedding_args).compute_embeddings(chunks, output_path)
		e6 = time.time() - s6
		print("\nEmbedding time : ", e6)

		# The plain dense and BM25 results of this chunk type, filled by the index and vector DB loops
		retrieval_args = make_retrieval_args(args, chunking_args)
		retrieval = Retrieval(retrieval_args, query_args, len(val_queries))
		retrieved_items = []

		# 7. Indexing and retrieval, once per index type, on the index just built or loaded
		for index_type in args.index_types:
			print(f"\n\nIndexing : {index_type}")
			index_args = make_index_args(args, chunking_args, index_type)

			s7 = time.time()
			index = Indexing(index_args).compute_index(chunks, embeddings, output_path)
			e7 = time.time() - s7
			print("\nIndexing time : ", e7)

			print(f"\n\nRetrieval : {index_type}")
			s8 = time.time()
			index_retrieved_items = retrieval.compute_retrieval(index, index_type, val_queries, output_path)
			retrieved_items.extend(index_retrieved_items)
			e8 = time.time() - s8
			print("\nRetrieval time : ", e8)

		# 8. Vector DB indexing and retrieval, once per vector DB type, on the collection just built or loaded
		for vectordb_type in args.vectordb_types:
			print(f"\n\nVector DB indexing : {vectordb_type}")
			vectordb_args = make_vectordb_args(args, chunking_args, vectordb_type)

			s9 = time.time()
			vectordb = IndexingVectorDb(vectordb_args).compute_vectordb(chunks, embeddings, output_path)
			e9 = time.time() - s9
			print("\nVector DB indexing time : ", e9)

			print(f"\n\nRetrieval : {vectordb_type}")
			s10 = time.time()
			vectordb_retrieved_items = retrieval.compute_vectordb_retrieval(vectordb, val_queries, output_path)
			retrieved_items.extend(vectordb_retrieved_items)
			e10 = time.time() - s10
			print("\nRetrieval time : ", e10)

		# 9. Hybrid retrieval, from the dense and BM25 results of this chunk type
		if HYBRID_RETRIEVAL in retrieval_args.retrieval_types:
			print("\n\nHybrid retrieval")
			s11 = time.time()
			hybrid_retrieved_items = retrieval.hybrid_retrieval(retrieved_items, output_path)
			retrieved_items.extend(hybrid_retrieved_items)
			e11 = time.time() - s11
			print("\nHybrid retrieval time : ", e11)

		# 10. Reranking of every plain result, kept as a separate list
		if retrieval_args.reranking:
			print("\n\nReranking")
			s12 = time.time()
			reranked_retrieved_items = retrieval.rerank_retrieval(retrieved_items, chunks, qna_items, output_path)
			e12 = time.time() - s12
			print(f"Reranked results : {len(reranked_retrieved_items)}")
			print("\nReranking time : ", e12)

		print(f"\nPlain results : {len(retrieved_items)}")
