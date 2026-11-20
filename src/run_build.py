import time

from common import paths_temp
from common.args_temp import parse_args
from common.args_temp import make_ingestion_args
from common.args_temp import make_chunking_args
from common.args_temp import make_embedding_args
from common.args_temp import make_index_args
from common.args_temp import make_vectordb_args
from rag.build.ingestion import ingestion
from rag.build.chunking import Chunking
from rag.build.embedding import Embedding
from rag.build.indexing import Indexing
from rag.build.indexing_vectordb import IndexingVectorDb


def run_build(args) -> None:
	"""
	Input: args, the parsed args.
	Output: nothing.
	What it does: builds the indexes: ingestion, then per chunk type chunking, embedding, every index type
	and every vector DB type. Each layer loads what is already saved and builds only what is missing.
	"""
	# 1. Ingestion
	print("\n\nIngestion")
	ingestion_args = make_ingestion_args(args)

	s1 = time.time()
	documents = ingestion(ingestion_args)
	e1 = time.time() - s1
	print(f"Documents : {len(documents)}")
	print("\nIngestion time : ", e1)

	output_path = paths_temp.get_output_path(args.output_path, args.dataset_name)

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
			Indexing(index_args).compute_index(chunks, embeddings, output_path)
			e4 = time.time() - s4
			print("\nIndexing time : ", e4)

		# 5. Vector DB indexing, once per vector DB type
		for vectordb_type in args.vectordb_types:
			print(f"\n\nVector DB indexing : {vectordb_type}")
			vectordb_args = make_vectordb_args(args, chunking_args, vectordb_type)

			s5 = time.time()
			IndexingVectorDb(vectordb_args).compute_vectordb(chunks, embeddings, output_path)
			e5 = time.time() - s5
			print("\nVector DB indexing time : ", e5)


if __name__ == "__main__":
	print("\n ------------- Build -------- \n")
	start = time.time()

	"""
	Input: --config and any override flags from the command line.
	Output: nothing.
	What it does: parses the args and runs the build.
	"""
	args = parse_args()
	run_build(args)

	elapsed_time = time.time() - start
	print("\n\nElapsed time : ", elapsed_time)
