import time

from common import paths
from common.args import parse_build_args
from common.args import make_ingest_args
from common.args import make_chunking_args
from common.args import make_index_args
from stages.build.ingestion.ingestion import ingestion
from stages.build.chunking.chunking import Chunking
from stages.build.embedding.embedding import Embedding
from stages.build.indexing.indexing import Indexing

def run_build(args) -> None:
	"""
	Input: args, the parsed build arguments.
	Output: nothing. The documents, chunks, embeddings and indexes are saved under the dataset's output folder.
	What it does: runs ingestion, then chunking, embedding and indexing once per chunk type.
	"""
	# 1. Ingestion
	print("\n\nIngestion")
	ingest_args = make_ingest_args(args)

	s1 = time.time()
	documents = ingestion(ingest_args)
	e1 = time.time() - s1
	print("\nIngestion time : ", e1)

	output_path = paths.get_output_path(ingest_args.input_path, ingest_args.output_path)

	# Chunking, embedding and indexing are done once per chunk type
	for chunk_type in args.chunk_types:
		print(f"\n\n========== Chunk type : {chunk_type} ==========")

		# 2. Chunking
		print("\n\nChunking")
		chunking_args = make_chunking_args(args, chunk_type)

		s2 = time.time()
		chunks = Chunking(chunking_args).compute_chunks(documents, output_path)
		e2 = time.time() - s2
		print("\nChunking time : ", e2)

		# 3. Embedding
		print("\n\nEmbedding")

		s3 = time.time()
		embeddings = Embedding(chunking_args).compute_embeddings(chunks, output_path)
		e3 = time.time() - s3
		print("\nEmbedding time : ", e3)

		# 4. Indexing
		print("\n\nIndexing")

		s4 = time.time()
		# One index at a time. An index type not in the list is not built
		for index_type in args.index_types:
			index_args = make_index_args(args, index_type)
			Indexing(index_args, chunking_args).compute_index(chunks, embeddings, output_path)
		e4 = time.time() - s4
		print("\nIndexing time : ", e4)

if __name__ == "__main__":
	print("\n ------------- RAG -------- \n")
	start = time.time()

	args = parse_build_args()
	print(args)

	run_build(args)

	# ---------
	end = time.time()
	elapsed_time = end - start
	print("\n\nElapsed time : ", elapsed_time)
