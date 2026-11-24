import time

from common import paths
from common.args import parse_args
from common.args import make_ingestion_args
from common.args import make_embedder_args
from common.args import make_chunking_args
from common.args import make_embedding_args
from common.args import make_index_args
from common.args import make_vectordb_args
from rag.build.ingestion import compute_input_hash
from rag.build.ingestion import ingestion
from rag.build.chunking import Chunking
from rag.build.embedding import Embedding
from rag.build.indexing import Indexing
from rag.build.indexing_vectordb import IndexingVectorDb
from rag.build.manifest import is_build_complete
from rag.build.manifest import make_chunk_entry
from rag.build.manifest import make_index_entry
from rag.build.manifest import make_vectordb_entry
from rag.build.manifest import save_build_manifest
from rag.build.manifest import start_build_manifest


def run_build(args) -> None:
	"""
	Input: args, the parsed args.
	Output: nothing.
	What it does: builds the indexes: ingestion, then per chunk type chunking, embedding, every index type
	and every vector DB type, all with the one embedder from the config. Each layer loads what is already saved and builds only what is missing. At the end it
	saves the build manifest: what was built and where, so the later steps only load it. If a build with exactly these
	args finished and all its outputs are still there, it does nothing.
	"""
	output_path = paths.get_output_path(args.output_path, args.dataset_name)
	ingestion_args = make_ingestion_args(args)
	embedder_args = make_embedder_args(args)

	# 0. The hash of the input files, and a finished build with these args and input files, with all its outputs
	# still there
	print("\n\nBuild manifest check")
	input_hash = compute_input_hash(ingestion_args)
	print(f"Input hash : {input_hash}")
	if is_build_complete(output_path, args, input_hash):
		print(f"Build complete, nothing to do : {paths.get_build_manifest_path(output_path, args, input_hash)}")
		return

	# 1. Ingestion
	print("\n\nIngestion")

	s1 = time.time()
	documents = ingestion(ingestion_args, output_path)
	e1 = time.time() - s1
	print(f"Documents : {len(documents)}")
	print("\nIngestion time : ", e1)

	manifest = start_build_manifest(ingestion_args, embedder_args, args, input_hash)

	# 2. Chunking, once per chunk type
	for chunk_type in args.chunk_types:
		print(f"\n\n========== Chunk type : {chunk_type} ==========")
		print("\n\nChunking")
		chunking_args = make_chunking_args(args, chunk_type)

		s2 = time.time()
		chunking = Chunking(embedder_args, chunking_args)
		chunks = chunking.compute_chunks(documents, output_path)
		e2 = time.time() - s2
		print("\nChunking time : ", e2)

		# 3. Embedding
		print("\n\nEmbedding")
		embedding_args = make_embedding_args(embedder_args, chunking_args)

		s3 = time.time()
		embedding = Embedding(embedding_args)
		embeddings = embedding.compute_embeddings(chunks, output_path)
		e3 = time.time() - s3
		print("\nEmbedding time : ", e3)
		manifest.chunk_entries.append(make_chunk_entry(chunking_args, chunking.chunk_path, embedding.embedding_path, len(chunks)))

		# 4. Indexing, once per index type
		for index_type in args.index_types:
			print(f"\n\nIndexing : {index_type}")
			index_args = make_index_args(args, index_type, input_hash)

			s4 = time.time()
			index = Indexing(embedder_args, chunking_args, index_args).compute_index(chunks, embeddings, output_path)
			e4 = time.time() - s4
			print("\nIndexing time : ", e4)
			manifest.index_entries.append(make_index_entry(chunking_args, index_args, index, len(chunks)))

		# 5. Vector DB indexing, once per vector DB type
		for vectordb_type in args.vectordb_types:
			print(f"\n\nVector DB indexing : {vectordb_type}")
			vectordb_args = make_vectordb_args(args, vectordb_type, input_hash)

			s5 = time.time()
			vectordb = IndexingVectorDb(embedder_args, chunking_args, vectordb_args).compute_vectordb(chunks, embeddings, output_path)
			e5 = time.time() - s5
			print("\nVector DB indexing time : ", e5)
			manifest.index_entries.append(make_vectordb_entry(chunking_args, vectordb_args, vectordb.collection_name, len(chunks)))

	# 6. The build manifest, saved once every output is built
	save_build_manifest(manifest, output_path, args, input_hash)


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
