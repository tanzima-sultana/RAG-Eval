import time

from common import constants
from common import paths
from common.args import parse_args
from common.args import make_ingestion_args
from common.args import make_embedder_args
from common.args import make_chunking_args
from common.args import make_embedding_args
from common.args import make_bm25_indexing_args
from common.args import make_faiss_hnsw_args
from common.args import make_faiss_ivf_args
from common.args import make_qdrant_args
from rag.build.ingestion.ingestion import ingestion
from rag.build.chunking.chunking import Chunking
from rag.build.embedding.embedding import Embedding
from rag.build.indexing.indexing import Indexing
from rag.build.manifest import make_build_manifest
from rag.build.manifest import save_build_manifests
from schema.manifest import Bm25Entry
from schema.manifest import ChunkEntry
from schema.manifest import EmbeddingEntry
from schema.manifest import FaissEntry
from schema.manifest import IngestionEntry
from schema.manifest import VectorDbEntry


def run_build(args) -> None:
	"""
	Input: args, the parsed args.
	Output: nothing.
	What it does: builds the indexes: ingestion, then per chunk type chunking, embedding, BM25, every FAISS setup
	and every vector DB type, all with the one embedder from the config. Each layer loads what is already saved and builds only what is missing. Each
	setup gets its own manifest. At the end all the manifests are saved to one file: what was built and where, so the
	later steps only load it.
	"""
	output_path = paths.get_output_path(args.output_path, args.dataset_name)
	ingestion_args = make_ingestion_args(args)
	embedder_args = make_embedder_args(args)
	bm25_args = make_bm25_indexing_args(args)
	qdrant_args = None
	if len(args.vectordb_types) > 0:
		qdrant_args = make_qdrant_args(args)

	# 1. Ingestion
	print("\n\nIngestion")

	s1 = time.time()
	files, ingestion_hash, documents_path = ingestion(ingestion_args, output_path)
	e1 = time.time() - s1
	page_count = 0
	for file in files:
		page_count += len(file.pages)
	print(f"Files : {len(files)}, pages : {page_count}")
	print("\nIngestion time : ", e1)

	ingestion_entry = IngestionEntry(ingestion_args, ingestion_hash, documents_path)
	manifests = []

	# 2. Chunking, once per chunk type and fixed chunk overlap
	for chunk_type in args.chunk_types:
		for fixed_chunk_overlap in args.fixed_chunk_overlap:
			print(f"\n\n========== Chunk type : {chunk_type}, fixed chunk overlap : {fixed_chunk_overlap} ==========")
			print("\n\nChunking")
			chunking_args = make_chunking_args(args, chunk_type, fixed_chunk_overlap)

			s2 = time.time()
			chunking = Chunking(embedder_args, chunking_args, ingestion_hash)
			chunks, chunking_hash, chunk_path = chunking.compute_chunks(files, output_path)
			e2 = time.time() - s2
			print("\nChunking time : ", e2)

			# 3. Embedding
			print("\n\nEmbedding")
			embedding_args = make_embedding_args(embedder_args, chunking_args)

			s3 = time.time()
			embedding = Embedding(embedding_args, chunking_hash)
			embeddings, embedding_hash, embedding_path = embedding.compute_embeddings(chunks, output_path)
			e3 = time.time() - s3
			print("\nEmbedding time : ", e3)
			chunk_entry = ChunkEntry(chunking_args, chunking_hash, chunk_path)
			embedding_entry = EmbeddingEntry(embedding_args, embedding_hash, embedding_path)

			# 4. Indexing: BM25 always, then the FAISS and vector DB types of the config
			print("\n\nIndexing")
			indexing = Indexing(embedder_args, bm25_args, args.faiss_types, args.vectordb_types, qdrant_args, chunking_hash, embedding_hash)
			bm25_indexing = indexing.bm25_indexing
			faiss_indexing = indexing.faiss_indexing
			vectordb_indexing = indexing.vectordb_indexing

			s4 = time.time()

			# 4a. BM25, always
			print("\nBM25 indexing")
			bm25_index, bm25_hash, bm25_index_path = bm25_indexing.bm25_indexing(chunks, output_path)
			print(f"BM25 index hash : {bm25_hash}, path : {bm25_index_path}")
			bm25_entry = Bm25Entry(bm25_args, bm25_hash, bm25_index_path)

			# 4b. FAISS, once per FAISS type, and faiss_ivf once per ivf_nlist_multiplier
			for faiss_type in args.faiss_types:
				if faiss_type == constants.FAISS_FLATIP_INDEX:
					print(f"\nFAISS indexing : {faiss_type}")
					faiss_index, faiss_hash, faiss_index_path = faiss_indexing.faiss_flatip_indexing(chunks, embeddings, output_path)
					print(f"{faiss_type} index hash : {faiss_hash}, path : {faiss_index_path}")

					# Build done for one setup keeping vectordb as none
					faiss_entry = FaissEntry(faiss_type, None, faiss_hash, faiss_index_path)
					manifest = make_build_manifest(
						args.dataset_name,
						args.dataset_size,
						ingestion_entry,
						chunk_entry,
						embedder_args,
						embedding_entry,
						bm25_entry,
						faiss_entry,
						None
					)
					manifests.append(manifest)

				elif faiss_type == constants.FAISS_IVF_INDEX:
					if args.ivf_nlist_multiplier is None:
						raise ValueError("ivf_nlist_multiplier is missing. Give it in the faiss_ivf block of the faiss_indexing config.")
					for ivf_nlist_multiplier in args.ivf_nlist_multiplier:
						print(f"\nFAISS indexing : {faiss_type}, ivf nlist multiplier : {ivf_nlist_multiplier}")
						ivf_args = make_faiss_ivf_args(args, ivf_nlist_multiplier)
						faiss_index, faiss_hash, faiss_index_path = faiss_indexing.faiss_ivf_indexing(chunks, embeddings, output_path, ivf_args)
						print(f"{faiss_type} index hash : {faiss_hash}, path : {faiss_index_path}")

						# Build done for one setup keeping vectordb as none
						faiss_entry = FaissEntry(faiss_type, ivf_args, faiss_hash, faiss_index_path)
						manifest = make_build_manifest(
							args.dataset_name,
							args.dataset_size,
							ingestion_entry,
							chunk_entry,
							embedder_args,
							embedding_entry,
							bm25_entry,
							faiss_entry,
							None
						)
						manifests.append(manifest)

				elif faiss_type == constants.FAISS_HNSW_INDEX:
					print(f"\nFAISS indexing : {faiss_type}")
					hnsw_args = make_faiss_hnsw_args(args)
					faiss_index, faiss_hash, faiss_index_path = faiss_indexing.faiss_hnsw_indexing(chunks, embeddings, output_path, hnsw_args)
					print(f"{faiss_type} index hash : {faiss_hash}, path : {faiss_index_path}")

					# Build done for one setup keeping vectordb as none
					faiss_entry = FaissEntry(faiss_type, hnsw_args, faiss_hash, faiss_index_path)
					manifest = make_build_manifest(
						args.dataset_name,
						args.dataset_size,
						ingestion_entry,
						chunk_entry,
						embedder_args,
						embedding_entry,
						bm25_entry,
						faiss_entry,
						None
					)
					manifests.append(manifest)

			# 4c. Vector DB, once per vector DB type
			for vectordb_type in args.vectordb_types:
				if vectordb_type == constants.QDRANT_INDEX:
					print(f"\nVector DB indexing : {vectordb_type}")
					vectordb, vectordb_hash, collection_name = vectordb_indexing.qdrant_indexing(chunks, embeddings, output_path)
					print(f"{vectordb_type} collection hash : {vectordb_hash}, name : {collection_name}")

					# Build done for one setup keeping faiss as none
					vectordb_entry = VectorDbEntry(vectordb_type, qdrant_args, vectordb_hash, collection_name)
					manifest = make_build_manifest(
						args.dataset_name,
						args.dataset_size,
						ingestion_entry,
						chunk_entry,
						embedder_args,
						embedding_entry,
						bm25_entry,
						None,
						vectordb_entry
					)
					manifests.append(manifest)

			e4 = time.time() - s4
			print("\nIndexing time : ", e4)

	# 5. The build manifests, saved once every setup is built
	save_build_manifests(manifests, output_path, args.dataset_name, args.dataset_size)


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
