import dataclasses
import datetime
import json
import os

from common import constants
from common import paths_temp
from common.args_temp import ChunkingArgs
from common.args_temp import IndexArgs
from common.args_temp import IngestionArgs
from common.args_temp import VectorDbArgs
from schema.index import Bm25Index
from schema.index import FaissIndex
from schema.manifest import BuildManifest
from schema.manifest import ChunkEntry
from schema.manifest import IndexEntry


def get_index_settings(index_args: IndexArgs) -> dict:
	"""
	Input: index_args, the settings of one FAISS or BM25 index.
	Output: the settings of its index type as a dict: empty for faiss_flatip.
	What it does: picks the settings that change the saved index, so the manifest can record and compare them.
	"""
	index_settings = {}
	if index_args.index_type == constants.FAISS_IVF_INDEX:
		index_settings["ivf_nlist"] = index_args.ivf_nlist
	elif index_args.index_type == constants.FAISS_HNSW_INDEX:
		index_settings["hnsw_m"] = index_args.hnsw_m
		index_settings["hnsw_ef_construction"] = index_args.hnsw_ef_construction
	elif index_args.index_type == constants.BM25_INDEX:
		index_settings["bm25_k1"] = index_args.bm25_k1
		index_settings["bm25_b"] = index_args.bm25_b
		index_settings["bm25_tokenizer"] = index_args.bm25_tokenizer
	return index_settings


def start_build_manifest(ingestion_args: IngestionArgs, embedding_model: str) -> BuildManifest:
	"""
	Input: ingestion_args; embedding_model, the chunking model name.
	Output: a BuildManifest with no entries yet.
	What it does: fills in the dataset and the documents folder. The build adds an entry per output as it goes.
	"""
	output_path = paths_temp.get_output_path(ingestion_args.output_path, ingestion_args.dataset_name)
	return BuildManifest(
		dataset_name=ingestion_args.dataset_name,
		dataset_size=ingestion_args.dataset_size,
		documents_path=paths_temp.get_documents_path(output_path),
		embedding_model=embedding_model,
		built_at="",
		chunk_entries=[],
		index_entries=[]
	)


def make_chunk_entry(chunking_args: ChunkingArgs, chunk_path: str, embedding_path: str, chunk_count: int) -> ChunkEntry:
	"""
	Input: chunking_args; chunk_path and embedding_path, the folders the build saved to; chunk_count.
	Output: the ChunkEntry of one chunk type.
	What it does: records the chunk settings with the folders.
	"""
	return ChunkEntry(
		chunk_type=chunking_args.chunk_type,
		min_chunk_size=chunking_args.min_chunk_size,
		max_chunk_size=chunking_args.max_chunk_size,
		fixed_chunk_overlap=chunking_args.fixed_chunk_overlap,
		semantic_threshold=chunking_args.semantic_threshold,
		chunk_path=chunk_path,
		embedding_path=embedding_path,
		chunk_count=chunk_count
	)


def make_index_entry(index_args: IndexArgs, index: FaissIndex | Bm25Index, count: int) -> IndexEntry:
	"""
	Input: index_args, the settings of one FAISS or BM25 index; index, the index the build returned; count, the
	vectors or chunks it holds.
	Output: the IndexEntry of the index.
	What it does: records the index type and its settings with its folder. A BM25 index_path is its pickle file, so
	its folder is taken.
	"""
	index_folder = index.index_path
	if index_args.index_type == constants.BM25_INDEX:
		index_folder = os.path.dirname(index.index_path)

	return IndexEntry(
		chunk_type=index_args.chunking_args.chunk_type,
		index_type=index_args.index_type,
		index_settings=get_index_settings(index_args),
		location=index_folder,
		count=count
	)


def make_vectordb_entry(vectordb_args: VectorDbArgs, collection_name: str, count: int) -> IndexEntry:
	"""
	Input: vectordb_args; collection_name, the collection the build filled; count, the points it holds.
	Output: the IndexEntry of the collection.
	What it does: records the vector DB type with the collection name as its location.
	"""
	return IndexEntry(
		chunk_type=vectordb_args.chunking_args.chunk_type,
		index_type=vectordb_args.vectordb_type,
		index_settings={},
		location=collection_name,
		count=count
	)


def save_build_manifest(manifest: BuildManifest, output_path: str) -> None:
	"""
	Input: manifest; output_path, the dataset's output folder.
	Output: nothing.
	What it does: stamps the build time and saves the manifest as JSON through a temporary file.
	"""
	manifest.built_at = datetime.datetime.now().isoformat(timespec="seconds")

	manifest_file = paths_temp.get_build_manifest_path(output_path)
	os.makedirs(os.path.dirname(manifest_file), exist_ok=True)
	with open(manifest_file + ".tmp", "w", encoding="utf-8") as f:
		json.dump(dataclasses.asdict(manifest), f, indent=2)
	os.replace(manifest_file + ".tmp", manifest_file)
	print(f"Build manifest saved : {manifest_file}")


def load_build_manifest(output_path: str) -> BuildManifest:
	"""
	Input: output_path, the dataset's output folder.
	Output: the saved BuildManifest.
	What it does: reads the manifest JSON, and raises FileNotFoundError if the build has not saved one.
	"""
	manifest_file = paths_temp.get_build_manifest_path(output_path)
	if not os.path.exists(manifest_file):
		raise FileNotFoundError(f"No build manifest at {manifest_file}. Run scripts/build_temp.sh first.")

	with open(manifest_file, encoding="utf-8") as f:
		manifest_dict = json.load(f)

	chunk_entries = []
	for chunk_entry_dict in manifest_dict["chunk_entries"]:
		chunk_entries.append(ChunkEntry(**chunk_entry_dict))
	index_entries = []
	for index_entry_dict in manifest_dict["index_entries"]:
		index_entries.append(IndexEntry(**index_entry_dict))

	manifest_dict["chunk_entries"] = chunk_entries
	manifest_dict["index_entries"] = index_entries
	return BuildManifest(**manifest_dict)


def find_chunk_entry(manifest: BuildManifest, chunking_args: ChunkingArgs) -> ChunkEntry:
	"""
	Input: manifest; chunking_args, the chunk settings the caller needs.
	Output: the ChunkEntry of that chunk type.
	What it does: finds the entry, and raises ValueError if the build has none or built it with other settings.
	"""
	if manifest.embedding_model != chunking_args.model_name:
		raise ValueError(
			f"The build used embedding model {manifest.embedding_model}, not {chunking_args.model_name}. "
			"Run scripts/build_temp.sh with these settings first."
		)

	for chunk_entry in manifest.chunk_entries:
		if chunk_entry.chunk_type != chunking_args.chunk_type:
			continue
		built_settings = (chunk_entry.min_chunk_size, chunk_entry.max_chunk_size, chunk_entry.fixed_chunk_overlap, chunk_entry.semantic_threshold)
		wanted_settings = (chunking_args.min_chunk_size, chunking_args.max_chunk_size, chunking_args.fixed_chunk_overlap, chunking_args.semantic_threshold)
		if built_settings != wanted_settings:
			raise ValueError(
				f"The build made {chunking_args.chunk_type} chunks with settings {built_settings}, not {wanted_settings}. "
				"Run scripts/build_temp.sh with these settings first."
			)
		return chunk_entry

	raise ValueError(f"The build has no {chunking_args.chunk_type} chunks. Run scripts/build_temp.sh with this chunk type first.")


def find_index_entry(manifest: BuildManifest, chunk_type: str, index_type: str, index_settings: dict) -> IndexEntry:
	"""
	Input: manifest; chunk_type; index_type, a FAISS or BM25 index type or a vector DB type; index_settings, the
	settings the caller needs, as get_index_settings gives them ({} for faiss_flatip and qdrant).
	Output: the IndexEntry of that index.
	What it does: finds the entry, and raises ValueError if the build has none or built it with other settings.
	"""
	for index_entry in manifest.index_entries:
		if index_entry.chunk_type != chunk_type or index_entry.index_type != index_type:
			continue
		if index_entry.index_settings != index_settings:
			raise ValueError(
				f"The build made the {chunk_type} {index_type} index with settings {index_entry.index_settings}, not "
				f"{index_settings}. Run scripts/build_temp.sh with these settings first."
			)
		return index_entry

	raise ValueError(f"The build has no {chunk_type} {index_type} index. Run scripts/build_temp.sh with this index type first.")
