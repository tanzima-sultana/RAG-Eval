import dataclasses
import json
import os

from common import paths
from common.args import Bm25IndexingArgs
from common.args import ChunkingArgs
from common.args import EmbedderArgs
from common.args import EmbeddingArgs
from common.args import FaissHNSWArgs
from common.args import FaissIVFArgs
from common.args import IngestionArgs
from common.args import QdrantArgs
from common import constants
from schema.manifest import Bm25Entry
from schema.manifest import BuildManifest
from schema.manifest import ChunkEntry
from schema.manifest import EmbeddingEntry
from schema.manifest import FaissEntry
from schema.manifest import IngestionEntry
from schema.manifest import VectorDbEntry


def make_build_manifest(
	dataset_name: str,
	dataset_size: int,
	ingestion_entry: IngestionEntry,
	chunk_entry: ChunkEntry,
	embedder_args: EmbedderArgs,
	embedding_entry: EmbeddingEntry,
	bm25_entry: Bm25Entry,
	faiss_entry: FaissEntry | None,
	vectordb_entry: VectorDbEntry | None
) -> BuildManifest:
	"""
	Input: dataset_name; dataset_size; the ingestion, chunk, embedding and BM25 entries and the embedder args that every
	setup has; faiss_entry or vectordb_entry, the one this setup has, and None for the other.
	Output: the BuildManifest of one setup.
	What it does: works out the build hash of the setup from the hashes of all its entries, with "" for the index entry
	that is None, and puts everything into one BuildManifest.
	"""
	faiss_hash = ""
	if faiss_entry is not None:
		faiss_hash = faiss_entry.hash

	vectordb_hash = ""
	if vectordb_entry is not None:
		vectordb_hash = vectordb_entry.hash

	build_hash = paths.get_build_hash(
		ingestion_entry.hash,
		chunk_entry.hash,
		embedding_entry.hash,
		bm25_entry.hash,
		faiss_hash,
		vectordb_hash
	)

	build_manifest = BuildManifest(
		dataset_name=dataset_name,
		dataset_size=dataset_size,
		build_hash=build_hash,
		ingestion_entry=ingestion_entry,
		chunk_entry=chunk_entry,
		embedder=embedder_args,
		embedding_entry=embedding_entry,
		bm25_index_entry=bm25_entry,
		faiss_index_entry=faiss_entry,
		vectordb_index_entry=vectordb_entry
	)
	return build_manifest


def save_build_manifests(manifests: list[BuildManifest], output_path: str, dataset_name: str, dataset_size: int) -> str:
	"""
	Input: manifests, the BuildManifest of every setup of the run, in the order they were built; output_path, the
	dataset's output folder; dataset_name; dataset_size.
	Output: the path of the file the manifests were saved to.
	What it does: saves the list of manifests as JSON to the file get_build_manifest_path names. It writes to a temporary
	file first and then moves it into place, so a crash never leaves a half-written file. A file with the same name is
	overwritten.
	"""
	build_hashes = []
	manifest_dicts = []
	for manifest in manifests:
		build_hashes.append(manifest.build_hash)
		manifest_dicts.append(dataclasses.asdict(manifest))

	manifest_path = paths.get_build_manifest_path(output_path, dataset_name, dataset_size, build_hashes)
	os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

	temporary_manifest_path = manifest_path + ".tmp"
	with open(temporary_manifest_path, "w", encoding="utf-8") as f:
		json.dump(manifest_dicts, f, indent=2)
	os.replace(temporary_manifest_path, manifest_path)

	print(f"Build manifests : {len(manifests)} setups, saved to {manifest_path}")
	return manifest_path


def load_build_manifests(manifest_path: str) -> list[BuildManifest]:
	"""
	Input: manifest_path, a file save_build_manifests wrote.
	Output: the list of BuildManifest in it, in the order they were saved.
	What it does: reads the JSON and rebuilds each BuildManifest with its entries and args. The args of a FAISS entry
	are rebuilt from its faiss_type: FaissIVFArgs for faiss_ivf, FaissHNSWArgs for faiss_hnsw, None for faiss_flatip.
	"""
	with open(manifest_path, encoding="utf-8") as f:
		manifest_dicts = json.load(f)

	manifests = []
	for manifest_dict in manifest_dicts:
		ingestion_dict = manifest_dict["ingestion_entry"]
		ingestion_entry = IngestionEntry(
			args=IngestionArgs(**ingestion_dict["args"]),
			hash=ingestion_dict["hash"],
			path=ingestion_dict["path"]
		)

		chunk_dict = manifest_dict["chunk_entry"]
		chunk_entry = ChunkEntry(
			args=ChunkingArgs(**chunk_dict["args"]),
			hash=chunk_dict["hash"],
			path=chunk_dict["path"]
		)

		embedding_dict = manifest_dict["embedding_entry"]
		embedding_entry = EmbeddingEntry(
			args=EmbeddingArgs(
				embedder_args=EmbedderArgs(**embedding_dict["args"]["embedder_args"]),
				chunking_args=ChunkingArgs(**embedding_dict["args"]["chunking_args"])
			),
			hash=embedding_dict["hash"],
			path=embedding_dict["path"]
		)

		bm25_dict = manifest_dict["bm25_index_entry"]
		bm25_entry = Bm25Entry(
			args=Bm25IndexingArgs(**bm25_dict["args"]),
			hash=bm25_dict["hash"],
			path=bm25_dict["path"]
		)

		faiss_entry = None
		faiss_dict = manifest_dict["faiss_index_entry"]
		if faiss_dict is not None:
			faiss_args = None
			if faiss_dict["faiss_type"] == constants.FAISS_IVF_INDEX:
				faiss_args = FaissIVFArgs(**faiss_dict["faiss_args"])
			elif faiss_dict["faiss_type"] == constants.FAISS_HNSW_INDEX:
				faiss_args = FaissHNSWArgs(**faiss_dict["faiss_args"])
			faiss_entry = FaissEntry(
				faiss_type=faiss_dict["faiss_type"],
				faiss_args=faiss_args,
				hash=faiss_dict["hash"],
				path=faiss_dict["path"]
			)

		vectordb_entry = None
		vectordb_dict = manifest_dict["vectordb_index_entry"]
		if vectordb_dict is not None:
			vectordb_entry = VectorDbEntry(
				vectordb_type=vectordb_dict["vectordb_type"],
				args=QdrantArgs(**vectordb_dict["args"]),
				hash=vectordb_dict["hash"],
				path=vectordb_dict["path"]
			)

		manifest = BuildManifest(
			dataset_name=manifest_dict["dataset_name"],
			dataset_size=manifest_dict["dataset_size"],
			build_hash=manifest_dict["build_hash"],
			ingestion_entry=ingestion_entry,
			chunk_entry=chunk_entry,
			embedder=EmbedderArgs(**manifest_dict["embedder"]),
			embedding_entry=embedding_entry,
			bm25_index_entry=bm25_entry,
			faiss_index_entry=faiss_entry,
			vectordb_index_entry=vectordb_entry
		)
		manifests.append(manifest)

	return manifests
