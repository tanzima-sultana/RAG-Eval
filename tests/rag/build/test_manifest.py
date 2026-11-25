"""
test_manifest.py

Tests for rag.build.manifest on the build side: make_build_manifest, save_build_manifests and load_build_manifests.
"""

import json
import os

import pytest

from common import paths
from common.args import EmbeddingArgs
from common.args import IngestionArgs
from rag.build.manifest import load_build_manifests
from rag.build.manifest import make_build_manifest
from rag.build.manifest import save_build_manifests
from schema.manifest import Bm25Entry
from schema.manifest import BuildManifest
from schema.manifest import ChunkEntry
from schema.manifest import EmbeddingEntry
from schema.manifest import FaissEntry
from schema.manifest import IngestionEntry
from schema.manifest import VectorDbEntry


@pytest.fixture
def make_entries(make_embedder_args, make_chunking_args, make_bm25_indexing_args):
    """
    Input: the make_embedder_args, make_chunking_args and make_bm25_indexing_args fixtures.
    Output: a function that builds the entries every setup has.
    What it does: returns a builder of the ingestion, chunk, embedding and BM25 entries with fixed hashes and paths.
    """
    def build_entries():
        """
        Input: none.
        Output: (ingestion_entry, chunk_entry, embedding_entry, bm25_entry).
        What it does: builds the four entries.
        """
        ingestion_args = IngestionArgs(dataset_size=2, dataset_name="dataset", input_path="data/", output_path="out/", document_type="pdf")
        ingestion_entry = IngestionEntry(ingestion_args, "ingestionhash", "out/dataset/documents")
        chunk_entry = ChunkEntry(make_chunking_args(), "chunkinghash", "out/dataset/chunks/chunks_x")
        embedding_args = EmbeddingArgs(embedder_args=make_embedder_args(), chunking_args=make_chunking_args())
        embedding_entry = EmbeddingEntry(embedding_args, "embeddinghash", "out/dataset/embeddings/embeddings_x")
        bm25_entry = Bm25Entry(make_bm25_indexing_args(), "bm25hash", "out/dataset/indexes/bm25_x")
        return ingestion_entry, chunk_entry, embedding_entry, bm25_entry

    return build_entries


@pytest.fixture
def make_manifest(make_entries, make_embedder_args):
    """
    Input: the make_entries and make_embedder_args fixtures.
    Output: a function that builds one BuildManifest from a FAISS entry or a vector DB entry.
    What it does: returns a builder that calls make_build_manifest with the default entries.
    """
    def build_manifest(faiss_entry, vectordb_entry):
        """
        Input: faiss_entry, a FaissEntry or None; vectordb_entry, a VectorDbEntry or None.
        Output: the BuildManifest of that setup.
        What it does: calls make_build_manifest.
        """
        ingestion_entry, chunk_entry, embedding_entry, bm25_entry = make_entries()
        return make_build_manifest(
            "dataset",
            2,
            ingestion_entry,
            chunk_entry,
            make_embedder_args(),
            embedding_entry,
            bm25_entry,
            faiss_entry,
            vectordb_entry
        )

    return build_manifest


# ---------- make_build_manifest ----------


def test_make_build_manifest_for_a_faiss_setup(make_manifest, make_faiss_ivf_args):
    """
    Input: a FAISS IVF entry and no vector DB entry.
    Output: the BuildManifest.
    What it does: checks the vector DB entry is None, the FAISS entry is kept, and the build hash is the one
    get_build_hash gives with an empty vector DB hash.
    """
    faiss_entry = FaissEntry("faiss_ivf", make_faiss_ivf_args(), "faisshash", "out/dataset/indexes/faiss_ivf_x")

    manifest = make_manifest(faiss_entry, None)

    assert manifest.vectordb_index_entry is None
    assert manifest.faiss_index_entry == faiss_entry
    assert manifest.build_hash == paths.get_build_hash("ingestionhash", "chunkinghash", "embeddinghash", "bm25hash", "faisshash", "")
    assert manifest.dataset_name == "dataset"
    assert manifest.dataset_size == 2


def test_make_build_manifest_for_a_vector_db_setup(make_manifest, make_qdrant_args):
    """
    Input: a vector DB entry and no FAISS entry.
    Output: the BuildManifest.
    What it does: checks the FAISS entry is None, the vector DB entry is kept, and the build hash is the one
    get_build_hash gives with an empty FAISS hash.
    """
    vectordb_entry = VectorDbEntry("qdrant", make_qdrant_args(), "vectordbhash", "qdrant_x")

    manifest = make_manifest(None, vectordb_entry)

    assert manifest.faiss_index_entry is None
    assert manifest.vectordb_index_entry == vectordb_entry
    assert manifest.build_hash == paths.get_build_hash("ingestionhash", "chunkinghash", "embeddinghash", "bm25hash", "", "vectordbhash")


def test_make_build_manifest_gives_each_setup_its_own_build_hash(make_manifest, make_faiss_ivf_args, make_faiss_hnsw_args):
    """
    Input: two FAISS setups of the same chunk set, with different index hashes.
    Output: the two build hashes.
    What it does: checks each setup has its own build hash.
    """
    ivf_manifest = make_manifest(FaissEntry("faiss_ivf", make_faiss_ivf_args(), "ivfhash", "path_a"), None)
    hnsw_manifest = make_manifest(FaissEntry("faiss_hnsw", make_faiss_hnsw_args(), "hnswhash", "path_b"), None)

    assert ivf_manifest.build_hash != hnsw_manifest.build_hash


# ---------- save_build_manifests ----------


def test_save_build_manifests_writes_one_file_named_by_dataset_size_and_the_build_hashes(make_manifest, make_faiss_hnsw_args, make_qdrant_args, tmp_path):
    """
    Input: the manifests of a FAISS setup and a vector DB setup.
    Output: the saved file.
    What it does: checks one file is written to the path get_build_manifest_path gives, and it holds both manifests in
    order.
    """
    faiss_manifest = make_manifest(FaissEntry("faiss_hnsw", make_faiss_hnsw_args(), "hnswhash", "path_a"), None)
    vectordb_manifest = make_manifest(None, VectorDbEntry("qdrant", make_qdrant_args(), "vectordbhash", "qdrant_x"))

    manifest_path = save_build_manifests([faiss_manifest, vectordb_manifest], str(tmp_path), "dataset", 2)

    expected_path = paths.get_build_manifest_path(str(tmp_path), "dataset", 2, [faiss_manifest.build_hash, vectordb_manifest.build_hash])
    assert manifest_path == expected_path
    assert os.listdir(os.path.join(str(tmp_path), "build")) == [os.path.basename(expected_path)]
    with open(manifest_path, encoding="utf-8") as f:
        saved_manifests = json.load(f)
    assert [saved["build_hash"] for saved in saved_manifests] == [faiss_manifest.build_hash, vectordb_manifest.build_hash]
    assert saved_manifests[0]["vectordb_index_entry"] is None
    assert saved_manifests[1]["faiss_index_entry"] is None


def test_save_build_manifests_overwrites_a_file_with_the_same_name(make_manifest, make_faiss_hnsw_args, tmp_path):
    """
    Input: the same manifests saved twice.
    Output: the files in the build folder.
    What it does: checks the second save writes to the same file, and no temporary file is left.
    """
    manifest = make_manifest(FaissEntry("faiss_hnsw", make_faiss_hnsw_args(), "hnswhash", "path_a"), None)

    first_path = save_build_manifests([manifest], str(tmp_path), "dataset", 2)
    second_path = save_build_manifests([manifest], str(tmp_path), "dataset", 2)

    assert first_path == second_path
    assert os.listdir(os.path.join(str(tmp_path), "build")) == [os.path.basename(first_path)]


def test_save_build_manifests_gives_another_set_of_setups_another_file(make_manifest, make_faiss_ivf_args, make_faiss_hnsw_args, tmp_path):
    """
    Input: one manifest saved, then two manifests saved.
    Output: the files in the build folder.
    What it does: checks a different set of setups is saved to a different file, and the first file stays.
    """
    ivf_manifest = make_manifest(FaissEntry("faiss_ivf", make_faiss_ivf_args(), "ivfhash", "path_a"), None)
    hnsw_manifest = make_manifest(FaissEntry("faiss_hnsw", make_faiss_hnsw_args(), "hnswhash", "path_b"), None)

    save_build_manifests([ivf_manifest], str(tmp_path), "dataset", 2)
    save_build_manifests([ivf_manifest, hnsw_manifest], str(tmp_path), "dataset", 2)

    assert len(os.listdir(os.path.join(str(tmp_path), "build"))) == 2


# ---------- load_build_manifests ----------


def test_load_build_manifests_gives_back_the_saved_manifests(make_manifest, make_faiss_ivf_args, make_faiss_hnsw_args, make_qdrant_args, tmp_path):
    """
    Input: the manifests of an IVF setup, an HNSW setup, a flat setup and a vector DB setup, saved and loaded.
    Output: the loaded manifests.
    What it does: checks every manifest is equal to the saved one, so the entries and their args come back as their own
    classes, with None where there is no FAISS args or no entry.
    """
    manifests = []
    manifests.append(make_manifest(FaissEntry("faiss_ivf", make_faiss_ivf_args(), "ivfhash", "path_a"), None))
    manifests.append(make_manifest(FaissEntry("faiss_hnsw", make_faiss_hnsw_args(), "hnswhash", "path_b"), None))
    manifests.append(make_manifest(FaissEntry("faiss_flatip", None, "fliphash", "path_c"), None))
    manifests.append(make_manifest(None, VectorDbEntry("qdrant", make_qdrant_args(), "vectordbhash", "qdrant_x")))
    manifest_path = save_build_manifests(manifests, str(tmp_path), "dataset", 2)

    loaded_manifests = load_build_manifests(manifest_path)

    assert loaded_manifests == manifests
    assert isinstance(loaded_manifests[0], BuildManifest)
    assert loaded_manifests[2].faiss_index_entry.faiss_args is None
    assert loaded_manifests[3].faiss_index_entry is None
