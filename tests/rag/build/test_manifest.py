"""
test_manifest.py

Tests for rag.build.manifest on the build side: start_build_manifest, make_index_entry, make_vectordb_entry,
save_build_manifest, load_build_manifest_from_path and is_build_complete.
"""

import argparse
import json
import os
import shutil
import types

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance
from qdrant_client.models import VectorParams

from common import paths
from common.args import IngestionArgs
from rag.build import manifest as manifest_module
from rag.build.manifest import is_build_complete
from rag.build.manifest import load_build_manifest_from_path
from rag.build.manifest import make_index_entry
from rag.build.manifest import make_vectordb_entry
from rag.build.manifest import save_build_manifest
from rag.build.manifest import start_build_manifest
from schema.manifest import ChunkEntry
from schema.manifest import IndexEntry


def make_parsed_build_args():
    """
    Input: none.
    Output: an argparse.Namespace standing in for the parsed build config.
    What it does: holds a few build args, enough to give a build hash.
    """
    return argparse.Namespace(dataset_size=2, dataset_name="dataset", output_path="out/", model_name="fake-model")


def test_start_build_manifest_fills_the_build_hash_and_dataset(tmp_path, make_embedder_args):
    """
    Input: ingestion args and the parsed build args.
    Output: the started manifest.
    What it does: checks the build hash, dataset and documents folder are filled, with no entries yet.
    """
    ingestion_args = IngestionArgs(dataset_size=2, dataset_name="dataset", input_path="data/", output_path=str(tmp_path), document_type="pdf")
    parsed_args = make_parsed_build_args()
    embedder_args = make_embedder_args()

    manifest = start_build_manifest(ingestion_args, embedder_args, parsed_args, "testinputhash")

    assert manifest.build_hash == paths.get_build_hash(parsed_args, "testinputhash")
    assert manifest.input_hash == "testinputhash"
    assert manifest.dataset_name == "dataset"
    assert manifest.dataset_size == 2
    assert manifest.documents_path == os.path.join(str(tmp_path), "dataset", "documents")
    assert manifest.embedder == make_embedder_args()
    assert manifest.chunk_entries == []
    assert manifest.index_entries == []


def test_save_build_manifest_writes_to_the_build_hash_file(tmp_path, make_embedder_args):
    """
    Input: a started manifest and the dataset's output folder.
    Output: the saved file.
    What it does: checks the manifest is saved to build/manifest_{dataset_name}_{dataset_size}_{build hash}.json with
    its build hash and build time.
    """
    ingestion_args = IngestionArgs(dataset_size=2, dataset_name="dataset", input_path="data/", output_path=str(tmp_path), document_type="pdf")
    parsed_args = make_parsed_build_args()
    output_path = os.path.join(str(tmp_path), "dataset")
    embedder_args = make_embedder_args()
    manifest = start_build_manifest(ingestion_args, embedder_args, parsed_args, "testinputhash")

    save_build_manifest(manifest, output_path, parsed_args, "testinputhash")

    manifest_file = os.path.join(output_path, "build", f"manifest_dataset_2_{paths.get_build_hash(parsed_args, 'testinputhash')}.json")
    with open(manifest_file, encoding="utf-8") as f:
        saved_manifest = json.load(f)
    assert saved_manifest["build_hash"] == paths.get_build_hash(parsed_args, "testinputhash")
    assert saved_manifest["input_hash"] == "testinputhash"
    assert saved_manifest["built_at"] != ""
    assert not os.path.exists(manifest_file + ".tmp")


# ---------- make_index_entry / make_vectordb_entry ----------


def make_built_index(index_path, nlist):
    """
    Input: index_path, where the built index is; nlist, the nlist of its FAISS index, or None for no FAISS index.
    Output: a stand-in for the FaissIndex or Bm25Index the build returns.
    What it does: holds index_path and, when nlist is given, vector_store.index.nlist, the fields make_index_entry reads.
    """
    if nlist is None:
        return types.SimpleNamespace(index_path=index_path)
    faiss_index = types.SimpleNamespace(nlist=nlist)
    vector_store = types.SimpleNamespace(index=faiss_index)
    return types.SimpleNamespace(index_path=index_path, vector_store=vector_store)


def test_make_index_entry_records_the_built_nlist_for_faiss_ivf(make_chunking_args, make_index_args):
    """
    Input: faiss_ivf index args with ivf_nlist_multiplier 4, and a built IVF index with nlist 7.
    Output: the IndexEntry.
    What it does: checks the multiplier goes into index_settings and the built nlist into built_settings.
    """
    index_args = make_index_args(index_type="faiss_ivf", ivf_nlist_multiplier=4)

    index_entry = make_index_entry(make_chunking_args(), index_args, make_built_index("indexes/faiss_ivf_test", 7), 10)

    assert index_entry.chunk_type == "fixed"
    assert index_entry.index_type == "faiss_ivf"
    assert index_entry.index_settings == {"ivf_nlist_multiplier": 4}
    assert index_entry.built_settings == {"nlist": 7}
    assert index_entry.location == "indexes/faiss_ivf_test"
    assert index_entry.count == 10


def test_make_index_entry_has_empty_built_settings_for_other_index_types(make_chunking_args, make_index_args):
    """
    Input: bm25 index args and a built BM25 index whose path is its pickle file.
    Output: the IndexEntry.
    What it does: checks built_settings is empty and the location is the pickle's folder.
    """
    index_args = make_index_args(index_type="bm25")

    index_entry = make_index_entry(make_chunking_args(), index_args, make_built_index("indexes/bm25_test/bm25_retriever.pkl", None), 10)

    assert index_entry.built_settings == {}
    assert index_entry.location == "indexes/bm25_test"


def test_make_vectordb_entry_has_empty_settings(make_chunking_args, make_vectordb_args):
    """
    Input: the vector DB args and a collection name.
    Output: the IndexEntry.
    What it does: checks the collection name is the location and both settings dicts are empty.
    """
    index_entry = make_vectordb_entry(make_chunking_args(), make_vectordb_args(), "qdrant_test", 10)

    assert index_entry.index_type == "qdrant"
    assert index_entry.index_settings == {}
    assert index_entry.built_settings == {}
    assert index_entry.location == "qdrant_test"


# ---------- load_build_manifest_from_path ----------


def test_load_build_manifest_from_path_gives_the_saved_manifest(tmp_path, make_embedder_args):
    """
    Input: a manifest with one chunk and one index entry, saved to its build hash file.
    Output: the manifest loaded from that file's path.
    What it does: checks the loaded manifest equals the saved one, with ChunkEntry and IndexEntry entries.
    """
    ingestion_args = IngestionArgs(dataset_size=2, dataset_name="dataset", input_path="data/", output_path=str(tmp_path), document_type="pdf")
    parsed_args = make_parsed_build_args()
    output_path = os.path.join(str(tmp_path), "dataset")
    embedder_args = make_embedder_args()
    manifest = start_build_manifest(ingestion_args, embedder_args, parsed_args, "testinputhash")
    manifest.chunk_entries.append(ChunkEntry("fixed", 2, 6, 2, "chunks_test", "embeddings_test", 4))
    manifest.index_entries.append(IndexEntry("fixed", "faiss_ivf", {"ivf_nlist_multiplier": 1.0}, {"nlist": 2}, "faiss_ivf_test", 4))
    save_build_manifest(manifest, output_path, parsed_args, "testinputhash")
    manifest_file = paths.get_build_manifest_path(output_path, parsed_args, "testinputhash")

    loaded_manifest = load_build_manifest_from_path(manifest_file)

    assert loaded_manifest == manifest
    assert isinstance(loaded_manifest.chunk_entries[0], ChunkEntry)
    assert isinstance(loaded_manifest.index_entries[0], IndexEntry)


def test_load_build_manifest_from_path_raises_when_the_file_is_missing(tmp_path):
    """
    Input: a path with no manifest file.
    Output: FileNotFoundError.
    What it does: checks a missing manifest is reported with its path.
    """
    with pytest.raises(FileNotFoundError, match="No build manifest"):
        load_build_manifest_from_path(os.path.join(str(tmp_path), "manifest_missing.json"))


# ---------- is_build_complete ----------


def save_complete_build(tmp_path, with_qdrant, embedder_args):
    """
    Input: tmp_path; with_qdrant, True to add a Qdrant entry named "qdrant_test"; embedder_args.
    Output: (output_path, parsed_args), with a saved manifest of input hash "testinputhash" whose outputs all exist on disk.
    What it does: makes the documents list, one chunk and embedding folder and one FAISS folder, and saves a manifest
    listing them.
    """
    ingestion_args = IngestionArgs(dataset_size=2, dataset_name="dataset", input_path="data/", output_path=str(tmp_path), document_type="pdf")
    parsed_args = make_parsed_build_args()
    parsed_args.qdrant_url = "http://qdrant-test:6333"
    output_path = os.path.join(str(tmp_path), "dataset")
    manifest = start_build_manifest(ingestion_args, embedder_args, parsed_args, "testinputhash")

    os.makedirs(manifest.documents_path)
    with open(os.path.join(manifest.documents_path, "doc_ids_testinputhash.json"), "w", encoding="utf-8") as f:
        json.dump(["doc1", "doc2"], f)

    chunk_path = os.path.join(output_path, "chunks", "chunks_test")
    embedding_path = os.path.join(output_path, "embeddings", "embeddings_test")
    faiss_path = os.path.join(output_path, "indexes", "faiss_flatip_test")
    os.makedirs(chunk_path)
    os.makedirs(embedding_path)
    os.makedirs(faiss_path)

    manifest.chunk_entries.append(ChunkEntry("fixed", 2, 6, 2, chunk_path, embedding_path, 4))
    manifest.index_entries.append(IndexEntry("fixed", "faiss_flatip", {}, {}, faiss_path, 4))
    if with_qdrant:
        manifest.index_entries.append(IndexEntry("fixed", "qdrant", {}, {}, "qdrant_test", 4))

    save_build_manifest(manifest, output_path, parsed_args, "testinputhash")
    return output_path, parsed_args


def use_memory_qdrant(monkeypatch, collection_names):
    """
    Input: pytest's monkeypatch fixture; collection_names, the collections the fake server holds.
    Output: nothing.
    What it does: makes is_build_complete connect to an in-memory Qdrant holding those collections.
    """
    qdrant_client = QdrantClient(":memory:")
    for collection_name in collection_names:
        qdrant_client.create_collection(collection_name, vectors_config=VectorParams(size=4, distance=Distance.COSINE))

    def connect_memory_qdrant_client(url=None):
        """
        Input: url, ignored.
        Output: the in-memory client.
        What it does: stands in for QdrantClient(url=...).
        """
        return qdrant_client

    monkeypatch.setattr(manifest_module, "QdrantClient", connect_memory_qdrant_client)


def test_is_build_complete_is_false_without_a_manifest(tmp_path):
    """
    Input: an output folder with no manifest.
    Output: False.
    What it does: checks a build with new args runs.
    """
    assert is_build_complete(os.path.join(str(tmp_path), "dataset"), make_parsed_build_args(), "testinputhash") is False


def test_is_build_complete_is_false_when_the_input_files_changed(tmp_path, monkeypatch, make_embedder_args):
    """
    Input: a complete build of input hash "testinputhash", checked with another input hash.
    Output: False.
    What it does: checks changed input files make the build run again.
    """
    output_path, parsed_args = save_complete_build(tmp_path, with_qdrant=True, embedder_args=make_embedder_args())
    use_memory_qdrant(monkeypatch, ["qdrant_test"])

    assert is_build_complete(output_path, parsed_args, "otherinputhash") is False


def test_is_build_complete_is_true_when_every_output_exists(tmp_path, monkeypatch, make_embedder_args):
    """
    Input: a saved manifest whose folders and Qdrant collection all exist.
    Output: True.
    What it does: checks a finished build with the same args is skipped.
    """
    output_path, parsed_args = save_complete_build(tmp_path, with_qdrant=True, embedder_args=make_embedder_args())
    use_memory_qdrant(monkeypatch, ["qdrant_test"])

    assert is_build_complete(output_path, parsed_args, "testinputhash") is True


@pytest.mark.parametrize("missing_folder", [
    os.path.join("chunks", "chunks_test"),
    os.path.join("embeddings", "embeddings_test"),
    os.path.join("indexes", "faiss_flatip_test"),
    os.path.join("documents", "doc_ids_testinputhash.json"),
])
def test_is_build_complete_is_false_when_one_output_is_missing(tmp_path, monkeypatch, missing_folder, make_embedder_args):
    """
    Input: a saved manifest with one listed output removed.
    Output: False.
    What it does: checks the build runs again if any one output is missing.
    """
    output_path, parsed_args = save_complete_build(tmp_path, with_qdrant=True, embedder_args=make_embedder_args())
    use_memory_qdrant(monkeypatch, ["qdrant_test"])
    missing_path = os.path.join(output_path, missing_folder)
    if os.path.isdir(missing_path):
        shutil.rmtree(missing_path)
    else:
        os.remove(missing_path)

    assert is_build_complete(output_path, parsed_args, "testinputhash") is False


def test_is_build_complete_is_false_when_a_qdrant_collection_is_missing(tmp_path, monkeypatch, make_embedder_args):
    """
    Input: a saved manifest with a Qdrant entry, and a Qdrant server without that collection.
    Output: False.
    What it does: checks a deleted collection makes the build run again.
    """
    output_path, parsed_args = save_complete_build(tmp_path, with_qdrant=True, embedder_args=make_embedder_args())
    use_memory_qdrant(monkeypatch, [])

    assert is_build_complete(output_path, parsed_args, "testinputhash") is False


def test_is_build_complete_raises_when_qdrant_is_needed_and_not_reachable(tmp_path, monkeypatch, make_embedder_args):
    """
    Input: a saved manifest with a Qdrant entry, and a Qdrant client that cannot connect.
    Output: ConnectionError.
    What it does: checks the check stops instead of going on when Qdrant is down.
    """
    output_path, parsed_args = save_complete_build(tmp_path, with_qdrant=True, embedder_args=make_embedder_args())

    class UnreachableQdrantClient:
        def __init__(self, url=None):
            """
            Input: url, ignored.
            Output: nothing.
            What it does: stands in for a client whose server is down.
            """
            self.url = url

        def get_collections(self):
            """
            Input: none.
            Output: none.
            What it does: fails like a refused connection.
            """
            raise OSError("Connection refused")

    monkeypatch.setattr(manifest_module, "QdrantClient", UnreachableQdrantClient)

    with pytest.raises(ConnectionError, match="not reachable"):
        is_build_complete(output_path, parsed_args, "testinputhash")


def test_is_build_complete_does_not_connect_to_qdrant_without_qdrant_entries(tmp_path, monkeypatch, make_embedder_args):
    """
    Input: a saved manifest with no Qdrant entry, and a Qdrant client that fails if built.
    Output: True.
    What it does: checks Qdrant is not touched when the build made no Qdrant collection.
    """
    output_path, parsed_args = save_complete_build(tmp_path, with_qdrant=False, embedder_args=make_embedder_args())

    def fail_to_connect(url=None):
        """
        Input: url.
        Output: none.
        What it does: fails the test if a Qdrant client is built.
        """
        raise AssertionError("Qdrant was contacted for a build with no Qdrant entries")

    monkeypatch.setattr(manifest_module, "QdrantClient", fail_to_connect)

    assert is_build_complete(output_path, parsed_args, "testinputhash") is True
