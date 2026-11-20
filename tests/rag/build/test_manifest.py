"""
test_manifest.py

Tests for rag.build.manifest on the build side: start_build_manifest, save_build_manifest and is_build_complete.
"""

import argparse
import json
import os
import shutil

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance
from qdrant_client.models import VectorParams

from common import paths
from common.args import IngestionArgs
from rag.build import manifest as manifest_module
from rag.build.manifest import is_build_complete
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


def test_start_build_manifest_fills_the_build_hash_and_dataset(tmp_path):
    """
    Input: ingestion args and the parsed build args.
    Output: the started manifest.
    What it does: checks the build hash, dataset and documents folder are filled, with no entries yet.
    """
    ingestion_args = IngestionArgs(dataset_size=2, dataset_name="dataset", input_path="data/", output_path=str(tmp_path), document_type="pdf")
    parsed_args = make_parsed_build_args()

    manifest = start_build_manifest(ingestion_args, "fake-model", parsed_args, "testinputhash")

    assert manifest.build_hash == paths.get_build_hash(parsed_args, "testinputhash")
    assert manifest.input_hash == "testinputhash"
    assert manifest.dataset_name == "dataset"
    assert manifest.dataset_size == 2
    assert manifest.documents_path == os.path.join(str(tmp_path), "dataset", "documents")
    assert manifest.embedding_model == "fake-model"
    assert manifest.chunk_entries == []
    assert manifest.index_entries == []


def test_save_build_manifest_writes_to_the_build_hash_file(tmp_path):
    """
    Input: a started manifest and the dataset's output folder.
    Output: the saved file.
    What it does: checks the manifest is saved to build/manifest_{build hash}.json with its build hash and build time.
    """
    ingestion_args = IngestionArgs(dataset_size=2, dataset_name="dataset", input_path="data/", output_path=str(tmp_path), document_type="pdf")
    parsed_args = make_parsed_build_args()
    output_path = os.path.join(str(tmp_path), "dataset")
    manifest = start_build_manifest(ingestion_args, "fake-model", parsed_args, "testinputhash")

    save_build_manifest(manifest, output_path, parsed_args, "testinputhash")

    manifest_file = os.path.join(output_path, "build", f"manifest_{paths.get_build_hash(parsed_args, 'testinputhash')}.json")
    with open(manifest_file, encoding="utf-8") as f:
        saved_manifest = json.load(f)
    assert saved_manifest["build_hash"] == paths.get_build_hash(parsed_args, "testinputhash")
    assert saved_manifest["input_hash"] == "testinputhash"
    assert saved_manifest["built_at"] != ""
    assert not os.path.exists(manifest_file + ".tmp")


# ---------- is_build_complete ----------


def save_complete_build(tmp_path, with_qdrant):
    """
    Input: tmp_path; with_qdrant, True to add a Qdrant entry named "qdrant_test".
    Output: (output_path, parsed_args), with a saved manifest of input hash "testinputhash" whose outputs all exist on disk.
    What it does: makes the documents list, one chunk and embedding folder and one FAISS folder, and saves a manifest
    listing them.
    """
    ingestion_args = IngestionArgs(dataset_size=2, dataset_name="dataset", input_path="data/", output_path=str(tmp_path), document_type="pdf")
    parsed_args = make_parsed_build_args()
    parsed_args.qdrant_url = "http://qdrant-test:6333"
    output_path = os.path.join(str(tmp_path), "dataset")
    manifest = start_build_manifest(ingestion_args, "fake-model", parsed_args, "testinputhash")

    os.makedirs(manifest.documents_path)
    with open(os.path.join(manifest.documents_path, "doc_ids_testinputhash.json"), "w", encoding="utf-8") as f:
        json.dump(["doc1", "doc2"], f)

    chunk_path = os.path.join(output_path, "chunks", "chunks_test")
    embedding_path = os.path.join(output_path, "embeddings", "embeddings_test")
    faiss_path = os.path.join(output_path, "indexes", "faiss_flatip_test")
    os.makedirs(chunk_path)
    os.makedirs(embedding_path)
    os.makedirs(faiss_path)

    manifest.chunk_entries.append(ChunkEntry("fixed", 2, 6, 2, 0.5, chunk_path, embedding_path, 4))
    manifest.index_entries.append(IndexEntry("fixed", "faiss_flatip", {}, faiss_path, 4))
    if with_qdrant:
        manifest.index_entries.append(IndexEntry("fixed", "qdrant", {}, "qdrant_test", 4))

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


def test_is_build_complete_is_false_when_the_input_files_changed(tmp_path, monkeypatch):
    """
    Input: a complete build of input hash "testinputhash", checked with another input hash.
    Output: False.
    What it does: checks changed input files make the build run again.
    """
    output_path, parsed_args = save_complete_build(tmp_path, with_qdrant=True)
    use_memory_qdrant(monkeypatch, ["qdrant_test"])

    assert is_build_complete(output_path, parsed_args, "otherinputhash") is False


def test_is_build_complete_is_true_when_every_output_exists(tmp_path, monkeypatch):
    """
    Input: a saved manifest whose folders and Qdrant collection all exist.
    Output: True.
    What it does: checks a finished build with the same args is skipped.
    """
    output_path, parsed_args = save_complete_build(tmp_path, with_qdrant=True)
    use_memory_qdrant(monkeypatch, ["qdrant_test"])

    assert is_build_complete(output_path, parsed_args, "testinputhash") is True


@pytest.mark.parametrize("missing_folder", [
    os.path.join("chunks", "chunks_test"),
    os.path.join("embeddings", "embeddings_test"),
    os.path.join("indexes", "faiss_flatip_test"),
    os.path.join("documents", "doc_ids_testinputhash.json"),
])
def test_is_build_complete_is_false_when_one_output_is_missing(tmp_path, monkeypatch, missing_folder):
    """
    Input: a saved manifest with one listed output removed.
    Output: False.
    What it does: checks the build runs again if any one output is missing.
    """
    output_path, parsed_args = save_complete_build(tmp_path, with_qdrant=True)
    use_memory_qdrant(monkeypatch, ["qdrant_test"])
    missing_path = os.path.join(output_path, missing_folder)
    if os.path.isdir(missing_path):
        shutil.rmtree(missing_path)
    else:
        os.remove(missing_path)

    assert is_build_complete(output_path, parsed_args, "testinputhash") is False


def test_is_build_complete_is_false_when_a_qdrant_collection_is_missing(tmp_path, monkeypatch):
    """
    Input: a saved manifest with a Qdrant entry, and a Qdrant server without that collection.
    Output: False.
    What it does: checks a deleted collection makes the build run again.
    """
    output_path, parsed_args = save_complete_build(tmp_path, with_qdrant=True)
    use_memory_qdrant(monkeypatch, [])

    assert is_build_complete(output_path, parsed_args, "testinputhash") is False


def test_is_build_complete_raises_when_qdrant_is_needed_and_not_reachable(tmp_path, monkeypatch):
    """
    Input: a saved manifest with a Qdrant entry, and a Qdrant client that cannot connect.
    Output: ConnectionError.
    What it does: checks the check stops instead of going on when Qdrant is down.
    """
    output_path, parsed_args = save_complete_build(tmp_path, with_qdrant=True)

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


def test_is_build_complete_does_not_connect_to_qdrant_without_qdrant_entries(tmp_path, monkeypatch):
    """
    Input: a saved manifest with no Qdrant entry, and a Qdrant client that fails if built.
    Output: True.
    What it does: checks Qdrant is not touched when the build made no Qdrant collection.
    """
    output_path, parsed_args = save_complete_build(tmp_path, with_qdrant=False)

    def fail_to_connect(url=None):
        """
        Input: url.
        Output: none.
        What it does: fails the test if a Qdrant client is built.
        """
        raise AssertionError("Qdrant was contacted for a build with no Qdrant entries")

    monkeypatch.setattr(manifest_module, "QdrantClient", fail_to_connect)

    assert is_build_complete(output_path, parsed_args, "testinputhash") is True
